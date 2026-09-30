"""Baseline interactive search: dual-encoder fusion, tar image reader, remote encoder client."""

from __future__ import annotations

import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("safetensors.numpy")

from conftest import VIDEOS

from multimedia_video_rag.retrieval.build import build_index
from multimedia_video_rag.retrieval.keyframe_images import KeyframeImages
from multimedia_video_rag.retrieval.remote_encoder import (
    RemoteEncoder,
    RemoteEncoderError,
    check_models,
)
from multimedia_video_rag.retrieval.search import RetrievalIndex
from multimedia_video_rag.retrieval.system import (
    FUSIONS,
    DenseScorer,
    dual_encoder_search,
    video_context,
)


@pytest.fixture
def index(cache: Path, tmp_path: Path) -> RetrievalIndex:
    build_index(cache, tmp_path / "index")
    return RetrievalIndex(tmp_path / "index")


def frame_vectors(cache: Path, video_id: str, position: int) -> dict[str, np.ndarray]:
    level, _ = VIDEOS[video_id]
    return {
        module: np.load(cache / module / "data" / level / video_id / "expected.npy")[position]
        for module in ("siglip", "beit3")
    }


@pytest.mark.parametrize("fusion", FUSIONS)
def test_frame_vectors_find_their_own_frame(index: RetrievalIndex, cache: Path, fusion: str):
    results = dual_encoder_search(
        index, frame_vectors(cache, "L22_V004", 1), fusion=fusion, collapse_shots=False
    )
    top = results[0]
    assert index.frames.iloc[top.frame_row].video_id == "L22_V004"
    assert top.frame_row == 4  # rows: L21_V001 -> 0..2, L22_V004 -> 3..4
    assert top.rank == {"siglip": 1, "beit3": 1}
    assert top.similarity["siglip"] == pytest.approx(1.0, abs=1e-3)
    assert len(results) == len(index.frames)  # tiny corpus: every frame is a candidate


def test_every_candidate_gets_both_scores(index: RetrievalIndex, cache: Path):
    vectors = frame_vectors(cache, "L21_V001", 0)
    results = dual_encoder_search(index, vectors, fusion="max-norm", collapse_shots=False)
    for result in results:
        stored = {
            m: index.reconstruct(m, np.array([result.frame_row]))[0] @ vectors[m]
            for m in ("siglip", "beit3")
        }
        for module, value in stored.items():
            assert result.similarity[module] == pytest.approx(float(value), abs=1e-5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_weight_one_rrf_follows_siglip(index: RetrievalIndex, cache: Path):
    vectors = frame_vectors(cache, "L21_V001", 2)
    rrf = dual_encoder_search(index, vectors, fusion="rrf", weight_siglip=1.0, collapse_shots=False)
    siglip = dual_encoder_search(index, vectors, fusion="siglip", collapse_shots=False)
    assert [r.frame_row for r in rrf] == [r.frame_row for r in siglip]


def test_collapse_keeps_one_frame_per_shot(index: RetrievalIndex, cache: Path):
    results = dual_encoder_search(index, frame_vectors(cache, "L21_V001", 0))
    keys = [
        (index.frames.iloc[r.frame_row].video_id, int(index.frames.iloc[r.frame_row].shot_id))
        for r in results
    ]
    assert len(keys) == len(set(keys))


def test_rejects_bad_arguments(index: RetrievalIndex, cache: Path):
    vectors = frame_vectors(cache, "L21_V001", 0)
    with pytest.raises(ValueError):
        dual_encoder_search(index, vectors, fusion="sum")
    with pytest.raises(ValueError):
        dual_encoder_search(index, vectors, weight_siglip=1.5)


@pytest.mark.parametrize("use_torch", [True, False])
@pytest.mark.parametrize("fusion", FUSIONS)
def test_dense_scorer_matches_faiss(
    index: RetrievalIndex, cache: Path, tmp_path: Path, fusion: str, use_torch: bool
):
    vectors = frame_vectors(cache, "L21_V001", 1)
    expected = dual_encoder_search(index, vectors, fusion=fusion, pool_k=3, collapse_shots=False)
    scorer = DenseScorer(index, tmp_path / "matrices")
    if not use_torch:
        scorer._tensors = None
    elif scorer._tensors is None:
        pytest.skip("torch not installed")
    got = dual_encoder_search(
        index, vectors, fusion=fusion, pool_k=3, collapse_shots=False, scorer=scorer
    )
    assert [r.frame_row for r in got] == [r.frame_row for r in expected]
    for a, b in zip(got, expected, strict=True):
        assert a.rank == b.rank
        assert a.similarity["beit3"] == pytest.approx(b.similarity["beit3"], abs=1e-3)


def test_dense_scorer_reuses_cache(index: RetrievalIndex, tmp_path: Path):
    first = DenseScorer(index, tmp_path / "matrices")
    np.save(tmp_path / "matrices" / "siglip.fp16.npy", first.matrices["siglip"] * 0)
    second = DenseScorer(index, tmp_path / "matrices")
    assert not second.matrices["siglip"].any()  # loaded from cache, not rebuilt


def test_video_context_stays_in_video(index: RetrievalIndex):
    assert list(video_context(index, 3, radius=5)) == [3, 4]
    assert list(video_context(index, 1, radius=1)) == [0, 1, 2]


def test_keyframe_images_read_from_tar(index: RetrievalIndex, tmp_path: Path):
    root = tmp_path / "keyframe"
    directory = root / "data" / "L22" / "L22_V004"
    directory.mkdir(parents=True)
    payloads = {"L22_V004/0001.jpg": b"first-image", "L22_V004/0002.jpg": b"second"}
    with tarfile.open(directory / "keyframes.tar", "w") as archive:
        for name, data in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    images = KeyframeImages(root, index.frames)
    assert images.read(3) == b"first-image"
    assert images.read(4) == b"second"
    assert images.available("L22_V004", "L22")
    with pytest.raises(FileNotFoundError):
        images.read(0)  # L21_V001 not downloaded


class StubEncoder(BaseHTTPRequestHandler):
    manifest: ClassVar[dict] = {}
    key = "secret"
    revision = "rev1"

    def log_message(self, *args) -> None:
        pass

    def reply(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def meta(self) -> dict:
        models = {
            m: {
                "model_id": f"org/{m}",
                "model_revision": self.revision,
                "dimension": e["dimension"],
            }
            for m, e in self.manifest["faiss"].items()
        }
        return {"server_version": 1, "device": "cpu", "models": models}

    def do_GET(self) -> None:
        if self.headers.get("X-API-Key") != self.key:
            return self.reply({"detail": "invalid X-API-Key"}, 401)
        self.reply(self.meta())

    def do_POST(self) -> None:
        # Read the body before answering, or the client may see a connection reset.
        texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
        if self.headers.get("X-API-Key") != self.key:
            return self.reply({"detail": "invalid X-API-Key"}, 401)
        payload = self.meta()
        for module, entry in self.manifest["faiss"].items():
            vector = np.zeros(entry["dimension"])
            vector[0] = 1.0
            payload[module] = [vector.tolist() for _ in texts]
        self.reply(payload)


@pytest.fixture
def stub_server(index: RetrievalIndex):
    StubEncoder.manifest = index.manifest
    StubEncoder.revision = "rev1"
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubEncoder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_remote_encoder_round_trip(index: RetrievalIndex, stub_server: str):
    encoder = RemoteEncoder(stub_server, "secret", index.manifest)
    assert encoder.health()["device"] == "cpu"
    vectors = encoder.encode("  a  dog ")
    assert set(vectors) == {"siglip", "beit3"}
    assert vectors["siglip"].shape == (index.manifest["faiss"]["siglip"]["dimension"],)
    assert encoder.encode("a dog") is vectors  # whitespace-normalized cache hit


def test_remote_encoder_rejects_wrong_key(index: RetrievalIndex, stub_server: str):
    with pytest.raises(RemoteEncoderError, match="401"):
        RemoteEncoder(stub_server, "wrong", index.manifest).encode("a dog")


def test_remote_encoder_rejects_other_revision(index: RetrievalIndex, stub_server: str):
    StubEncoder.revision = "rev2"
    with pytest.raises(RemoteEncoderError, match="revision"):
        RemoteEncoder(stub_server, "secret", index.manifest).encode("a dog")


def test_check_models_rejects_unknown_server_version(index: RetrievalIndex):
    with pytest.raises(RemoteEncoderError):
        check_models({"server_version": 99, "models": {}}, index.manifest)
