"""Build a tiny index from synthetic artifacts and query it back."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from conftest import VIDEOS  # noqa: E402

from multimedia_video_rag.ingestion.schemas import SchemaError  # noqa: E402
from multimedia_video_rag.retrieval.build import VISUAL_MODULES, build_index  # noqa: E402
from multimedia_video_rag.retrieval.sources import allow_patterns  # noqa: E402


def test_build_publishes_aligned_indexes(cache: Path, tmp_path: Path):
    output = tmp_path / "index"
    manifest = build_index(cache, output, sources={"keyframe": {"revision": "abc"}})

    assert manifest["frame_count"] == 5
    assert manifest["video_count"] == 2
    assert manifest["caption_count"] == 5
    assert manifest["faiss"]["siglip"]["model_ids"] == ["org/siglip"]
    assert manifest["faiss"]["beit3"]["model_revisions"] == ["rev1"]
    assert not output.with_name("index.partial").exists()

    connection = sqlite3.connect(output / "metadata.sqlite")
    frames = pd.read_sql("SELECT * FROM frames ORDER BY frame_row", connection)
    assert frames["frame_uid"].tolist() == [
        "L21_V001:0", "L21_V001:30", "L21_V001:90", "L22_V004:5", "L22_V004:60",
    ]  # fmt: skip

    for module in VISUAL_MODULES:
        index = faiss.read_index(str(output / f"{module}.faiss"))
        assert index.ntotal == 5
        for frame_row, uid in enumerate(frames["frame_uid"]):
            video_id = uid.split(":")[0]
            level, frame_indices = VIDEOS[video_id]
            position = frame_indices.index(int(uid.split(":")[1]))
            expected = np.load(cache / module / "data" / level / video_id / "expected.npy")
            query = expected[position : position + 1].astype(np.float32)
            scores, ids = index.search(query, 1)
            assert ids[0, 0] == frame_row
            assert scores[0, 0] == pytest.approx(1.0, abs=1e-2)

    captions = connection.execute(
        "SELECT frame_row, caption_en FROM captions ORDER BY frame_row"
    ).fetchall()
    assert captions[3] == (3, "a red bus on street L22_V004 0")
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert not {"od_detections", "asr_segments", "captions_fts"} & tables
    connection.close()


def test_video_subset(cache: Path, tmp_path: Path):
    manifest = build_index(cache, tmp_path / "subset", video_ids=["L22_V004"])
    assert manifest["frame_count"] == 2


def test_missing_marker_blocks_build(cache: Path, tmp_path: Path):
    (cache / "caption" / "data" / "L22" / "L22_V004" / "_CAPTION_SUCCESS.json").unlink()
    with pytest.raises(SchemaError, match="caption: 1 video"):
        build_index(cache, tmp_path / "index")
    assert not (tmp_path / "index").exists()


def test_frame_mismatch_blocks_build(cache: Path, tmp_path: Path):
    path = cache / "caption" / "data" / "L21" / "L21_V001" / "captions.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path)
    with pytest.raises(SchemaError, match="frame_uid mismatch"):
        build_index(cache, tmp_path / "index")


def test_unnormalized_embeddings_block_build(cache: Path):
    path = cache / "siglip" / "data" / "L21" / "L21_V001" / "embeddings.safetensors"
    tensor = safetensors_numpy.load_file(str(path))["embeddings"]
    safetensors_numpy.save_file({"embeddings": tensor * 2}, str(path))
    with pytest.raises(SchemaError, match="not L2-normalized"):
        build_index(cache, cache.parent / "index")


def test_existing_output_requires_overwrite(cache: Path, tmp_path: Path):
    output = tmp_path / "index"
    build_index(cache, output)
    with pytest.raises(FileExistsError):
        build_index(cache, output)
    build_index(cache, output, overwrite=True)


def test_download_patterns_exclude_keyframe_images():
    patterns = allow_patterns("keyframe")
    assert "data/*/*/frames.parquet" in patterns
    assert not any("keyframes.tar" in pattern for pattern in patterns)
    assert allow_patterns("caption", ["L21_V001"]) == [
        "data/*/L21_V001/captions.parquet",
        "data/*/L21_V001/_CAPTION_SUCCESS.json",
    ]
