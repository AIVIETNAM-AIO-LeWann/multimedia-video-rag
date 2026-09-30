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
    assert manifest["asr_chunk_count"] == 2
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

    hits = connection.execute(
        "SELECT rowid FROM captions_fts WHERE captions_fts MATCH ? ORDER BY rank",
        ["bus AND v004"],
    ).fetchall()
    assert {row[0] for row in hits} == {3, 4}

    accented = connection.execute(
        "SELECT video_id FROM asr_segments JOIN asr_fts ON asr_fts.rowid = segment_row "
        "WHERE asr_fts MATCH ?",
        ['"việt nam"'],
    ).fetchall()
    plain = connection.execute(
        "SELECT rowid FROM asr_fts WHERE asr_fts MATCH ?", ['"viet nam"']
    ).fetchall()
    assert len(accented) == 2
    assert len(plain) == 2
    chunk_hits = connection.execute(
        "SELECT c.chunk_id FROM asr_chunks c JOIN asr_chunks_fts f ON f.rowid = c.chunk_row "
        "WHERE asr_chunks_fts MATCH ? ORDER BY c.chunk_row",
        ['"thoi su"'],
    ).fetchall()
    assert [row[0] for row in chunk_hits] == [
        "L21_V001__asr_chunk_00000",
        "L22_V004__asr_chunk_00000",
    ]

    left = connection.execute(
        "SELECT frame_row FROM od_detections WHERE label_en = 'car' "
        "AND position_horizontal = 'left' ORDER BY frame_row"
    ).fetchall()
    assert left == [(0,), (3,)]

    cars = connection.execute(
        "SELECT frame_row, object_count, max_confidence FROM od_frame_labels "
        "WHERE label_en = 'car' ORDER BY frame_row"
    ).fetchall()
    assert cars == [(0, 2, 0.9), (3, 2, 0.9)]
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
    assert allow_patterns("asr", ["L21_V001"]) == [
        "data/*/L21_V001/asr.parquet",
        "data/*/L21_V001/asr_chunks.parquet",
        "data/*/L21_V001/_ASR_SUCCESS.json",
    ]


def test_speech_without_chunks_blocks_build(cache: Path, tmp_path: Path):
    path = cache / "asr" / "data" / "L21" / "L21_V001" / "asr_chunks.parquet"
    pd.read_parquet(path).iloc[:0].to_parquet(path)
    with pytest.raises(SchemaError, match="segments and videos with chunks differ"):
        build_index(cache, tmp_path / "index")
