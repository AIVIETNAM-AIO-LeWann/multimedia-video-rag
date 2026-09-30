"""Visual search and helpers on the synthetic index (see conftest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("safetensors.numpy")

from conftest import VIDEOS

from multimedia_video_rag.retrieval.build import build_index
from multimedia_video_rag.retrieval.encoders import beit3_text_inputs
from multimedia_video_rag.retrieval.search import Hit, RetrievalIndex


@pytest.fixture
def index(cache: Path, tmp_path: Path) -> RetrievalIndex:
    build_index(cache, tmp_path / "index")
    return RetrievalIndex(tmp_path / "index")


def test_visual_search_finds_the_encoded_frame(index: RetrievalIndex, cache: Path):
    level, _ = VIDEOS["L22_V004"]
    query = np.load(cache / "beit3" / "data" / level / "L22_V004" / "expected.npy")[1]
    hits = index.visual("beit3", query.astype(np.float32), k=3)
    assert hits[0].frame_row == 4
    assert hits[0].score == pytest.approx(1.0, abs=1e-2)
    batch = index.visual_batch("beit3", np.stack([query, query]).astype(np.float32), k=2)
    assert [h.frame_row for h in batch[1]] == [h.frame_row for h in hits[:2]]
    with pytest.raises(ValueError):
        index.visual("beit3", np.ones(5, dtype=np.float32))


def test_describe_adds_captions_for_display(index: RetrievalIndex):
    info = index.describe([4, 0])
    assert info["frame_uid"].tolist() == ["L22_V004:60", "L21_V001:0"]
    assert info["caption_en"].tolist() == [
        "a red bus on street L22_V004 1",
        "a red bus on street L21_V001 0",
    ]
    assert index.describe([]).empty


def test_beit3_text_inputs_mirror_unilm():
    tokens_out, padding = beit3_text_inputs([5, 6, 7], bos=0, eos=2, pad=1, max_length=8)
    assert tokens_out == [0, 5, 6, 7, 2, 1, 1, 1]
    assert padding == [0, 0, 0, 0, 0, 1, 1, 1]
    long_tokens, _ = beit3_text_inputs(list(range(10, 30)), bos=0, eos=2, pad=1, max_length=8)
    assert long_tokens == [0, 10, 11, 12, 13, 14, 15, 2]
    with pytest.raises(ValueError):
        beit3_text_inputs([], bos=0, eos=2, pad=1)


def test_neighbor_aggregate_uses_same_video_window(index: RetrievalIndex, cache: Path):
    from multimedia_video_rag.retrieval.rerank import neighbor_aggregate

    level, _ = VIDEOS["L21_V001"]
    query = np.load(cache / "siglip" / "data" / level / "L21_V001" / "expected.npy")[1]
    query = query.astype(np.float32)
    vectors = index.reconstruct("siglip", np.arange(5))
    sims = vectors @ query
    hits = [Hit(row, float(sims[row])) for row in range(5)]
    reranked = {
        hit.frame_row: hit.score for hit in neighbor_aggregate(index, "siglip", query, hits)
    }
    # Rows 0-2 are L21_V001, rows 3-4 are L22_V004: the window never crosses videos.
    assert reranked[2] == pytest.approx(sims[[0, 1, 2]].mean(), abs=1e-5)
    assert reranked[3] == pytest.approx(sims[[3, 4]].mean(), abs=1e-5)
    narrow = neighbor_aggregate(index, "siglip", query, hits, window=0)
    assert narrow[0].frame_row == 1
    assert neighbor_aggregate(index, "siglip", query, []) == []


def test_weighted_max_norm():
    from multimedia_video_rag.retrieval.fusion import weighted_max_norm

    rankings = {"a": [Hit(1, 0.8), Hit(2, 0.4)], "b": [Hit(2, 0.5), Hit(3, 0.25)]}
    fused = weighted_max_norm(rankings, {"a": 1.0, "b": 1.0})
    assert [hit.frame_row for hit in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(0.4 / 0.8 + 1.0)
    assert fused[0].ranks == {"a": 2, "b": 1}
    assert weighted_max_norm({"a": [Hit(1, -0.1)]}) == []
