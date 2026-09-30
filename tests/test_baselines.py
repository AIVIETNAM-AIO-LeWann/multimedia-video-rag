"""Paper baseline (arXiv:2504.08384) on the synthetic index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("safetensors.numpy")

from conftest import VIDEOS

from multimedia_video_rag.retrieval.baselines import (
    dual_query_frame_pair,
    paper_2504_08384_search,
)
from multimedia_video_rag.retrieval.build import build_index
from multimedia_video_rag.retrieval.rerank import neighbor_aggregate
from multimedia_video_rag.retrieval.search import Hit, RetrievalIndex


@pytest.fixture
def index(cache: Path, tmp_path: Path) -> RetrievalIndex:
    build_index(cache, tmp_path / "index")
    return RetrievalIndex(tmp_path / "index")


def frame_vector(cache: Path, module: str, video_id: str, position: int) -> np.ndarray:
    level, _ = VIDEOS[video_id]
    vectors = np.load(cache / module / "data" / level / video_id / "expected.npy")
    return vectors[position].astype(np.float32)


def test_sum_reduce_follows_algorithm_2(index: RetrievalIndex, cache: Path):
    query = frame_vector(cache, "siglip", "L21_V001", 1)
    sims = index.reconstruct("siglip", np.arange(5)) @ query
    hits = [Hit(row, float(sims[row])) for row in range(5)]
    summed = {
        h.frame_row: h.score for h in neighbor_aggregate(index, "siglip", query, hits, reduce="sum")
    }
    assert summed[0] == pytest.approx(sims[[0, 1, 2]].sum(), abs=1e-5)
    assert summed[4] == pytest.approx(sims[[3, 4]].sum(), abs=1e-5)
    with pytest.raises(ValueError):
        neighbor_aggregate(index, "siglip", query, hits, reduce="max")


def test_paper_search_uses_both_encoders(index: RetrievalIndex, cache: Path):
    vectors = {
        "siglip": frame_vector(cache, "siglip", "L22_V004", 0),
        "beit3": frame_vector(cache, "beit3", "L22_V004", 0),
    }
    fused = paper_2504_08384_search(index, vectors, top_m=3)
    assert fused[0].frame_row == 3
    assert set(fused[0].ranks) == {"siglip", "beit3"}
    only_siglip = paper_2504_08384_search(index, {"siglip": vectors["siglip"], "beit3": None})
    assert all(set(hit.ranks) == {"siglip"} for hit in only_siglip)


def test_dual_query_finds_start_and_end(index: RetrievalIndex, cache: Path):
    start = {m: frame_vector(cache, m, "L21_V001", 0) for m in ("siglip", "beit3")}
    end = {m: frame_vector(cache, m, "L21_V001", 2) for m in ("siglip", "beit3")}
    pair = dual_query_frame_pair(index, 1, start, end, stop_quantile=0.0)
    assert (pair.start_row, pair.end_row) == (0, 2)
    narrow = dual_query_frame_pair(index, 1, start, end, stop_quantile=0.0, max_gap=1)
    assert narrow.end_row - narrow.start_row <= 1


def test_dual_query_stays_inside_pivot_video(index: RetrievalIndex, cache: Path):
    start = {"siglip": frame_vector(cache, "siglip", "L21_V001", 2)}
    end = {"siglip": frame_vector(cache, "siglip", "L21_V001", 0)}
    pair = dual_query_frame_pair(index, 3, start, end, stop_quantile=0.0)
    assert {pair.start_row, pair.end_row} <= {3, 4}
    assert pair.start_row <= 3 <= pair.end_row


def test_grab_superglobal_ranks_exact_match_first(index: RetrievalIndex, cache: Path):
    from multimedia_video_rag.retrieval.baselines import grab_search

    query = frame_vector(cache, "beit3", "L22_V004", 1)
    hits = grab_search(index, query, top_m=5, refine_k=1, expand_k=2)
    assert hits[0].frame_row == 4
    assert len(hits) == 5
    assert all(set(hit.ranks) == {"beit3"} for hit in hits)


def test_abts_picks_start_before_and_end_after_pivot(index: RetrievalIndex, cache: Path):
    from multimedia_video_rag.retrieval.baselines import abts_frame_pair

    start = frame_vector(cache, "beit3", "L21_V001", 0)  # t = 0.0 s
    end = frame_vector(cache, "beit3", "L21_V001", 2)  # t = 3.6 s
    pair = abts_frame_pair(index, 1, start, end, lambda_t=0.0)  # pivot t = 1.2 s
    assert (pair.start_row, pair.end_row) == (0, 2)
    narrow = abts_frame_pair(index, 1, start, end, windows_sec=(1.0,), lambda_t=0.0)
    assert (narrow.start_row, narrow.end_row) == (1, 1)  # neighbours fall outside 1 s
    other_video = abts_frame_pair(index, 3, start, end, lambda_t=0.0)
    assert {other_video.start_row, other_video.end_row} <= {3, 4}
