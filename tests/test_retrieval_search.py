"""Branch search, fusion, configurations and evaluation on a tiny synthetic index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")
pytest.importorskip("safetensors.numpy")

from conftest import VIDEOS

from multimedia_video_rag.retrieval.build import build_index
from multimedia_video_rag.retrieval.encoders import beit3_text_inputs
from multimedia_video_rag.retrieval.evaluation import (
    first_relevant_rank,
    load_queries,
    summarize,
)
from multimedia_video_rag.retrieval.fusion import (
    FusedHit,
    collapse_by_shot,
    weighted_rrf,
)
from multimedia_video_rag.retrieval.pipeline import (
    CONFIGS,
    Query,
    run_config,
    search_all,
)
from multimedia_video_rag.retrieval.search import (
    Hit,
    ObjectConstraint,
    RetrievalIndex,
    parse_objects,
)
from multimedia_video_rag.retrieval.text import fts_or_query, remove_accents, tokens


def write_chunk(cache: Path, level: str, video_id: str, text: str, no_accent: str) -> None:
    path = cache / "asr" / "data" / level / video_id / "asr_chunks.parquet"
    chunks = pd.read_parquet(path)
    chunks["normalized_text"] = text
    chunks["raw_text"] = text
    chunks["normalized_no_accent"] = no_accent
    chunks.to_parquet(path)


@pytest.fixture
def index(cache: Path, tmp_path: Path) -> RetrievalIndex:
    # Same accent-free spelling, different meaning: only accents can separate them.
    write_chunk(cache, "L21", "L21_V001", "bão lũ miền trung", "bao lu mien trung")
    write_chunk(cache, "L22", "L22_V004", "báo lú hôm nay", "bao lu hom nay")
    build_index(cache, tmp_path / "index")
    return RetrievalIndex(tmp_path / "index")


def test_text_helpers():
    assert remove_accents("Đường Trường Sơn") == "duong truong son"
    assert tokens("The red BUS and the bus", stopwords=frozenset({"the", "and"})) == ["red", "bus"]
    assert fts_or_query(['a"b', "NOT"]) == '"a""b" OR "NOT"'
    assert fts_or_query([]) is None


def test_caption_bm25(index: RetrievalIndex):
    hits = index.caption("red bus v004")
    assert {hit.frame_row for hit in hits[:2]} == {3, 4}
    assert index.caption("the of a") == []


def test_asr_prefers_exact_accents_and_maps_chunks_to_frames(index: RetrievalIndex):
    chunks = index.asr_chunks("bão lũ")
    assert chunks["video_id"].tolist() == ["L21_V001", "L22_V004"]
    hits = index.asr("bão lũ")
    # Chunk 0-4.5 s (+1 s pad) covers every keyframe of both videos in this fixture.
    assert [hit.frame_row for hit in hits] == [0, 1, 2, 3, 4]
    assert hits[0].score == hits[2].score > hits[3].score
    assert "bão lũ" in hits[0].evidence


def test_frames_in_interval_falls_back_to_nearest(index: RetrievalIndex):
    assert index.frames_in_interval("L21_V001", 50.0, 51.0).tolist() == [2]
    assert index.frames_in_interval("L99_V999", 0, 1).tolist() == []


def test_objects_are_soft_constraints(index: RetrievalIndex):
    assert parse_objects("car:2; Person") == [
        ObjectConstraint("car", 2),
        ObjectConstraint("person"),
    ]
    assert [hit.frame_row for hit in index.objects(parse_objects("car:2"))] == [0, 3]
    assert index.objects(parse_objects("car:3")) == []
    assert [hit.frame_row for hit in index.objects(parse_objects("xe ô tô; boat"))] == [0, 3]


def test_weighted_rrf_ties_weights_and_candidates():
    rankings = {
        "a": [Hit(1, 0.9), Hit(2, 0.9), Hit(3, 0.5)],
        "b": [Hit(3, 5.0)],
        "c": [Hit(1, 1.0)],
    }
    fused = weighted_rrf(rankings, {"a": 1.0, "b": 2.0, "c": 0.0}, k=0)
    # Frame 3: a#3 (1/3) + weight 2 * b#1 (2/1) outranks frame 1 (a#1 only; c has weight 0).
    assert [hit.frame_row for hit in fused] == [3, 1, 2]
    assert fused[0].score == pytest.approx(1 / 3 + 2 / 1)
    assert fused[2].ranks == {"a": 1}  # tied score with frame 1 -> shared rank 1
    assert "c" not in fused[1].ranks
    assert [hit.frame_row for hit in weighted_rrf(rankings, candidates=[2])] == [2]


def test_collapse_by_shot():
    frames = pd.DataFrame({"video_id": ["v", "v", "w"], "shot_id": [0, 0, 0]})
    hits = [FusedHit(1, 3.0), FusedHit(0, 2.0), FusedHit(2, 1.0)]
    assert [hit.frame_row for hit in collapse_by_shot(hits, frames)] == [1, 2]


def test_configs_on_index(index: RetrievalIndex, cache: Path):
    level, _ = VIDEOS["L22_V004"]
    expected = np.load(cache / "siglip" / "data" / level / "L22_V004" / "expected.npy")
    queries = [
        Query("q1", "bão lũ", "red bus v004", tuple(parse_objects("car:2"))),
        Query("q2", "", ""),
    ]
    vectors = {
        "siglip": np.stack([expected[1], expected[0]]).astype(np.float32),
        "beit3": np.full((2, 1024), np.nan, dtype=np.float32),
    }
    q1, q2 = search_all(index, queries, vectors)
    assert q1.skipped == {"beit3": "no query text", "beit3+nbr": "no query text"}
    assert "siglip+nbr" in q1.hits
    assert "od" in q2.skipped and "caption" in q2.skipped

    top_a = run_config(CONFIGS["A"], q1, index, by_shot=False)
    assert top_a[0].frame_row == 4
    assert run_config(CONFIGS["B"], q1, index) == []
    # Every fixture keyframe shares shot 0 of its video: one hit per video after collapsing.
    assert len(run_config(CONFIGS["E"], q1, index)) == 2
    gated = run_config(CONFIGS["D"], q1, index, by_shot=False)
    assert {hit.frame_row for hit in gated} <= {hit.frame_row for hit in q1.hits["siglip"]}
    assert set(run_config(CONFIGS["E"], q1, index, by_shot=False)[0].ranks) >= {"siglip", "asr"}


def test_evaluation(tmp_path: Path, index: RetrievalIndex):
    queries = tmp_path / "queries.csv"
    queries.write_text(
        "query_id,split,query_type,query_vi,query_en,objects,video_id,start_sec,end_sec\n"
        "q1,dev,asr,bão lũ,,,L22_V004,2.0,3.0\n"
        "q1,dev,asr,bão lũ,,,L21_V001,50,60\n"
        "q2,test,visual,xe buýt,a bus,car:2,L21_V001,0,1\n",
        encoding="utf-8",
    )
    labeled = load_queries(queries)
    assert [item.query.query_id for item in labeled] == ["q1", "q2"]
    assert len(labeled[0].intervals) == 2
    assert labeled[1].query.objects == (ObjectConstraint("car", 2),)

    hits = [FusedHit(0, 1.0), FusedHit(4, 0.5)]  # L21@0.0s, L22@2.4s
    assert first_relevant_rank(hits, labeled[0].intervals, index.frames) == 2
    assert first_relevant_rank(hits, labeled[0].intervals, index.frames, tolerance_sec=0) == 2
    assert first_relevant_rank(hits[:1], labeled[0].intervals, index.frames) is None

    summary = summarize(
        pd.DataFrame({"config": ["A", "A"], "query_type": ["asr", "visual"], "rank": [2.0, np.nan]})
    )
    overall = summary[(summary.config == "A") & (summary.query_type == "ALL")].iloc[0]
    assert overall["R@1"] == 0.0
    assert overall["R@5"] == 0.5
    assert overall["MRR"] == pytest.approx(0.25)


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


def test_new_configs_run(index: RetrievalIndex, cache: Path):
    level, _ = VIDEOS["L22_V004"]
    expected = np.load(cache / "siglip" / "data" / level / "L22_V004" / "expected.npy")
    branches = search_all(
        index, [Query("q", "bão lũ", "")], {"siglip": expected[1:2].astype(np.float32)}
    )[0]
    for name in ("C-max", "C-nbr", "C-max-nbr", "E-nbr"):
        assert run_config(CONFIGS[name], branches, index), name
