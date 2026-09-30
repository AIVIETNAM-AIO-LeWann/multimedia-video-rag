"""Experimental retrieval configurations from docs/research (OCR not yet available).

A-E follow the architecture review. ``*-max`` / ``*-nbr`` add the ensemble search and
Neighbor Score Aggregation of Tran et al. (arXiv:2504.08384): ``siglip+nbr`` and
``beit3+nbr`` are the visual rankings re-scored with their temporal neighbours.

Branch weights are untuned starting points. They must be chosen on a development split
of labeled queries, never on the reported test split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from multimedia_video_rag.retrieval.fusion import (
    FusedHit,
    collapse_by_shot,
    weighted_max_norm,
    weighted_rrf,
)
from multimedia_video_rag.retrieval.rerank import NEIGHBOR_WINDOW, neighbor_aggregate
from multimedia_video_rag.retrieval.search import Hit, ObjectConstraint, RetrievalIndex

VISUAL_K = 1000
TEXT_K = 300
VISUAL_GATE_K = 1000


@dataclass(frozen=True)
class Query:
    query_id: str
    text_vi: str
    text_en: str = ""
    objects: tuple[ObjectConstraint, ...] = ()
    query_type: str = ""


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    description: str
    weights: Mapping[str, float]
    visual_gate: bool = False
    fusion: str = "rrf"  # "rrf" or "max_norm" (visual branches only)


NBR = {"siglip+nbr": 1.0, "beit3+nbr": 1.0}
TEXT_WEIGHTS = {"caption": 0.5, "asr": 0.7, "od": 0.3}

CONFIGS: dict[str, RetrievalConfig] = {
    config.name: config
    for config in (
        RetrievalConfig("A", "SigLIP 2 only", {"siglip": 1.0}),
        RetrievalConfig("B", "BEiT-3 only", {"beit3": 1.0}),
        RetrievalConfig("C", "SigLIP 2 + BEiT-3 RRF", {"siglip": 1.0, "beit3": 1.0}),
        RetrievalConfig(
            "D",
            "C candidates re-ranked with caption/ASR/OD (visual gate)",
            {"siglip": 1.0, "beit3": 1.0, "caption": 0.5, "asr": 0.7, "od": 0.3},
            visual_gate=True,
        ),
        RetrievalConfig(
            "E",
            "Union of every branch, weighted RRF",
            {"siglip": 1.0, "beit3": 1.0, "caption": 0.5, "asr": 0.7, "od": 0.3},
        ),
        RetrievalConfig(
            "C-max",
            "SigLIP 2 + BEiT-3, max-normalized score ensemble (2504.08384 Alg. 3)",
            {"siglip": 1.0, "beit3": 1.0},
            fusion="max_norm",
        ),
        RetrievalConfig("C-nbr", "C on neighbour-aggregated visual scores (Alg. 2)", NBR),
        RetrievalConfig(
            "C-max-nbr",
            "Max-normalized ensemble of neighbour-aggregated scores (Alg. 2 + 3)",
            NBR,
            fusion="max_norm",
        ),
        RetrievalConfig("E-nbr", "E with neighbour-aggregated visual branches", NBR | TEXT_WEIGHTS),
    )
}


@dataclass
class BranchResults:
    """Every branch ranking for one query, computed once and reused by all configs."""

    hits: dict[str, list[Hit]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def text_branches(index: RetrievalIndex, query: Query, *, k: int = TEXT_K) -> BranchResults:
    results = BranchResults()
    if query.text_en.strip():
        results.hits["caption"] = index.caption(query.text_en, k=k)
    else:
        results.skipped["caption"] = "no English query"
    results.hits["asr"] = index.asr(query.text_vi, k=k) if query.text_vi.strip() else []
    if query.objects:
        results.hits["od"] = index.objects(query.objects, k=k)
    else:
        results.skipped["od"] = "no object constraints"
    return results


def add_visual(
    results: BranchResults, module: str, hits: list[Hit] | None, reason: str = ""
) -> None:
    if hits is None:
        results.skipped[module] = reason
    else:
        results.hits[module] = hits


def search_all(
    index: RetrievalIndex,
    queries: Sequence[Query],
    vectors: Mapping[str, np.ndarray | None],
    *,
    visual_k: int = VISUAL_K,
    text_k: int = TEXT_K,
    neighbor_window: int = NEIGHBOR_WINDOW,
) -> list[BranchResults]:
    """Run every branch for every query; visual branches are batched per module.

    ``vectors[module]`` holds one row per query; a row of NaN means the query has no
    usable text for that encoder (e.g. BEiT-3 without an English query). With
    ``neighbor_window > 0`` each visual ranking also yields a ``<module>+nbr`` ranking.
    """
    results = [text_branches(index, query, k=text_k) for query in queries]
    for module in index.visual_modules:
        matrix = vectors.get(module)
        if matrix is None:
            for result in results:
                add_visual(result, module, None, "encoder not loaded")
            continue
        usable = ~np.isnan(matrix).any(axis=1)
        batched = iter(
            index.visual_batch(module, matrix[usable], k=visual_k) if usable.any() else []
        )
        for result, vector, ok in zip(results, matrix, usable, strict=True):
            hits = next(batched) if ok else None
            add_visual(result, module, hits, "no query text")
            if neighbor_window > 0:
                reranked = (
                    neighbor_aggregate(index, module, vector, hits, window=neighbor_window)
                    if hits is not None
                    else None
                )
                add_visual(result, f"{module}+nbr", reranked, "no query text")
    return results


def run_config(
    config: RetrievalConfig,
    branches: BranchResults,
    index: RetrievalIndex,
    *,
    limit: int = 1000,
    by_shot: bool = True,
) -> list[FusedHit]:
    candidates = None
    if config.visual_gate:
        visual = {
            name: branches.hits[name] for name in ("siglip", "beit3") if name in branches.hits
        }
        gate = weighted_rrf(visual, limit=VISUAL_GATE_K)
        candidates = [hit.frame_row for hit in gate]
    fuse = weighted_max_norm if config.fusion == "max_norm" else weighted_rrf
    fused = fuse(branches.hits, config.weights, candidates=candidates)
    if by_shot:
        fused = collapse_by_shot(fused, index.frames)
    return fused[:limit]
