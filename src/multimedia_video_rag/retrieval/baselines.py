"""Re-implementations of published methods, used as baselines.

``paper_2504_08384`` follows Tran et al., "Towards Efficient and Robust Moment Retrieval
System: A Unified Framework for Multi-Granularity Models and Temporal Reranking"
(arXiv:2504.08384): per-model top-M search, Neighbor Score Aggregation (Algorithm 2),
max-normalized ensemble (Algorithm 3) and dual-query temporal frame-pair search
(Algorithm 4). Our encoders replace theirs: SigLIP 2 stands in for OpenCLIP next to BEiT-3.

Details the paper leaves open, and the choices made here:

- neighbourhood: ``window`` keyframes on each side within the same video, summed as in
  Algorithm 2 (``reduce="sum"``);
- temporal-search similarity: mean cosine over the available encoders;
- "similarity too low": below the ``stop_quantile`` of the sub-query's similarities over
  the whole pivot video;
- ``gap_C``: at most ``max_gap`` keyframes between the start and end frames;
- pivot: the top-ranked frame (the paper lets the user pick it).

``grab_search`` / ``abts_frame_pair`` follow Nguyen-Nhu et al., "A Lightweight Moment
Retrieval System with Global Re-Ranking and Robust Adaptive Bidirectional Temporal Search"
(GRAB, arXiv:2504.09298): BEiT-3 search, SuperGlobal reranking (Eqs. 4-5) and Adaptive
Bidirectional Temporal Search (Algorithms 1-2, Eqs. 6-8). Choices for what GRAB leaves open:

- SuperGlobal database-side refinement averages (p = 1) each candidate with its
  ``refine_k`` most similar images *within the top-M candidates* (SuperGlobal precomputes
  neighbours over the whole database; with 335k keyframes on CPU we approximate it);
- query expansion max-pools (p -> inf) the text embedding with the top ``expand_k`` image
  embeddings, then re-normalizes;
- ABTS: start frames are searched in [pivot - w, pivot] and end frames in [pivot, pivot + w]
  seconds for w in {10, 15, 20}; the stability neighbourhood is ``stability_window``
  keyframes on each side; ``lambda_s``/``lambda_t`` weight similarity and stability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from multimedia_video_rag.retrieval.fusion import FusedHit, weighted_max_norm
from multimedia_video_rag.retrieval.rerank import neighbor_aggregate
from multimedia_video_rag.retrieval.search import RetrievalIndex

TOP_M = 50
NEIGHBOR_WINDOW = 2
MAX_SIDE_FRAMES = 20
MAX_GAP = 40
STOP_QUANTILE = 0.5


@dataclass(frozen=True)
class FramePair:
    pivot_row: int
    start_row: int
    end_row: int
    score: float


def usable(vector: np.ndarray | None) -> bool:
    return vector is not None and not np.isnan(vector).any()


def paper_2504_08384_search(
    index: RetrievalIndex,
    vectors: Mapping[str, np.ndarray | None],
    *,
    top_m: int = TOP_M,
    window: int = NEIGHBOR_WINDOW,
) -> list[FusedHit]:
    """Algorithms 2 + 3: per-model top-M, neighbour sum, max-normalized ensemble."""
    rankings = {}
    for module, vector in vectors.items():
        if not usable(vector):
            continue
        hits = index.visual(module, vector, k=top_m)
        rankings[module] = neighbor_aggregate(
            index, module, vector, hits, window=window, reduce="sum"
        )
    return weighted_max_norm(rankings)


def _video_bounds(index: RetrievalIndex, row: int) -> tuple[int, int]:
    """Frame rows of one video are contiguous (frames are ordered by video, then time)."""
    codes = index.video_codes
    code = codes[row]
    start = row
    while start > 0 and codes[start - 1] == code:
        start -= 1
    end = row
    while end + 1 < len(codes) and codes[end + 1] == code:
        end += 1
    return start, end


def _similarities(
    index: RetrievalIndex, vectors: Mapping[str, np.ndarray | None], rows: np.ndarray
) -> np.ndarray:
    scores = [
        index.reconstruct(module, rows) @ np.asarray(vector, dtype=np.float32).reshape(-1)
        for module, vector in vectors.items()
        if usable(vector)
    ]
    if not scores:
        raise ValueError("No usable query vector for temporal search")
    return np.mean(scores, axis=0)


def _walk(similarity: np.ndarray, pivot: int, step: int, limit: int, threshold: float) -> list:
    """Frames from the pivot outwards (excluding it) until too dissimilar or ``limit``."""
    frames, position = [], pivot + step
    while 0 <= position < len(similarity) and len(frames) < limit:
        if similarity[position] < threshold:
            break
        frames.append(position)
        position += step
    return frames


def dual_query_frame_pair(
    index: RetrievalIndex,
    pivot_row: int,
    start_vectors: Mapping[str, np.ndarray | None],
    end_vectors: Mapping[str, np.ndarray | None],
    *,
    max_side_frames: int = MAX_SIDE_FRAMES,
    max_gap: int = MAX_GAP,
    stop_quantile: float = STOP_QUANTILE,
) -> FramePair:
    """Algorithm 4: walk left with the start query, right with the end query, best pair."""
    first, last = _video_bounds(index, pivot_row)
    rows = np.arange(first, last + 1)
    start_sim = _similarities(index, start_vectors, rows)
    end_sim = _similarities(index, end_vectors, rows)
    pivot = pivot_row - first
    left = _walk(
        start_sim, pivot, -1, max_side_frames, float(np.quantile(start_sim, stop_quantile))
    )
    right = _walk(end_sim, pivot, 1, max_side_frames, float(np.quantile(end_sim, stop_quantile)))
    best = None
    for i in [*left, pivot]:
        for j in [pivot, *right]:
            if j - i > max_gap:
                continue
            score = float(start_sim[i] + end_sim[j])
            if best is None or score > best[0]:
                best = (score, i, j)
    score, i, j = best  # (pivot, pivot) always satisfies the constraints
    return FramePair(pivot_row, first + i, first + j, score)


# ---- GRAB (arXiv:2504.09298) -------------------------------------------------------------

GRAB_MODULE = "beit3"
GRAB_TOP_M = 100
REFINE_K = 5
EXPAND_K = 10
ABTS_WINDOWS_SEC = (10.0, 15.0, 20.0)
STABILITY_WINDOW = 2
LAMBDA_S = 1.0
LAMBDA_T = 0.2


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def grab_search(
    index: RetrievalIndex,
    query_vector: np.ndarray,
    *,
    module: str = GRAB_MODULE,
    top_m: int = GRAB_TOP_M,
    refine_k: int = REFINE_K,
    expand_k: int = EXPAND_K,
) -> list[FusedHit]:
    """SuperGlobal reranking of the top-M BEiT-3 candidates: S = (S1 + S2) / 2."""
    query = _unit(np.asarray(query_vector, dtype=np.float32).reshape(-1))
    hits = index.visual(module, query, k=top_m)
    if not hits:
        return []
    rows = np.array([hit.frame_row for hit in hits], dtype=np.int64)
    images = _unit(index.reconstruct(module, rows))

    # Database-side refinement (p = 1): mean of each image and its nearest candidates.
    affinity = images @ images.T
    k = min(refine_k, len(rows) - 1)
    neighbours = np.argsort(-affinity, axis=1)[:, : k + 1]  # includes the image itself
    refined = _unit(images[neighbours].mean(axis=1))

    # Query expansion (p -> inf): element-wise max over the query and top images.
    expanded = _unit(np.max(np.vstack([query[None, :], images[:expand_k]]), axis=0))

    s1 = refined @ query
    s2 = images @ expanded
    final = (s1 + s2) / 2
    order = np.argsort(-final, kind="stable")
    return [FusedHit(int(rows[i]), float(final[i]), ranks={module: int(i) + 1}) for i in order]


def _stability(images: np.ndarray, window: int) -> np.ndarray:
    """Eq. 7: t_i = 1 - min(1, 2 * std of cos(e_j, e_i) over neighbours j of i)."""
    stability = np.zeros(len(images))
    for i in range(len(images)):
        lo, hi = max(0, i - window), min(len(images), i + window + 1)
        others = [j for j in range(lo, hi) if j != i]
        if not others:
            stability[i] = 1.0
            continue
        sims = images[others] @ images[i]
        stability[i] = 1.0 - min(1.0, 2.0 * float(np.std(sims)))
    return stability


def abts_frame_pair(
    index: RetrievalIndex,
    pivot_row: int,
    start_vector: np.ndarray,
    end_vector: np.ndarray,
    *,
    module: str = GRAB_MODULE,
    windows_sec: tuple[float, ...] = ABTS_WINDOWS_SEC,
    stability_window: int = STABILITY_WINDOW,
    lambda_s: float = LAMBDA_S,
    lambda_t: float = LAMBDA_T,
) -> FramePair:
    """Algorithms 1-2: best start before and best end after the pivot, over several windows."""
    first, last = _video_bounds(index, pivot_row)
    rows = np.arange(first, last + 1)
    times = index.frames["timestamp_sec"].to_numpy()[rows]
    images = _unit(index.reconstruct(module, rows))
    start_q = _unit(np.asarray(start_vector, dtype=np.float32).reshape(-1))
    end_q = _unit(np.asarray(end_vector, dtype=np.float32).reshape(-1))
    stability = _stability(images, stability_window)
    start_conf = lambda_s * (images @ start_q) + lambda_t * stability
    end_conf = lambda_s * (images @ end_q) + lambda_t * stability

    pivot = pivot_row - first
    pivot_time = times[pivot]
    best_start, best_end = pivot, pivot
    for window in windows_sec:
        before = np.flatnonzero((times >= pivot_time - window) & (np.arange(len(rows)) <= pivot))
        after = np.flatnonzero((times <= pivot_time + window) & (np.arange(len(rows)) >= pivot))
        start = int(before[np.argmax(start_conf[before])])
        end = int(after[np.argmax(end_conf[after])])
        if start_conf[start] > start_conf[best_start]:
            best_start = start
        if end_conf[end] > end_conf[best_end]:
            best_end = end
    score = float(start_conf[best_start] + end_conf[best_end])
    return FramePair(pivot_row, first + best_start, first + best_end, score)
