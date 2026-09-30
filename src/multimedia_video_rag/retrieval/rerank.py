"""Temporal-context reranking of visual hits.

Neighbor Score Aggregation follows Tran et al., "Towards Efficient and Robust Moment
Retrieval System" (arXiv:2504.08384, Algorithm 2): a keyframe is re-scored with the query
similarity of its temporal neighbours, so frames inside a stretch that keeps matching the
query rise and isolated look-alike frames fall.

The paper leaves the neighbourhood unspecified and sums neighbour scores. Here the
neighbourhood is the ``window`` keyframes on each side within the same video (optionally
the same shot), and the score is the *mean* similarity over the frame and its available
neighbours, so frames at video edges are not penalized for having fewer neighbours.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from multimedia_video_rag.retrieval.search import Hit, RetrievalIndex

NEIGHBOR_WINDOW = 2


def neighbor_aggregate(
    index: RetrievalIndex,
    module: str,
    query_vector: np.ndarray,
    hits: Sequence[Hit],
    *,
    window: int = NEIGHBOR_WINDOW,
    same_shot: bool = False,
    reduce: str = "mean",
) -> list[Hit]:
    """Re-score ``hits`` of one visual module by query similarity over their neighbourhood.

    ``reduce="mean"`` is our variant; ``reduce="sum"`` follows the paper's Algorithm 2
    literally (frames with fewer same-video neighbours get lower totals).
    """
    if reduce not in {"mean", "sum"}:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")
    if not hits:
        return []
    rows = np.array([hit.frame_row for hit in hits], dtype=np.int64)
    offsets = np.arange(-window, window + 1, dtype=np.int64)
    neighbours = rows[:, None] + offsets[None, :]
    total = len(index.frames)
    valid = (neighbours >= 0) & (neighbours < total)
    clipped = np.clip(neighbours, 0, total - 1)
    valid &= index.video_codes[clipped] == index.video_codes[rows][:, None]
    if same_shot:
        valid &= index.shot_ids[clipped] == index.shot_ids[rows][:, None]

    unique = np.unique(clipped[valid])
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    similarities = index.reconstruct(module, unique) @ query
    positions = np.clip(np.searchsorted(unique, clipped), 0, len(unique) - 1)
    matrix = np.where(valid, similarities[positions], np.nan)
    # The frame itself is always valid, so neither reduction sees an all-NaN row.
    scores = np.nanmean(matrix, axis=1) if reduce == "mean" else np.nansum(matrix, axis=1)

    order = np.argsort(-scores, kind="stable")
    return [Hit(int(rows[i]), float(scores[i]), hits[i].evidence) for i in order]
