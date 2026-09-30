"""Rank fusion across branches whose raw scores are not comparable."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from multimedia_video_rag.retrieval.search import Hit

RRF_K = 60


@dataclass
class FusedHit:
    frame_row: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)


def weighted_rrf(
    rankings: Mapping[str, Sequence[Hit]],
    weights: Mapping[str, float] | None = None,
    *,
    k: int = RRF_K,
    limit: int | None = None,
    candidates: Iterable[int] | None = None,
) -> list[FusedHit]:
    """score(frame) = sum_b weight_b / (k + rank_b), ranks starting at 1.

    Hits sharing a score within a branch (e.g. frames of one ASR chunk) share a rank.
    ``candidates`` restricts the output to a given frame set (visual-gate configuration).
    """
    allowed = set(candidates) if candidates is not None else None
    fused: dict[int, FusedHit] = {}
    for branch, hits in rankings.items():
        weight = 1.0 if weights is None else weights.get(branch, 0.0)
        if weight == 0:
            continue
        rank, previous_score = 0, None
        for position, hit in enumerate(hits, start=1):
            if hit.score != previous_score:
                rank, previous_score = position, hit.score
            if allowed is not None and hit.frame_row not in allowed:
                continue
            entry = fused.setdefault(hit.frame_row, FusedHit(hit.frame_row, 0.0))
            if branch in entry.ranks:
                continue
            entry.score += weight / (k + rank)
            entry.ranks[branch] = rank
            if hit.evidence:
                entry.evidence[branch] = hit.evidence
    ordered = sorted(fused.values(), key=lambda item: (-item.score, item.frame_row))
    return ordered[:limit] if limit is not None else ordered


def weighted_max_norm(
    rankings: Mapping[str, Sequence[Hit]],
    weights: Mapping[str, float] | None = None,
    *,
    limit: int | None = None,
    candidates: Iterable[int] | None = None,
) -> list[FusedHit]:
    """score(frame) = sum_b weight_b * score_b / max(score_b).

    Ensemble search of Tran et al. (arXiv:2504.08384, Algorithm 3). Only meaningful for
    branches whose scores are similarities on a comparable scale (the visual branches);
    a branch whose best score is not positive is skipped.
    """
    allowed = set(candidates) if candidates is not None else None
    fused: dict[int, FusedHit] = {}
    for branch, hits in rankings.items():
        weight = 1.0 if weights is None else weights.get(branch, 0.0)
        if weight == 0 or not hits:
            continue
        best = max(hit.score for hit in hits)
        if best <= 0:
            continue
        for rank, hit in enumerate(hits, start=1):
            if allowed is not None and hit.frame_row not in allowed:
                continue
            entry = fused.setdefault(hit.frame_row, FusedHit(hit.frame_row, 0.0))
            if branch in entry.ranks:
                continue
            entry.score += weight * hit.score / best
            entry.ranks[branch] = rank
            if hit.evidence:
                entry.evidence[branch] = hit.evidence
    ordered = sorted(fused.values(), key=lambda item: (-item.score, item.frame_row))
    return ordered[:limit] if limit is not None else ordered


def collapse_by_shot(hits: Sequence[FusedHit], frames: pd.DataFrame) -> list[FusedHit]:
    """Keep the best-scoring frame of each (video, shot) so one shot cannot flood the list."""
    video = frames["video_id"].to_numpy()
    shot = frames["shot_id"].to_numpy()
    seen: set[tuple[str, int]] = set()
    kept = []
    for hit in hits:
        key = (str(video[hit.frame_row]), int(shot[hit.frame_row]))
        if key not in seen:
            seen.add(key)
            kept.append(hit)
    return kept
