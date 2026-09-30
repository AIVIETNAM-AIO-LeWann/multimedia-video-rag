"""Score fusion of the two visual encoders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from multimedia_video_rag.retrieval.search import Hit

RRF_K = 60


@dataclass
class FusedHit:
    frame_row: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)


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
