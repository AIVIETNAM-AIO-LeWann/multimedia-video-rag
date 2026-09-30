"""Labeled-query evaluation: Recall@K and MRR per configuration and query type.

Queries file (CSV, UTF-8): one row per relevant interval; repeat ``query_id`` for
queries with several correct moments. Columns:

``query_id, split, query_type, query_vi, query_en, objects, video_id, start_sec, end_sec, notes``
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from multimedia_video_rag.retrieval.fusion import FusedHit
from multimedia_video_rag.retrieval.pipeline import Query
from multimedia_video_rag.retrieval.search import parse_objects

QUERY_COLUMNS = (
    "query_id",
    "split",
    "query_type",
    "query_vi",
    "query_en",
    "objects",
    "video_id",
    "start_sec",
    "end_sec",
    "notes",
)
REQUIRED_COLUMNS = ("query_id", "query_vi", "video_id", "start_sec", "end_sec")
RECALL_KS = (1, 5, 10, 20, 50, 100)
# A keyframe counts as correct when its timestamp is inside the labeled interval,
# widened by this tolerance to absorb keyframe sampling gaps.
DEFAULT_TOLERANCE_SEC = 2.0


@dataclass(frozen=True)
class Interval:
    video_id: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class LabeledQuery:
    query: Query
    split: str
    intervals: tuple[Interval, ...]


def load_queries(path: Path) -> list[LabeledQuery]:
    # utf-8-sig also accepts the BOM Excel writes when saving CSV as UTF-8.
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for column in QUERY_COLUMNS:
        if column not in table.columns:
            table[column] = ""
    labeled = []
    for query_id, group in table.groupby("query_id", sort=False):
        first = group.iloc[0]
        for column in ("query_vi", "query_en", "query_type", "objects", "split"):
            if group[column].nunique() > 1:
                raise ValueError(f"{query_id}: rows disagree on {column}")
        intervals = []
        for row in group.itertuples(index=False):
            start, end = float(row.start_sec), float(row.end_sec)
            if end < start:
                raise ValueError(f"{query_id}: end_sec < start_sec")
            intervals.append(Interval(str(row.video_id).strip(), start, end))
        labeled.append(
            LabeledQuery(
                Query(
                    query_id=str(query_id),
                    text_vi=first["query_vi"],
                    text_en=first["query_en"],
                    objects=tuple(parse_objects(first["objects"])),
                    query_type=first["query_type"] or "unlabeled",
                ),
                split=first["split"] or "dev",
                intervals=tuple(intervals),
            )
        )
    return labeled


def first_relevant_rank(
    hits: Sequence[FusedHit],
    intervals: Iterable[Interval],
    frames: pd.DataFrame,
    *,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
) -> int | None:
    """1-based rank of the first hit inside any labeled interval, or None."""
    video = frames["video_id"].to_numpy()
    time = frames["timestamp_sec"].to_numpy()
    targets = list(intervals)
    for rank, hit in enumerate(hits, start=1):
        for interval in targets:
            if (
                video[hit.frame_row] == interval.video_id
                and interval.start_sec - tolerance_sec
                <= time[hit.frame_row]
                <= interval.end_sec + tolerance_sec
            ):
                return rank
    return None


def summarize(ranks: pd.DataFrame, ks: Sequence[int] = RECALL_KS) -> pd.DataFrame:
    """``ranks`` has columns config, query_type, rank (NaN = not found).

    Returns one row per (config, query_type) plus an ``ALL`` row per config.
    """

    def metrics(group: pd.DataFrame) -> pd.Series:
        rank = group["rank"].to_numpy(dtype=float)
        found = ~np.isnan(rank)
        values = {"queries": len(rank)}
        for k in ks:
            values[f"R@{k}"] = float(np.mean(found & (np.nan_to_num(rank, nan=np.inf) <= k)))
        values["MRR"] = float(np.mean(np.where(found, 1.0 / np.nan_to_num(rank, nan=1.0), 0.0)))
        return pd.Series(values)

    per_type = ranks.groupby(["config", "query_type"]).apply(metrics, include_groups=False)
    overall = ranks.groupby("config").apply(metrics, include_groups=False)
    overall.index = pd.MultiIndex.from_arrays(
        [overall.index, ["ALL"] * len(overall)], names=["config", "query_type"]
    )
    return pd.concat([overall, per_type]).sort_index().reset_index()
