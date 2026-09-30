"""Run a paper baseline on curated queries and write gallery-ready results.

``--method paper1`` is arXiv:2504.08384 (configs P1, P1-start, P1-end); ``--method paper2``
is GRAB, arXiv:2504.09298 (configs P2, P2-start, P2-end).

Query vectors come from notebooks/retrieval/encode-query-vectors.ipynb run on a CSV whose
``query_id`` values are ``<id>#full``, ``<id>#start`` and ``<id>#end`` (see
artifacts/eval/encode/aic2026-encode-queries.csv). For queries with start/end texts, the
top-ranked frame is the pivot of the dual-query temporal search.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from multimedia_video_rag.retrieval.baselines import (
    MAX_GAP,
    MAX_SIDE_FRAMES,
    STOP_QUANTILE,
    TOP_M,
    abts_frame_pair,
    dual_query_frame_pair,
    grab_search,
    paper_2504_08384_search,
)
from multimedia_video_rag.retrieval.query_vectors import load_query_vectors
from multimedia_video_rag.retrieval.search import RetrievalIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("indexes/dev"))
    parser.add_argument("--query-vectors", type=Path, required=True)
    parser.add_argument("--method", choices=("paper1", "paper2"), default="paper1")
    parser.add_argument("--output", type=Path, help="Default: artifacts/eval/<method>-results.csv")
    parser.add_argument("--top", type=int, default=5, help="Pivot candidates to keep per query.")
    parser.add_argument("--top-m", type=int, default=TOP_M)
    parser.add_argument("--max-side-frames", type=int, default=MAX_SIDE_FRAMES)
    parser.add_argument("--max-gap", type=int, default=MAX_GAP)
    parser.add_argument("--stop-quantile", type=float, default=STOP_QUANTILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output or Path(f"artifacts/eval/{args.method}-results.csv")
    config = "P1" if args.method == "paper1" else "P2"
    index = RetrievalIndex(args.index)
    loaded = load_query_vectors(args.query_vectors, index.manifest)
    rows = {query.query_id: position for position, query in enumerate(loaded.queries)}

    def vectors(key: str) -> dict[str, np.ndarray]:
        return {module: matrix[rows[key]] for module, matrix in loaded.vectors.items()}

    base_ids = sorted({key.split("#")[0] for key in rows if key.endswith("#full")})
    frames = index.frames
    out, timings = [], []
    for query_id in base_ids:
        started = time.perf_counter()
        if args.method == "paper1":
            fused = paper_2504_08384_search(index, vectors(f"{query_id}#full"), top_m=args.top_m)
        else:
            fused = grab_search(index, vectors(f"{query_id}#full")["beit3"])
        timings.append(time.perf_counter() - started)
        for rank, hit in enumerate(fused[: args.top], 1):
            out.append((query_id, config, rank, hit.frame_row))
        if fused and f"{query_id}#start" in rows:
            if args.method == "paper1":
                pair = dual_query_frame_pair(
                    index,
                    fused[0].frame_row,
                    vectors(f"{query_id}#start"),
                    vectors(f"{query_id}#end"),
                    max_side_frames=args.max_side_frames,
                    max_gap=args.max_gap,
                    stop_quantile=args.stop_quantile,
                )
            else:
                pair = abts_frame_pair(
                    index,
                    fused[0].frame_row,
                    vectors(f"{query_id}#start")["beit3"],
                    vectors(f"{query_id}#end")["beit3"],
                )
            out.append((query_id, f"{config}-start", 1, pair.start_row))
            out.append((query_id, f"{config}-end", 1, pair.end_row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "config", "rank", "video_id", "timestamp_sec", "frame_uid"])
        for query_id, config, rank, row in out:
            info = frames.iloc[row]
            writer.writerow(
                [
                    query_id,
                    config,
                    rank,
                    info.video_id,
                    round(info.timestamp_sec, 2),
                    info.frame_uid,
                ]
            )
    print(f"Queries: {len(base_ids)} | results: {args.output}")
    print(f"Search latency: mean {np.mean(timings):.2f}s, max {np.max(timings):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
