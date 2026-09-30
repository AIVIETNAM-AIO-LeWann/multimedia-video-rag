"""Run one query against an experimental index and print ranked keyframes with evidence."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from multimedia_video_rag.retrieval.pipeline import CONFIGS, Query, run_config, search_all
from multimedia_video_rag.retrieval.search import RetrievalIndex, parse_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("indexes/dev"))
    parser.add_argument("--vi", default="", help="Vietnamese query (ASR; SigLIP 2 by default).")
    parser.add_argument("--en", default="", help="English query (BEiT-3 and captions).")
    parser.add_argument("--objects", default="", help='Object constraints, e.g. "car:2; person".')
    parser.add_argument("--config", default="E", choices=sorted(CONFIGS))
    parser.add_argument(
        "--visual",
        default="siglip,beit3",
        help='Comma-separated text encoders to load (siglip, beit3); "none" for text-only.',
    )
    parser.add_argument(
        "--query-vectors",
        type=Path,
        help="query_vectors.npz from notebooks/retrieval/encode-query-vectors.ipynb; "
        "replaces local encoders and --vi/--en/--objects.",
    )
    parser.add_argument("--query-id", help="Query to run from --query-vectors.")
    parser.add_argument("--siglip-text", default="en", choices=("vi", "en"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-shot-collapse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = RetrievalIndex(args.index)
    if args.query_vectors:
        query, vectors = query_from_file(args, index)
    else:
        if not (args.vi or args.en or args.objects):
            raise SystemExit("Provide --vi, --en and/or --objects (or --query-vectors)")
        query = Query("cli", args.vi, args.en, tuple(parse_objects(args.objects)))
        vectors = encode_locally(args, index)
    print(f"Query {query.query_id}: vi={query.text_vi!r} en={query.text_en!r}")

    branches = search_all(index, [query], vectors)[0]
    for branch, reason in branches.skipped.items():
        print(f"[{branch}] skipped: {reason}")
    print_results(args, index, branches)
    return 0


def query_from_file(args, index):
    from multimedia_video_rag.retrieval.query_vectors import load_query_vectors

    loaded = load_query_vectors(args.query_vectors, index.manifest)
    ids = [query.query_id for query in loaded.queries]
    if args.query_id is None:
        if len(ids) != 1:
            raise SystemExit(f"Choose --query-id from: {', '.join(ids)}")
        args.query_id = ids[0]
    position = loaded.row(args.query_id)
    vectors = {module: matrix[position : position + 1] for module, matrix in loaded.vectors.items()}
    return loaded.queries[position], vectors


def encode_locally(args, index) -> dict[str, np.ndarray | None]:
    vectors: dict[str, np.ndarray | None] = {}
    for module in [name for name in args.visual.split(",") if name and name != "none"]:
        from multimedia_video_rag.retrieval.encoders import load_encoder

        text = (args.vi if args.siglip_text == "vi" else args.en) if module == "siglip" else args.en
        if not text.strip():
            print(f"[{module}] skipped: no query text for this encoder")
            continue
        encoder = load_encoder(index.manifest, module, models_dir=args.models_dir, dtype=args.dtype)
        vectors[module] = encoder.encode([text])
        del encoder  # Free the model before loading the next one (RAM is limited).
        gc.collect()
    return vectors


def print_results(args, index, branches) -> None:
    config = CONFIGS[args.config]
    fused = run_config(config, branches, index, limit=args.top, by_shot=not args.no_shot_collapse)
    print(f"\nConfig {config.name}: {config.description}\n")
    info = index.describe([hit.frame_row for hit in fused])
    for position, (hit, row) in enumerate(zip(fused, info.itertuples(index=False), strict=True), 1):
        ranks = " ".join(f"{name}#{rank}" for name, rank in sorted(hit.ranks.items()))
        print(
            f"{position:>3}. {row.frame_uid:<18} t={row.timestamp_sec:8.2f}s "
            f"shot={row.shot_id:<4} score={hit.score:.4f}  [{ranks}]"
        )
        print(f"     caption: {row.caption_en[:110]}")
        for branch in ("asr", "od"):
            if branch in hit.evidence:
                print(f"     {branch}: {hit.evidence[branch][:110]}")


if __name__ == "__main__":
    raise SystemExit(main())
