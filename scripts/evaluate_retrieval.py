"""Evaluate retrieval configurations A-E on labeled queries (Recall@K, MRR)."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multimedia_video_rag.retrieval.evaluation import first_relevant_rank, load_queries, summarize
from multimedia_video_rag.retrieval.pipeline import CONFIGS, run_config, search_all
from multimedia_video_rag.retrieval.search import RetrievalIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("indexes/dev"))
    parser.add_argument("--queries", type=Path, required=True, help="Labeled queries CSV.")
    parser.add_argument("--split", default="dev", help='Split to evaluate ("all" for every row).')
    parser.add_argument("--configs", default="A,B,C,D,E")
    parser.add_argument(
        "--query-vectors",
        type=Path,
        help="query_vectors.npz encoded on Colab/Kaggle; skips local encoders.",
    )
    parser.add_argument("--visual", default="siglip,beit3")
    parser.add_argument("--siglip-text", default="en", choices=("vi", "en"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/retrieval-eval"))
    return parser.parse_args()


def query_texts(labeled, module: str, siglip_text: str) -> list[str]:
    use_vi = module == "siglip" and siglip_text == "vi"
    return [item.query.text_vi if use_vi else item.query.text_en for item in labeled]


def encode_with_cache(index, module, texts, args) -> np.ndarray:
    """Encode once per (model revision, dtype, texts); empty texts become NaN rows."""
    entry = index.manifest["faiss"][module]
    key = hashlib.sha256(
        json.dumps([module, entry.get("model_revisions"), args.dtype, texts]).encode("utf-8")
    ).hexdigest()[:16]
    cache = args.output_dir / "query-vectors" / f"{module}-{key}.npy"
    if cache.is_file():
        return np.load(cache)
    from multimedia_video_rag.retrieval.encoders import load_encoder

    matrix = np.full((len(texts), entry["dimension"]), np.nan, dtype=np.float32)
    present = [i for i, text in enumerate(texts) if text.strip()]
    if present:
        encoder = load_encoder(index.manifest, module, models_dir=args.models_dir, dtype=args.dtype)
        for start in range(0, len(present), 32):
            batch = present[start : start + 32]
            matrix[batch] = encoder.encode([texts[i] for i in batch])
        del encoder
        gc.collect()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, matrix)
    return matrix


def main() -> int:
    args = parse_args()
    index = RetrievalIndex(args.index)
    labeled = load_queries(args.queries)
    if args.split != "all":
        labeled = [item for item in labeled if item.split == args.split]
    if not labeled:
        raise SystemExit(f"No queries for split {args.split!r}")
    print(f"Queries: {len(labeled)} | index frames: {len(index.frames)}")

    if args.query_vectors:
        from multimedia_video_rag.retrieval.query_vectors import align_to, load_query_vectors

        loaded = load_query_vectors(args.query_vectors, index.manifest)
        vectors = align_to(loaded, [item.query for item in labeled])
        print(f"Query vectors: {args.query_vectors} (SigLIP text: {loaded.meta['siglip_text']})")
    else:
        vectors = {
            module: encode_with_cache(
                index, module, query_texts(labeled, module, args.siglip_text), args
            )
            for module in [name for name in args.visual.split(",") if name and name != "none"]
        }
    branches = search_all(index, [item.query for item in labeled], vectors)

    rows = []
    for item, result in zip(labeled, branches, strict=True):
        for name in args.configs.split(","):
            fused = run_config(CONFIGS[name], result, index)
            rank = first_relevant_rank(
                fused, item.intervals, index.frames, tolerance_sec=args.tolerance_sec
            )
            top = fused[0].frame_row if fused else None
            rows.append(
                {
                    "config": name,
                    "query_id": item.query.query_id,
                    "query_type": item.query.query_type,
                    "rank": np.nan if rank is None else rank,
                    "skipped_branches": ",".join(sorted(result.skipped)),
                    "top1_frame_uid": None if top is None else index.frames.frame_uid[top],
                }
            )
    ranks = pd.DataFrame(rows)
    summary = summarize(ranks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(args.output_dir / "per-query-ranks.csv", index=False, encoding="utf-8")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(3).to_string(index=False))
    print(f"\nReports: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
