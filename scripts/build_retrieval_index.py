"""Build experimental FAISS + SQLite retrieval indexes from ingested Hub artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from multimedia_video_rag.retrieval.build import build_index
from multimedia_video_rag.retrieval.sources import download_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download keyframe/SigLIP/BEiT-3/caption artifacts (no keyframe images) "
            "and build an experimental index directory."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hf"),
        help="Local mirror of the ingestion datasets (default: data/hf).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("indexes/dev"),
        help="Index directory to publish (default: indexes/dev).",
    )
    parser.add_argument("--revision", default="main", help="Hub revision for every dataset.")
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Limit the build to this video; repeat for several (smoke test).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse the existing cache; provenance is read from the previous sources.json.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Parallel file downloads per dataset; lower it if the Hub rate-limits (default: 4).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace --output-dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources_path = args.cache_dir / "sources.json"
    previous = (
        json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.is_file() else {}
    )
    if args.skip_download:
        sources = previous
    else:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

        def save_progress(progress: dict[str, dict[str, str]]) -> None:
            sources_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        sources = download_sources(
            args.cache_dir,
            token=os.environ.get("HF_TOKEN") or None,
            revision=args.revision,
            video_ids=args.video_ids,
            max_workers=args.download_workers,
            completed=previous,
            on_module_done=save_progress,
        )
        save_progress(sources)
    manifest = build_index(
        args.cache_dir,
        args.output_dir,
        sources=sources,
        video_ids=args.video_ids,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "faiss"}, indent=2))
    print(f"Index: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
