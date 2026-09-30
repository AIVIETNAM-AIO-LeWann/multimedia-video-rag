"""Mirror every video's ``keyframes.tar`` at the keyframe revision pinned in an index manifest.

Images are only needed for display, so they are kept out of the index build. Files already
present in ``--cache-dir`` are skipped, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from multimedia_video_rag.retrieval.sources import with_retries

KEYFRAME_REPO = "aqpahm/aic2026-keyframes-transnetv2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("indexes/dev"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf/keyframe"))
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    from huggingface_hub import snapshot_download

    args = parse_args()
    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"]["keyframe"]
    if source["repo_id"] != KEYFRAME_REPO:
        raise ValueError(f"Unexpected keyframe repository {source['repo_id']}")
    print(f"{KEYFRAME_REPO}@{source['revision']} -> {args.cache_dir}", flush=True)
    with_retries(
        lambda: snapshot_download(
            KEYFRAME_REPO,
            repo_type="dataset",
            revision=source["revision"],
            local_dir=args.cache_dir,
            allow_patterns=["data/*/*/keyframes.tar"],
            max_workers=args.workers,
            token=os.environ.get("HF_TOKEN") or None,
        ),
        "keyframes.tar",
    )
    tars = sorted(args.cache_dir.glob("data/*/*/keyframes.tar"))
    print(f"keyframes.tar files: {len(tars)} / {manifest['video_count']} videos", flush=True)
    return 0 if len(tars) == manifest["video_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
