"""Audit completed AIC ingestion datasets on Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from multimedia_video_rag.ingestion.audit import (
    HuggingFaceHubClient,
    audit_repositories,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit keyframe, visual, OD, ASR, and caption artifacts on Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ingestion-audit"),
        help="Local report directory (default: artifacts/ingestion-audit).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only inspect repository membership and required file/marker presence.",
    )
    parser.add_argument(
        "--verify-large-hashes",
        action="store_true",
        help="Also download every visual safetensors file and verify its SHA-256.",
    )
    parser.add_argument("--revision", default="main", help="Hub revision to inspect.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of videos audited concurrently (default: 8).",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Audit only this real source video; repeat the option for multiple videos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    client = HuggingFaceHubClient(token=token, revision=args.revision)
    frame, summary = audit_repositories(
        client,
        quick=args.quick,
        verify_large_hashes=args.verify_large_hashes,
        workers=args.workers,
        video_ids=args.video_ids,
    )
    write_report(frame, summary, args.output_dir)

    print(f"Source videos: {summary['source_video_count']}")
    print(f"Audited: {summary['audited_video_count']}")
    print(f"Complete: {summary['complete_video_count']}")
    print(f"Errors: {summary['error_video_count']}")
    print(f"Report: {args.output_dir.resolve()}")
    if summary["error_video_count"]:
        failing = frame.loc[frame["status"] != "complete", "video_id"].tolist()
        print("Failing videos:", ", ".join(failing[:20]))
        if len(failing) > 20:
            print(f"... and {len(failing) - 20} more")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
