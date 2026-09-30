"""Print the real schema of one video's artifacts in every ingestion dataset.

Downloads one video's parquet, marker and embedding files (a few MB in total);
for embeddings only the safetensors header (tensor names, dtype, shape) is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from multimedia_video_rag.retrieval.sources import MODULE_FILES, REPOSITORIES


def show_parquet(path: str) -> None:
    frame = pd.read_parquet(path)
    print(f"    rows={len(frame)}")
    for column, dtype in frame.dtypes.items():
        print(f"    - {column}: {dtype}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.head(2).to_string(max_colwidth=80))
    for column in ("status", "language", "model_id", "label_en"):
        if column in frame.columns:
            print(f"    distinct {column}: {frame[column].astype(str).unique()[:10].tolist()}")


def show_safetensors(path: str) -> None:
    with open(path, "rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    for key, value in header.items():
        print(f"    - {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--level", default=None, help="Defaults to the video_id prefix.")
    args = parser.parse_args()
    level = args.level or args.video_id.split("_")[0]
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    for module, repo_id in REPOSITORIES.items():
        sha = api.dataset_info(repo_id).sha
        print(f"\n=== {module}: {repo_id}@{sha[:12]}")
        for name in MODULE_FILES[module]:
            filename = f"data/{level}/{args.video_id}/{name}"
            print(f"  [{name}]")
            try:
                path = hf_hub_download(
                    repo_id, filename, repo_type="dataset", revision=sha, token=token
                )
                if name.endswith(".safetensors"):
                    show_safetensors(path)
                elif name.endswith(".parquet"):
                    show_parquet(path)
                else:
                    with open(path, encoding="utf-8") as stream:
                        print("    " + json.dumps(json.load(stream), ensure_ascii=False)[:1500])
            except Exception as error:
                print(f"    ERROR {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
