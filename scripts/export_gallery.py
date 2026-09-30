"""Build a local HTML gallery of retrieval candidates to verify answers by eye.

Downloads ``keyframes.tar`` only for the videos that appear among the candidates
(pinned to the keyframe revision the index was built from), extracts each candidate
keyframe with its neighbours, and writes ``index.html`` with thumbnails.

Input is a results CSV with columns ``query_id, config, rank, frame_uid`` (for example
``artifacts/eval/sotuyen1-results.csv``) and the queries CSV for the query texts.
"""

from __future__ import annotations

import argparse
import html
import io
import os
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

from multimedia_video_rag.retrieval.search import RetrievalIndex

KEYFRAME_REPO = "aqpahm/aic2026-keyframes-transnetv2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("indexes/dev"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-ids", default="", help="Comma-separated; empty = all.")
    parser.add_argument("--configs", default="C,E")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--context", type=int, default=2, help="Neighbour frames each side.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf/keyframe"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/gallery"))
    parser.add_argument("--thumb-width", type=int, default=400)
    return parser.parse_args()


def keyframe_tar(video_id: str, level: str, revision: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            KEYFRAME_REPO,
            f"data/{level}/{video_id}/keyframes.tar",
            repo_type="dataset",
            revision=revision,
            local_dir=cache_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )


def normalized(name: str) -> str:
    return PurePosixPath(str(name).replace("\\", "/")).as_posix().lstrip("./")


def extract(tar_path: Path, wanted: dict[str, Path], width: int) -> None:
    """Write thumbnails for the requested image paths (keys: image_path in frames)."""
    by_path, by_name = {}, {}
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive.getmembers():
            if member.isfile():
                name = normalized(member.name)
                by_path[name] = member
                by_name.setdefault(PurePosixPath(name).name, []).append(member)
        for image_path, target in wanted.items():
            key = normalized(image_path)
            member = by_path.get(key) or (by_name.get(PurePosixPath(key).name) or [None])[0]
            if member is None:
                raise FileNotFoundError(f"{image_path} not in {tar_path}")
            with Image.open(io.BytesIO(archive.extractfile(member).read())) as image:
                image = image.convert("RGB")
                image.thumbnail((width, width))
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target, format="JPEG", quality=85)


def clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes:02d}:{rest:02d}"


def main() -> int:
    args = parse_args()
    index = RetrievalIndex(args.index)
    revision = index.manifest["sources"]["keyframe"]["revision"]
    frames = index.frames.set_index("frame_uid")
    results = pd.read_csv(args.results, encoding="utf-8-sig")
    queries = pd.read_csv(
        args.queries, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    ).set_index("query_id")
    ids = [value for value in args.query_ids.split(",") if value] or list(
        dict.fromkeys(results["query_id"])
    )
    configs = args.configs.split(",")
    picked = results[
        results["query_id"].isin(ids)
        & results["config"].isin(configs)
        & (results["rank"] <= args.top)
    ]

    # One candidate per (query, frame), remembering which configs/ranks found it.
    candidates: dict[str, dict[str, list[str]]] = {query_id: {} for query_id in ids}
    for row in picked.itertuples(index=False):
        candidates[row.query_id].setdefault(row.frame_uid, []).append(f"{row.config}#{row.rank}")

    # Candidate frames plus neighbours within the same video.
    needed: dict[str, dict[str, Path]] = {}
    strips: dict[str, list[tuple[str, bool]]] = {}
    for per_query in candidates.values():
        for frame_uid in per_query:
            row = int(frames.loc[frame_uid, "frame_row"])
            video_id = frames.loc[frame_uid, "video_id"]
            strip = []
            for offset in range(-args.context, args.context + 1):
                neighbour = row + offset
                if not 0 <= neighbour < len(index.frames):
                    continue
                info = index.frames.iloc[neighbour]
                if info["video_id"] != video_id:
                    continue
                target = args.output_dir / "img" / f"{info['frame_uid'].replace(':', '_')}.jpg"
                needed.setdefault(video_id, {})[info["image_path"]] = target
                strip.append((info["frame_uid"], offset == 0))
            strips[frame_uid] = strip

    for position, (video_id, wanted) in enumerate(sorted(needed.items()), 1):
        missing = {path: target for path, target in wanted.items() if not target.exists()}
        if not missing:
            continue
        level = frames[frames["video_id"] == video_id]["level"].iloc[0]
        print(f"[{position}/{len(needed)}] {video_id}: downloading keyframes.tar ...", flush=True)
        extract(keyframe_tar(video_id, level, revision, args.cache_dir), missing, args.thumb_width)

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Retrieval candidates</title>",
        "<style>body{font-family:sans-serif;margin:16px;background:#fafafa}"
        "section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;"
        "margin:0 0 20px}h2{margin:0 0 4px;font-size:18px}.q{color:#333}.orig{color:#777;"
        "font-size:13px;margin:4px 0 10px}.cand{margin:10px 0;padding-top:8px;border-top:"
        "1px solid #eee}.meta{font-size:14px;margin-bottom:4px}.strip{display:flex;gap:4px;"
        "overflow-x:auto}figure{margin:0;text-align:center;font-size:11px;color:#555}"
        "img{width:220px;border:3px solid transparent;border-radius:4px}"
        "figure.hit img{border-color:#e53935}</style>",
        "<h1>Ứng viên cần xác nhận</h1>",
        "<p>Khung đỏ là frame ứng viên; hai bên là frame liền trước/sau trong cùng video.</p>",
    ]
    for query_id in ids:
        query = queries.loc[query_id]

        def field(*names: str, query=query) -> str:
            """First non-empty column among ``names`` (old and curated query formats)."""
            return next((str(query[n]) for n in names if n in query and str(query[n])), "")

        extra = ""
        if field("events_en"):
            extra += f"<div class='orig'>Events: {html.escape(field('events_en'))}</div>"
        if field("question_en"):
            extra += f"<div class='orig'>Question: {html.escape(field('question_en'))}</div>"
        parts.append(f"<section id='{html.escape(query_id)}'>")
        parts.append(
            f"<h2>{html.escape(query_id)} "
            f"<small>[{html.escape(field('query_type', 'task'))}]</small></h2>"
            f"<div class='q'>{html.escape(field('query_en', 'query_vi'))}</div>{extra}"
            f"<div class='orig'>Đề gốc: {html.escape(field('original_vi', 'notes'))}</div>"
        )
        for frame_uid, found_by in candidates[query_id].items():
            info = frames.loc[frame_uid]
            parts.append(
                "<div class='cand'><div class='meta'>"
                f"<b>{html.escape(info['video_id'])}</b> @ {clock(info['timestamp_sec'])} "
                f"({info['timestamp_sec']:.1f}s, shot {info['shot_id']}) — "
                f"{html.escape(', '.join(found_by))}</div><div class='strip'>"
            )
            for neighbour_uid, is_hit in strips[frame_uid]:
                neighbour = frames.loc[neighbour_uid]
                src = f"img/{neighbour_uid.replace(':', '_')}.jpg"
                parts.append(
                    f"<figure class='{'hit' if is_hit else ''}'><img src='{src}' loading='lazy'>"
                    f"<figcaption>{clock(neighbour['timestamp_sec'])}</figcaption></figure>"
                )
            parts.append("</div></div>")
        parts.append("</section>")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page = args.output_dir / "index.html"
    page.write_text("\n".join(parts), encoding="utf-8")
    print(f"Gallery: {page.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
