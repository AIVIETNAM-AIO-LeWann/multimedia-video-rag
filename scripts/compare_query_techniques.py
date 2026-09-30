"""Compare query techniques on the curated queries and write a visual gallery (no labels).

Techniques, all on SigLIP 2 + BEiT-3 with weighted RRF over the full corpus:

- ``baseline``: the full English query;
- ``prf``: pseudo-relevance feedback. The image vectors of the baseline's top ``--prf-k``
  keyframes are averaged per encoder into a seed; the seed's ranking is fused with the
  baseline ranking by RRF;
- ``decomposition``: each sub-query (``#start``/``#end`` or ``#E1..#E4`` in the vectors
  file) is searched on its own; the best ordered chain in one video is found by dynamic
  programming over each sub-query's top ``--event-k`` keyframes, with every step strictly
  later in time and at most ``--max-gap`` seconds after the previous one.

Vectors come from notebooks/retrieval/encode-query-vectors.ipynb. The output is for
looking at results side by side; it measures nothing without ground truth.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from multimedia_video_rag.retrieval.keyframe_images import KeyframeImages
from multimedia_video_rag.retrieval.query_vectors import load_query_vectors
from multimedia_video_rag.retrieval.search import RetrievalIndex
from multimedia_video_rag.retrieval.system import (
    MODULES,
    DenseScorer,
    SearchResult,
    dual_encoder_search,
)

RRF_K = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("indexes/dev"))
    parser.add_argument(
        "--queries", type=Path, default=Path("artifacts/eval/aic2026-visual-queries.csv")
    )
    parser.add_argument(
        "--query-vectors",
        type=Path,
        default=Path("artifacts/eval/encode/aic2026-query_vectors.npz"),
    )
    parser.add_argument("--keyframes", type=Path, default=Path("data/hf/keyframe"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/techniques"))
    parser.add_argument("--top", type=int, default=5, help="Results shown per technique.")
    parser.add_argument("--prf-k", type=int, default=3)
    parser.add_argument("--event-k", type=int, default=300)
    parser.add_argument("--max-gap", type=float, default=60.0)
    parser.add_argument("--thumb-width", type=int, default=240)
    return parser.parse_args()


def event_order(key: str) -> int:
    """``#start`` < ``#E1`` < ``#E2`` < ... < ``#end``."""
    part = key.split("#", 1)[1]
    if part == "start":
        return 0
    if part == "end":
        return 1_000
    return int(re.sub(r"\D", "", part) or 0)


def rrf(*rankings: list[int]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, 1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda row: (-scores[row], row))


def collapse(index: RetrievalIndex, rows: list[int], limit: int) -> list[int]:
    seen, kept = set(), []
    for row in rows:
        key = (int(index.video_codes[row]), int(index.shot_ids[row]))
        if key not in seen:
            seen.add(key)
            kept.append(row)
            if len(kept) == limit:
                break
    return kept


def search(index, scorer, vectors, limit=1000) -> list[SearchResult]:
    return dual_encoder_search(
        index, vectors, fusion="rrf", collapse_shots=False, limit=limit, scorer=scorer
    )


def prf(index, scorer, baseline: list[SearchResult], k: int) -> list[int]:
    seeds = collapse(index, [r.frame_row for r in baseline], k)
    seed_vectors = {}
    for module in MODULES:
        mean = scorer.matrices[module][seeds].astype(np.float32).mean(axis=0)
        seed_vectors[module] = mean / np.linalg.norm(mean)
    expanded = search(index, scorer, seed_vectors)
    return rrf([r.frame_row for r in baseline], [r.frame_row for r in expanded])


def best_chains(index, events: list[list[SearchResult]], max_gap: float, limit: int):
    """Ordered chains (one keyframe per event) in one video; score = sum of RRF scores."""
    times = index.frames["timestamp_sec"].to_numpy()
    codes = index.video_codes
    # DP state per event: {row: (total, previous_row)}
    state = {r.frame_row: (r.score, None) for r in events[0]}
    history = [state]
    for hits in events[1:]:
        by_video: dict[int, list[tuple[float, int]]] = {}
        for row in state:
            by_video.setdefault(int(codes[row]), []).append((times[row], row))
        nxt = {}
        for hit in hits:
            row = hit.frame_row
            best = None
            for t_prev, prev in by_video.get(int(codes[row]), []):
                if 0 < times[row] - t_prev <= max_gap:
                    total = state[prev][0]
                    if best is None or total > best[0]:
                        best = (total, prev)
            if best is not None:
                nxt[row] = (best[0] + hit.score, best[1])
        state = nxt
        history.append(state)
    chains, used_videos = [], set()
    for row, (total, _) in sorted(state.items(), key=lambda item: -item[1][0]):
        if int(codes[row]) in used_videos:
            continue
        chain, current = [row], row
        for level in range(len(history) - 1, 0, -1):
            current = history[level][current][1]
            chain.append(current)
        chains.append((total, chain[::-1]))
        used_videos.add(int(codes[row]))
        if len(chains) == limit:
            break
    return chains


def main() -> int:
    args = parse_args()
    index = RetrievalIndex(args.index)
    scorer = DenseScorer(index, Path("artifacts/cache") / args.index.resolve().name)
    images = KeyframeImages(args.keyframes, index.frames)
    loaded = load_query_vectors(args.query_vectors, index.manifest)
    vectors = {
        query.query_id: {m: matrix[position] for m, matrix in loaded.vectors.items()}
        for position, query in enumerate(loaded.queries)
    }
    texts = {query.query_id: query.text_en for query in loaded.queries}
    table = pd.read_csv(args.queries, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    out_dir = args.output
    (out_dir / "img").mkdir(parents=True, exist_ok=True)

    results, summary = [], []
    for query in table.itertuples():
        qid = query.query_id
        baseline = search(index, scorer, vectors[f"{qid}#full"])
        base_rows = collapse(index, [r.frame_row for r in baseline], args.top)
        prf_rows = collapse(index, prf(index, scorer, baseline, args.prf_k), args.top)
        event_ids = sorted(
            (key for key in vectors if key.startswith(f"{qid}#") and not key.endswith("#full")),
            key=event_order,
        )
        # TRAKE queries carry both splits; their numbered events are the finer one.
        numbered = [key for key in event_ids if key.split("#")[1].startswith("E")]
        event_ids = numbered or event_ids
        chains = []
        if len(event_ids) >= 2:
            events = [search(index, scorer, vectors[e], limit=args.event_k) for e in event_ids]
            chains = best_chains(index, events, args.max_gap, 3)
        video = lambda row: index.frames.video_id.iat[row]  # noqa: E731
        summary.append(
            {
                "query_id": qid,
                "structure": query.structure,
                "baseline_top1_video": video(base_rows[0]),
                "prf_top1_video": video(prf_rows[0]),
                "decomposition_top1_video": video(chains[0][1][0]) if chains else "",
                "decomposed": len(event_ids) >= 2,
            }
        )
        results.append((query, base_rows, prf_rows, event_ids, chains))
        print(
            f"{qid}: baseline {video(base_rows[0])}, prf {video(prf_rows[0])}"
            + (f", chain {video(chains[0][1][0])}" if chains else ""),
            flush=True,
        )

    frame = pd.DataFrame(summary)
    frame.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    def thumb(row: int) -> str:
        info = index.frames.iloc[row]
        name = f"img/{info.frame_uid.replace(':', '_')}.jpg"
        target = out_dir / name
        if not target.exists():
            from PIL import Image

            with Image.open(io.BytesIO(images.read(row))) as image:
                image = image.convert("RGB")
                image.thumbnail((args.thumb_width, args.thumb_width))
                image.save(target, format="JPEG", quality=82)
        seconds = int(info.timestamp_sec)
        return (
            f'<figure><img loading="lazy" src="{name}" alt=""><figcaption>{info.video_id} · '
            f"{seconds // 60}:{seconds % 60:02d}</figcaption></figure>"
        )

    def strip(rows) -> str:
        return '<div class="strip">' + "".join(thumb(r) for r in rows) + "</div>"

    blocks = []
    for query, base_rows, prf_rows, event_ids, chains in results:
        vi = html.escape(getattr(query, "original_vi", ""))
        parts = [
            f'<section id="{query.query_id}"><h2>{query.query_id} · {query.task} · '
            f"{query.structure}</h2><p class='vi'>{vi}</p>"
            f"<p class='en'>{html.escape(query.query_en)}</p>",
            f"<h3>Baseline</h3>{strip(base_rows)}",
            f"<h3>PRF (top {args.prf_k} làm ảnh mẫu)</h3>{strip(prf_rows)}",
        ]
        if event_ids:
            subs = "".join(
                f"<li><b>{e.split('#')[1]}</b>: {html.escape(texts[e])}</li>" for e in event_ids
            )
            parts.append(f"<h3>Decomposition</h3><ol class='subs'>{subs}</ol>")
            if chains:
                for rank, (_, chain) in enumerate(chains, 1):
                    parts.append(f"<p class='chain'>Chuỗi #{rank}</p>{strip(chain)}")
            else:
                parts.append("<p class='none'>Không tìm được chuỗi thoả điều kiện.</p>")
        parts.append("</section>")
        blocks.append("".join(parts))

    decomposed = frame[frame.decomposed]
    stats = {
        "queries": len(frame),
        "prf_same_top1_video": int((frame.baseline_top1_video == frame.prf_top1_video).sum()),
        "decomposed": len(decomposed),
        "chain_found": int((decomposed.decomposition_top1_video != "").sum()),
        "chain_same_top1_video": int(
            (decomposed.baseline_top1_video == decomposed.decomposition_top1_video).sum()
        ),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    nav = " ".join(f'<a href="#{q.query_id}">{q.query_id}</a>' for q, *_ in results)
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Query techniques</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px;
  background:#f6f7f9;color:#1b1f24}}
section{{background:#fff;border:1px solid #d9dde3;border-radius:8px;padding:12px 16px;
  margin:16px 0}}
h2{{margin:0 0 4px;font-size:17px}} h3{{margin:12px 0 4px;font-size:14px;color:#2458d6}}
.vi{{margin:4px 0;padding:6px 10px;border-left:3px solid #2458d6;background:#e5ecfc}}
.en,.subs{{color:#5d6673;margin:4px 0}} .chain{{margin:8px 0 0;font-weight:600}}
.none{{color:#b3261e}}
.strip{{display:flex;gap:6px;overflow-x:auto}} figure{{margin:0;flex:0 0 {args.thumb_width}px}}
figure img{{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px;background:#ddd}}
figcaption{{font-size:12px;color:#5d6673}} nav a{{margin-right:6px}}
</style></head><body><h1>So sánh kỹ thuật truy vấn</h1>
<p>Chưa có đáp án chuẩn: trang này chỉ để xem bằng mắt. {json.dumps(stats, ensure_ascii=False)}</p>
<nav>{nav}</nav>{"".join(blocks)}</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Gallery: {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
