"""Local web UI for the baseline SigLIP 2 + BEiT-3 search (experimental).

Query text is encoded by notebooks/retrieval/query-encoder-server.ipynb running on
Colab/Kaggle (``--encoder-url``; key in the ``ENCODER_KEY`` environment variable). Without
a server, queries listed in ``--queries`` can still be searched when their vectors are in
``--query-vectors`` (ids ``<id>`` or ``<id>#full``).

Images are read from ``data/hf/keyframe`` (scripts/download_keyframes.py). Answers marked in
the UI are appended to ``--labels`` as CSV.

    $env:ENCODER_KEY = "..."
    uv run python scripts/serve_ui.py --encoder-url https://xxx.trycloudflare.com
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import threading
import time
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from multimedia_video_rag.retrieval.keyframe_images import KeyframeImages
from multimedia_video_rag.retrieval.query_vectors import load_query_vectors
from multimedia_video_rag.retrieval.remote_encoder import RemoteEncoder, RemoteEncoderError
from multimedia_video_rag.retrieval.search import RetrievalIndex
from multimedia_video_rag.retrieval.system import (
    FUSIONS,
    DenseScorer,
    dual_encoder_search,
    video_context,
)

PAGE = Path(__file__).resolve().parents[1] / "src/multimedia_video_rag/retrieval/ui.html"
LABEL_FIELDS = [
    "labeled_at",
    "query_id",
    "query_en",
    "video_id",
    "frame_uid",
    "timestamp_sec",
    "fusion",
    "weight_siglip",
    "rank",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("indexes/dev"))
    parser.add_argument("--keyframes", type=Path, default=Path("data/hf/keyframe"))
    parser.add_argument("--encoder-url", help="URL printed by query-encoder-server.ipynb")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("artifacts/eval/aic2026-visual-queries.csv"),
        help="CSV with query_id, query_en (listed in the UI; optional)",
    )
    parser.add_argument(
        "--query-vectors",
        type=Path,
        default=Path("artifacts/eval/encode/aic2026-query_vectors.npz"),
        help="Pre-encoded vectors used when the encoder server is off (optional)",
    )
    parser.add_argument(
        "--matrix-cache",
        type=Path,
        help="FP16 vector cache for full-corpus scoring (default: artifacts/cache/<index name>)",
    )
    parser.add_argument("--labels", type=Path, default=Path("artifacts/labels/labels.csv"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    return parser.parse_args()


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.index = RetrievalIndex(args.index)
        self.images = KeyframeImages(args.keyframes, self.index.frames)
        cache = args.matrix_cache or Path("artifacts/cache") / args.index.resolve().name
        self.scorer = DenseScorer(self.index, cache)
        self.lock = threading.Lock()  # sqlite connection and FAISS are shared by threads
        self.encoder = None
        self.encoder_status = "off"
        key = os.environ.get("ENCODER_KEY", "")
        if args.encoder_url:
            if not key:
                raise SystemExit("Set the ENCODER_KEY environment variable")
            self.encoder = RemoteEncoder(args.encoder_url, key, self.index.manifest)
            try:
                meta = self.encoder.health()
                self.encoder_status = f"online ({meta['device']})"
            except RemoteEncoderError as error:
                self.encoder_status = f"error: {error}"
        self.queries: list[dict[str, str]] = []
        if args.queries and args.queries.is_file():
            table = pd.read_csv(
                args.queries, dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
            if "original_vi" not in table.columns:
                table["original_vi"] = ""
            # original_vi is shown to the user only; search always encodes query_en.
            self.queries = [
                {
                    "query_id": row.query_id,
                    "query_en": row.query_en,
                    "original_vi": row.original_vi,
                }
                for row in table.itertuples()
                if row.query_en.strip()
            ]
        # query_id -> (encoded English text, vectors); ``<id>#full`` also answers ``<id>``.
        self.stored: dict[str, tuple[str, dict[str, np.ndarray]]] = {}
        if args.query_vectors and args.query_vectors.is_file():
            loaded = load_query_vectors(args.query_vectors, self.index.manifest)
            for position, query in enumerate(loaded.queries):
                vectors = {m: matrix[position] for m, matrix in loaded.vectors.items()}
                if set(vectors) == {"siglip", "beit3"} and all(
                    not np.isnan(v).any() for v in vectors.values()
                ):
                    entry = (" ".join(query.text_en.split()), vectors)
                    self.stored[query.query_id] = entry
                    self.stored.setdefault(query.query_id.removesuffix("#full"), entry)

    def info(self) -> dict:
        downloaded = sum(1 for _ in self.args.keyframes.glob("data/*/*/keyframes.tar"))
        return {
            "videos": self.index.manifest["video_count"],
            "frames": self.index.manifest["frame_count"],
            "images_downloaded": downloaded,
            "encoder": self.encoder_status,
            "fusions": list(FUSIONS),
            "queries": [
                {
                    **query,
                    "stored": self.stored.get(query["query_id"], ("",))[0]
                    == " ".join(query["query_en"].split()),
                }
                for query in self.queries
            ],
        }

    def vectors_for(self, text: str, query_id: str) -> tuple[dict[str, np.ndarray], str]:
        stored = self.stored.get(query_id)
        if self.encoder is None and stored and stored[0] == " ".join(text.split()):
            return stored[1], "stored"
        if self.encoder is None:
            raise RemoteEncoderError(
                "No encoder server (--encoder-url); only unedited listed queries with "
                "stored vectors can be searched"
            )
        return self.encoder.encode(text), "server"

    def rows_payload(self, rows: list[int]) -> list[dict]:
        with self.lock:
            info = self.index.describe(rows)
        return [
            {
                "row": int(row),
                "video_id": item.video_id,
                "frame_uid": item.frame_uid,
                "time": round(float(item.timestamp_sec), 2),
                "clock": clock(item.timestamp_sec),
                "shot": int(item.shot_id),
                "caption": item.caption_en,
                "image": self.images.available(item.video_id, item.level),
            }
            for row, item in zip(rows, info.itertuples(), strict=True)
        ]

    def search(self, body: dict) -> dict:
        text = str(body.get("query", "")).strip()
        fusion = str(body.get("fusion", "rrf"))
        weight = float(body.get("weight_siglip", 0.5))
        limit = max(1, min(int(body.get("limit", 100)), 500))
        started = time.perf_counter()
        vectors, source = self.vectors_for(text, str(body.get("query_id", "")))
        encoded = time.perf_counter()
        with self.lock:
            results = dual_encoder_search(
                self.index,
                vectors,
                fusion=fusion,
                weight_siglip=weight,
                collapse_shots=bool(body.get("collapse_shots", True)),
                limit=limit,
                scorer=self.scorer,
            )
        searched = time.perf_counter()
        items = self.rows_payload([r.frame_row for r in results])
        for item, result in zip(items, results, strict=True):
            item.update(score=result.score, similarity=result.similarity, rank=result.rank)
        return {
            "results": items,
            "vector_source": source,
            "encode_ms": round((encoded - started) * 1000),
            "search_ms": round((searched - encoded) * 1000),
        }

    def context(self, row: int, radius: int) -> dict:
        rows = [int(r) for r in video_context(self.index, row, radius=radius)]
        return {"center": row, "frames": self.rows_payload(rows)}

    def label(self, body: dict) -> dict:
        row = int(body["row"])
        info = self.index.frames.iloc[row]
        record = {
            "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "query_id": str(body.get("query_id", "")),
            "query_en": str(body.get("query", "")),
            "video_id": info.video_id,
            "frame_uid": info.frame_uid,
            "timestamp_sec": round(float(info.timestamp_sec), 2),
            "fusion": str(body.get("fusion", "")),
            "weight_siglip": body.get("weight_siglip", ""),
            "rank": body.get("rank", ""),
            "note": str(body.get("note", "")),
        }
        path = self.args.labels
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            new = not path.exists()
            with path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(record)
        return {"saved": str(path), **record}


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            pass

        def send_body(self, status: int, body: bytes, content_type: str, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_body(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            params = parse_qs(url.query)
            try:
                if url.path == "/":
                    body = PAGE.read_bytes()
                    self.send_body(HTTPStatus.OK, body, "text/html; charset=utf-8")
                elif url.path == "/api/info":
                    self.send_json(app.info())
                elif url.path == "/api/context":
                    row = int(params["row"][0])
                    radius = min(int(params.get("radius", ["8"])[0]), 50)
                    self.send_json(app.context(row, radius))
                elif url.path.startswith("/img/"):
                    row = int(url.path.removeprefix("/img/"))
                    self.send_body(HTTPStatus.OK, app.images.read(row), "image/jpeg", cache=True)
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, KeyError, ValueError, IndexError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/search":
                    self.send_json(app.search(body))
                elif self.path == "/api/label":
                    self.send_json(app.label(body))
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except RemoteEncoderError as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
            except (KeyError, ValueError, IndexError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> int:
    args = parse_args()
    print("Loading index ...", flush=True)
    app = App(args)
    info = app.info()
    print(
        f"Index: {info['videos']} videos, {info['frames']} keyframes | images for "
        f"{info['images_downloaded']} videos | encoder: {info['encoder']} | "
        f"stored query vectors: {len(app.stored)}",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Open http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
