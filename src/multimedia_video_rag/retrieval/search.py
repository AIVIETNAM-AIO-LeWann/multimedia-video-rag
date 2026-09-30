"""Read-only access to an index built by ``retrieval.build``.

Visual search returns ``Hit`` objects keyed by the shared ``frame_row`` id, best first.
Captions are only looked up for display.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Hit:
    frame_row: int
    score: float
    evidence: str = ""


class RetrievalIndex:
    """Read-only access to one index directory (FAISS files are loaded on first use)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest: dict[str, Any] = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.db = sqlite3.connect(
            f"file:{(self.root / self.manifest['sqlite']).as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self.frames = pd.read_sql(
            "SELECT frame_row, frame_uid, video_id, level, shot_id, timestamp_sec, image_path "
            "FROM frames ORDER BY frame_row",
            self.db,
        )
        if not self.frames["frame_row"].equals(pd.Series(range(len(self.frames)))):
            raise ValueError("frames.frame_row must be 0..N-1")
        self._faiss: dict[str, Any] = {}
        # Integer codes so temporal neighbours can be checked with vectorized numpy.
        self.video_codes = pd.factorize(self.frames["video_id"])[0]
        self.shot_ids = self.frames["shot_id"].to_numpy()

    # ---- visual -------------------------------------------------------------------------

    @property
    def visual_modules(self) -> list[str]:
        return list(self.manifest["faiss"])

    def faiss_index(self, module: str) -> Any:
        if module not in self._faiss:
            import faiss

            entry = self.manifest["faiss"][module]
            self._faiss[module] = faiss.read_index(str(self.root / entry["file"]))
        return self._faiss[module]

    def release_faiss(self) -> None:
        """Drop loaded FAISS indexes (reloaded on the next visual search)."""
        self._faiss.clear()

    def visual_batch(self, module: str, vectors: np.ndarray, k: int = 1000) -> list[list[Hit]]:
        """Search several unit-norm query vectors at once (much faster than one by one)."""
        index = self.faiss_index(module)
        queries = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        if queries.shape[1] != index.d:
            raise ValueError(f"{module}: query dim {queries.shape[1]} != index dim {index.d}")
        scores, rows = index.search(queries, k)
        return [
            [Hit(int(row), float(score)) for row, score in zip(r, s, strict=True) if row >= 0]
            for r, s in zip(rows, scores, strict=True)
        ]

    def reconstruct(self, module: str, frame_rows: np.ndarray) -> np.ndarray:
        """Stored (FP16-decoded) image vectors for the given frame rows, as float32."""
        rows = np.ascontiguousarray(frame_rows, dtype=np.int64)
        if len(rows) == 0:
            return np.empty((0, self.faiss_index(module).d), dtype=np.float32)
        return self.faiss_index(module).reconstruct_batch(rows)

    def visual(self, module: str, vector: np.ndarray, k: int = 1000) -> list[Hit]:
        return self.visual_batch(module, vector, k)[0]

    # ---- helpers ------------------------------------------------------------------------

    def describe(self, frame_rows: Iterable[int]) -> pd.DataFrame:
        rows = [int(row) for row in frame_rows]
        info = self.frames.iloc[rows].copy()
        if not rows:
            info["caption_en"] = []
            return info.reset_index(drop=True)
        captions = dict(
            self.db.execute(
                f"SELECT frame_row, caption_en FROM captions WHERE frame_row IN "
                f"({','.join('?' * len(rows))})",
                rows,
            ).fetchall()
        )
        info["caption_en"] = [captions.get(row, "") for row in rows]
        return info.reset_index(drop=True)
