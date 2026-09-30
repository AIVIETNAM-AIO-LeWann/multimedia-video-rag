"""Serve keyframe JPEGs straight from the per-video ``keyframes.tar`` (uncompressed POSIX tar).

Each archive's member table is read once and cached, then an image is a single seek + read,
so the 335k keyframes never need to be extracted to disk.
"""

from __future__ import annotations

import tarfile
import threading
from pathlib import Path, PurePosixPath

import pandas as pd


def _normalized(name: str) -> str:
    return PurePosixPath(str(name).replace("\\", "/")).as_posix().lstrip("./")


class KeyframeImages:
    def __init__(self, cache_dir: Path, frames: pd.DataFrame) -> None:
        self.cache_dir = Path(cache_dir)
        self.frames = frames
        self._members: dict[str, dict[str, tuple[int, int]]] = {}
        self._lock = threading.Lock()

    def tar_path(self, video_id: str, level: str) -> Path:
        return self.cache_dir / "data" / level / video_id / "keyframes.tar"

    def available(self, video_id: str, level: str) -> bool:
        return self.tar_path(video_id, level).is_file()

    def _table(self, path: Path) -> dict[str, tuple[int, int]]:
        key = str(path)
        with self._lock:
            if key not in self._members:
                table: dict[str, tuple[int, int]] = {}
                with tarfile.open(path, mode="r:") as archive:
                    for member in archive.getmembers():
                        if member.isfile():
                            entry = (member.offset_data, member.size)
                            name = _normalized(member.name)
                            table[name] = entry
                            table.setdefault(PurePosixPath(name).name, entry)
                self._members[key] = table
            return self._members[key]

    def read(self, frame_row: int) -> bytes:
        info = self.frames.iloc[frame_row]
        path = self.tar_path(info["video_id"], info["level"])
        if not path.is_file():
            raise FileNotFoundError(f"{path} not downloaded")
        table = self._table(path)
        name = _normalized(info["image_path"])
        entry = table.get(name) or table.get(PurePosixPath(name).name)
        if entry is None:
            raise FileNotFoundError(f"{info['image_path']} not in {path}")
        offset, size = entry
        with path.open("rb") as stream:
            stream.seek(offset)
            return stream.read(size)
