"""
Tiny on-disk JSON cache.

Intermediate outputs (transcripts, VLM frame descriptions, LLM answers) are expensive:
Whisper is slow on a CPU-only laptop and the Gemini free tier is rate-limited. Everything
that is expensive is cached under a key derived from the video's *content hash*, so
re-indexing the same file (or re-running an evaluation) costs nothing.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


def file_sha256(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class DiskCache:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        # Always hash the key: raw keys contain ':' (illegal in Windows file names) and can be long.
        return d / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, namespace: str, key: str, value: Any) -> None:
        p = self._path(namespace, key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)  # atomic: never leaves a half-written cache file
