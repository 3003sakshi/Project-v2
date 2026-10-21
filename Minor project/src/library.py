"""
Persistent, single-user video library used by the CLI and the MCP server.

(The Streamlit app deliberately does NOT use this - it keeps each visitor's index in memory
for privacy. The library is for local use, where you want videos to survive restarts.)

    library/
        index/      chunks.json + vectors.npy   (the hybrid index)
        cache/      transcripts, frame descriptions, answers, thumbnails
        inbox/      the only folder the MCP `index_video` tool may read videos from
"""
import os
import re
from pathlib import Path
from typing import Optional

from .index import HybridIndex
from .pipeline import process_video

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_.\- ]{1,64}$")


def sanitize_video_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\- ]", "_", raw.strip())[:64] or "video"


def resolve_in_dir(base: Path, relative: str) -> Path:
    """Resolve `relative` inside `base`; refuse anything that escapes it (../, absolute paths, symlinks)."""
    base = base.resolve()
    p = (base / relative).resolve()
    if p != base and base not in p.parents:
        raise ValueError("Path is outside the allowed directory")
    return p


class Library:
    def __init__(self, root: Optional[str] = None, embedder=None):
        self.root = Path(root or os.environ.get("INSIGHTRAG_LIBRARY", "library")).resolve()
        self.index_dir = self.root / "index"
        self.cache_dir = self.root / "cache"
        self.inbox = self.root / "inbox"
        for d in (self.index_dir, self.cache_dir, self.inbox):
            d.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder

    def load_index(self) -> HybridIndex:
        # Reload from disk each time so the MCP server sees videos added by the CLI meanwhile.
        return HybridIndex.load(str(self.index_dir), embedder=self._embedder)

    def add_video(self, video_path: str, video_id: str, **opts) -> dict:
        if not VIDEO_ID_RE.match(video_id):
            raise ValueError("video_id may only contain letters, digits, space, _ . - (max 64 chars)")
        chunks, info = process_video(video_path, video_id, str(self.cache_dir), **opts)
        idx = self.load_index()
        idx.remove_video(video_id)  # re-indexing replaces the old version
        idx.add_chunks(chunks)
        idx.save(str(self.index_dir))
        return info

    def delete_video(self, video_id: str) -> None:
        idx = self.load_index()
        idx.remove_video(video_id)
        idx.save(str(self.index_dir))
