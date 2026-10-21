"""Step 4 - embedder loading and the context block handed to the LLM."""
import re
from functools import lru_cache
from typing import List, Optional

from .chunking import format_timestamp
from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build_context_block(results: List[dict]) -> str:
    """Numbered, modality-tagged sources. The time window is what the LLM may cite from."""
    parts = []
    for n, r in enumerate(results, 1):
        if r["modality"] == "visual":
            head = (f"[SOURCE {n}] video={r['video_id']} | type=VISUAL (what was on screen) | "
                    f"frame at {format_timestamp(r['start'])}")
            body = r["text"]
        else:
            head = (f"[SOURCE {n}] video={r['video_id']} | type=SPEECH (what was said) | "
                    f"window {format_timestamp(r['start'])}-{format_timestamp(r['end'])}")
            segs = r.get("segments")
            body = ("\n".join(f"[{format_timestamp(s['start'])}] {s['text']}" for s in segs)
                    if segs else r["text"])
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)


def match_video_title(question: str, titles: List[str]) -> Optional[str]:
    """
    If the question names exactly one video by its title ("in Lecture 2, what is ..."), return it so the
    search can be limited to that video. Whole-word, case-insensitive; the longest title wins
    ("Lecture 10" beats "Lecture 1"). Returns None if no title - or several different ones - appear.
    """
    hits = [t for t in titles if t.strip()
            and re.search(rf"(?<!\w){re.escape(t.strip())}(?!\w)", question, re.IGNORECASE)]
    hits = [t for t in hits if not any(t != u and t.lower() in u.lower() for u in hits)]
    return hits[0] if len(hits) == 1 else None
