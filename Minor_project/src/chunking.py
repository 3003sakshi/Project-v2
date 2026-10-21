"""
Step 1b - merge short Whisper segments into ~40 s retrieval units, keeping exact timing.

Every chunk is a plain dict so it serialises to JSON and flows through the whole pipeline
(index, retrieval, citation validation) unchanged:

    {chunk_id, video_id, modality: "speech"|"visual", start, end, timestamp, text, ...}

Speech chunks also keep their original Whisper `segments` (start/end/text). The LLM is shown
those per-line times and citations are validated against them, so a citation is accurate to
a few seconds rather than to the whole ~40 s chunk window.
"""
from typing import Iterable, List


def format_timestamp(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_timestamp(value) -> float:
    """'mm:ss', 'h:mm:ss' or a number -> seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(p) for p in str(value).strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def make_chunk(video_id: str, modality: str, idx: int, start: float, end: float, text: str, **extra) -> dict:
    c = {
        "chunk_id": f"{video_id}:{modality}:{idx:04d}",
        "video_id": video_id,
        "modality": modality,
        "start": float(start),
        "end": float(end),
        "timestamp": format_timestamp(start),
        "text": text,
    }
    c.update(extra)
    return c


def chunk_segments(segments: Iterable, target_seconds: float = 40.0, max_chars: int = 800) -> List[dict]:
    """Greedy merge: keep adding consecutive segments until the window or char budget is hit."""
    chunks, buf = [], []

    def flush():
        if not buf:
            return
        text = " ".join(s.text for s in buf).strip()
        segs = [{"start": x.start, "end": x.end, "text": x.text} for x in buf]
        chunks.append(make_chunk(buf[0].video_id, "speech", len(chunks), buf[0].start, buf[-1].end, text,
                                 segments=segs))
        buf.clear()

    for seg in segments:
        if buf:
            span = seg.end - buf[0].start
            chars = sum(len(s.text) for s in buf) + len(seg.text)
            if span > target_seconds or chars > max_chars:
                flush()
        buf.append(seg)
    flush()
    return chunks
