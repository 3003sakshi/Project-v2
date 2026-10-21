"""
Step 5 - answer synthesis + programmatic citation validation.

The LLM is told to cite as [video_id @ mm:ss]. `validate_citations` then checks every cited
time against the chunks that were actually retrieved: a citation is VALID only if its video
matches and the time falls inside (or within a small tolerance of) a retrieved chunk's window.
This turns "the answer looks grounded" into a measurable number.
"""
import re
from typing import List, Optional

from . import llm
from .cache import DiskCache, text_sha256
from .chunking import parse_timestamp
from .config import CITATION_TOLERANCE, GEMINI_MODEL
from .retrieve import build_context_block

# [lecture1 @ 01:23]   [01:23]   **[lecture1 @ 1:02:03]**
CITATION_RE = re.compile(r"\[\s*(?:([^\[\]|@]+?)\s*[@|]\s*)?(\d{1,2}(?::\d{2}){1,2})\s*\]")

SYSTEM_PROMPT = (
    "You answer questions about videos using ONLY the numbered sources below. Sources are either "
    "SPEECH (transcript excerpts) or VISUAL (descriptions and on-screen text of video frames).\n"
    "Rules:\n"
    "1. Use only information in the sources. If the sources do not contain the answer, reply with the single "
    "word NOT_FOUND followed by one short sentence, and cite nothing.\n"
    "2. After every sentence that states a fact, cite its evidence as [video_id @ mm:ss]. For SPEECH "
    "sources use the [mm:ss] shown at the start of the transcript line that supports the claim; for VISUAL "
    "sources use the frame time. Never invent a time that is not shown in the sources. "
    "Example: [lecture1 @ 02:15].\n"
    "3. Put one citation per bracket; use multiple brackets for multiple sources.\n"
    "4. When a source is VISUAL, make clear it was shown on screen rather than said aloud.\n"
    "5. Be concise."
)


def _prompt(query: str, context: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nSOURCES:\n\n{context}\n\nQUESTION: {query}"


_NOT_FOUND_RE = re.compile(r"^\s*NOT_FOUND\b[:.\-\s]*")
_NO_INFO_RE = re.compile(
    r"\b(do(es)?\s*n['’]?t|do(es)?\s+not|cannot|can['’]?t|no)\b.{0,60}"
    r"\b(contain|mention|information|answer|cover|discuss|find|say)\b", re.IGNORECASE)


def is_abstention(raw_text: str, validation: dict) -> bool:
    """True if the model said the video does not contain the answer (explicit marker, or a clear
    'no information' reply that cites nothing)."""
    if _NOT_FOUND_RE.match(raw_text):
        return True
    return validation["n_citations"] == 0 and bool(_NO_INFO_RE.search(raw_text[:300]))


def _evidence_windows(r: dict):
    """Fine-grained time windows a citation may fall in: per-segment for speech, else the chunk window."""
    segs = r.get("segments")
    if segs:
        return [(x["start"], x["end"]) for x in segs]
    return [(r["start"], r["end"])]


def validate_citations(answer: str, results: List[dict], tolerance: float = CITATION_TOLERANCE) -> dict:
    citations = []
    for m in CITATION_RE.finditer(answer):
        vid = (m.group(1) or "").strip() or None
        secs = parse_timestamp(m.group(2))
        match = None
        for r in results:
            if vid is not None and r["video_id"] != vid:
                continue
            if any(a - tolerance <= secs <= b + tolerance for a, b in _evidence_windows(r)):
                match = r
                break
        citations.append({
            "raw": m.group(0), "video_id": vid, "seconds": secs, "valid": match is not None,
            "chunk_id": match["chunk_id"] if match else None,
            "modality": match["modality"] if match else None,
        })

    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.split()) >= 4]
    grounded = sum(1 for s in sentences if any(
        c["valid"] for c in citations if c["raw"] in s))
    n_valid = sum(c["valid"] for c in citations)
    return {
        "citations": citations,
        "n_citations": len(citations),
        "n_valid": n_valid,
        "validity_rate": (n_valid / len(citations)) if citations else None,
        "n_sentences": len(sentences),
        "n_grounded_sentences": grounded,
        "grounded_rate": (grounded / len(sentences)) if sentences else None,
    }


def generate_answer(query: str, results: List[dict], cache: Optional[DiskCache] = None,
                    model: str = GEMINI_MODEL) -> Optional[dict]:
    """Return {text, model, cached, validation, abstained} or None if there is no API key / no results."""
    if not results or not llm.available():
        return None
    prompt = _prompt(query, build_context_block(results))
    key = text_sha256(model, prompt)
    text = cache.get("answers", key) if cache else None
    cached = text is not None
    if text is None:
        text = llm.generate(prompt, model=model)
        if cache and text:
            cache.put("answers", key, text)
    clean = _NOT_FOUND_RE.sub("", text, count=1).strip() or text
    validation = validate_citations(clean, results)
    return {"text": clean, "model": model, "cached": cached, "validation": validation,
            "abstained": is_abstention(text, validation)}
