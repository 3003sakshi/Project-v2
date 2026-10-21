"""
MCP server: exposes the video collection to any MCP-compatible assistant (Claude Desktop,
IDE agents, ...) through a small set of *controlled* tools.

Run (stdio transport):   python -m src.mcp_server

Design for safety/control:
  * read-only tools by default (list / search / ask / transcript window)
  * the one write tool, `index_video`, can only read files inside <library>/inbox - an
    assistant cannot make it open arbitrary paths on your machine
  * inputs are validated and clamped (top_k, window size, id format, question length)
  * answers carry validated timestamp citations; nothing is returned that was not retrieved
  * logs go to stderr only - stdout is the protocol channel for stdio servers
"""
import logging
import sys

from mcp.server.fastmcp import FastMCP

from . import llm
from .answer import generate_answer
from .cache import DiskCache
from .chunking import format_timestamp
from .library import VIDEO_ID_RE, Library, resolve_in_dir

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("insightrag.mcp")

mcp = FastMCP("insightrag")
_lib = None

MAX_TOP_K = 10
MAX_WINDOW_SECONDS = 600
MAX_QUESTION_CHARS = 500


def lib() -> Library:
    global _lib
    if _lib is None:
        _lib = Library()
    return _lib


def _modalities(modality: str):
    if modality == "speech":
        return ("speech",)
    if modality == "visual":
        return ("visual",)
    return ("speech", "visual")


def _check_video(index, video_id):
    if video_id is not None and video_id not in index.videos():
        raise ValueError(f"Unknown video_id '{video_id}'. Call list_videos to see what is indexed.")


def _public(r: dict) -> dict:
    return {
        "video_id": r["video_id"],
        "modality": r["modality"],
        "timestamp": format_timestamp(r["start"]),
        "start_seconds": round(r["start"], 1),
        "end_seconds": round(r["end"], 1),
        "text": r["text"],
        "relevance": round(r["rel"], 3),
    }


@mcp.tool()
def list_videos() -> dict:
    """List the indexed videos and how many speech/visual segments each has."""
    return {"videos": lib().load_index().stats()}


@mcp.tool()
def search_video(query: str, video_id: str = None, top_k: int = 5,
                 modality: str = "both", mode: str = "hybrid") -> dict:
    """
    Find the most relevant moments in the indexed videos.

    Args:
        query: natural-language search query.
        video_id: restrict to one video (see list_videos); omit to search all.
        top_k: number of moments to return (1-10).
        modality: "speech" (what was said), "visual" (what was on screen) or "both".
        mode: "hybrid" (BM25 + dense, fused with RRF), "dense", or "bm25".
    Returns moments with timestamps, so you can point the user to exact positions.
    """
    query = query.strip()[:MAX_QUESTION_CHARS]
    if not query:
        raise ValueError("query is empty")
    if modality not in ("speech", "visual", "both") or mode not in ("hybrid", "dense", "bm25"):
        raise ValueError("invalid modality or mode")
    index = lib().load_index()
    _check_video(index, video_id)
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    res = index.search(query, top_k=top_k, mode=mode, modalities=_modalities(modality), video_id=video_id)
    return {"query": query, "results": [_public(r) for r in res]}


@mcp.tool()
def ask_video(question: str, video_id: str = None, top_k: int = 5) -> dict:
    """
    Answer a question about the videos with timestamp citations validated against the
    retrieved evidence. Needs GEMINI_API_KEY; without it, returns the evidence only.
    """
    question = question.strip()[:MAX_QUESTION_CHARS]
    if not question:
        raise ValueError("question is empty")
    index = lib().load_index()
    _check_video(index, video_id)
    res = index.search(question, top_k=max(1, min(int(top_k), MAX_TOP_K)), video_id=video_id)
    out = {"question": question, "evidence": [_public(r) for r in res]}
    if not llm.available():
        out["note"] = "GEMINI_API_KEY not set - returning evidence only."
        return out
    ans = generate_answer(question, res, cache=DiskCache(str(lib().cache_dir)))
    if ans:
        v = ans["validation"]
        out["answer"] = ans["text"]
        out["found_in_video"] = not ans["abstained"]
        out["citation_check"] = {
            "citations": v["n_citations"], "valid": v["n_valid"],
            "invalid": [c["raw"] for c in v["citations"] if not c["valid"]],
        }
    return out


@mcp.tool()
def get_transcript_window(video_id: str, start: str, end: str) -> dict:
    """
    Return everything indexed (speech + visual) that overlaps [start, end] of a video.
    start/end accept seconds or mm:ss. Windows are capped at 10 minutes.
    """
    from .chunking import parse_timestamp

    index = lib().load_index()
    _check_video(index, video_id)
    if video_id is None:
        raise ValueError("video_id is required")
    s, e = parse_timestamp(start), parse_timestamp(end)
    if e <= s:
        raise ValueError("end must be after start")
    e = min(e, s + MAX_WINDOW_SECONDS)
    hits = [c for c in index.chunks if c["video_id"] == video_id and c["start"] < e and c["end"] > s]
    hits.sort(key=lambda c: (c["start"], c["modality"]))
    return {"video_id": video_id, "window": [format_timestamp(s), format_timestamp(e)],
            "segments": [_public({**c, "rel": 1.0}) for c in hits]}


@mcp.tool()
def index_video(filename: str, video_id: str, analyze_visuals: bool = True) -> dict:
    """
    Index a video file that has been placed in the library's inbox folder.
    Long-running (transcription on CPU). `filename` is relative to the inbox; other paths are refused.
    """
    if not VIDEO_ID_RE.match(video_id):
        raise ValueError("video_id may only contain letters, digits, space, _ . - (max 64 chars)")
    path = resolve_in_dir(lib().inbox, filename)
    if not path.is_file():
        raise ValueError(f"'{filename}' not found in the inbox folder ({lib().inbox})")
    info = lib().add_video(str(path), video_id, use_visual=analyze_visuals)
    return {"video_id": video_id, **info}


@mcp.resource("insightrag://videos")
def videos_resource() -> str:
    """Read-only overview of the indexed collection."""
    stats = lib().load_index().stats()
    if not stats:
        return "No videos indexed yet."
    return "\n".join(f"{v}: {', '.join(f'{n} {m} chunks' for m, n in s.items())}" for v, s in stats.items())


if __name__ == "__main__":
    mcp.run()  # stdio transport
