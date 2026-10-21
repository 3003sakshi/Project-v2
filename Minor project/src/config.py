"""Central configuration. Everything can be overridden with environment variables / .env"""
import os

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", GEMINI_MODEL)

# Retrieval defaults
RRF_K = 60            # standard constant from Cormack et al. (2009)
CANDIDATE_DEPTH = 50  # how many results each individual ranker contributes to fusion

# Citation validation: seconds of slack when checking a cited time against a chunk window
CITATION_TOLERANCE = 2.0

# Retrieval / processing settings shared by the web app and the warm-up tool
FRAME_INTERVAL = 5.0   # seconds between sampled frames (near-duplicates are skipped automatically)
TOP_K_SINGLE = 6       # evidence chunks per question for one video


def top_k_for(n_videos: int) -> int:
    """More videos in scope -> a few more evidence chunks, so one video does not crowd out the others."""
    return min(12, TOP_K_SINGLE + 2 * max(0, n_videos - 1))
