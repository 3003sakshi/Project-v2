"""
Demo preparation: do ALL the paid/slow work ahead of time so the live demo needs (almost) no API calls.

    python -m src.warmup --video test.mp4 --id "Intro to cv" --questions demo_questions.txt
    # several videos: repeat --video/--id in the order you will upload them
    python -m src.warmup --video a.mp4 --id "Lecture 1" --video b.mp4 --id "Lecture 2" --questions q.txt

What it does (using exactly the same settings as the web app):
  1. transcribes the video and describes its slides/frames  -> saved in the cache
  2. asks every question in your file and stores each written answer -> saved in the cache

On demo day the app finds all of this in the cache, so processing is fast and your prepared
questions are answered instantly with ZERO API requests.

IMPORTANT for the cache to be hit on demo day:
  * use the same video file, the same --id (title), and the same question wording
  * keep INSIGHTRAG_CACHE_DIR set to the same folder in .env
"""
import argparse
import os
from typing import List, Optional

from . import llm
from .answer import generate_answer
from .cache import DiskCache
from .index import HybridIndex
from .library import sanitize_video_id
from .config import FRAME_INTERVAL, top_k_for
from .pipeline import process_video
from .retrieve import match_video_title


def read_questions(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def warm(videos: List[tuple], questions: List[str], cache_dir: str,
         use_visual: Optional[bool] = None, say=print) -> dict:
    """videos: [(path, title), ...] - add them in the same order you will upload them in the app."""
    use_visual = llm.available() if use_visual is None else use_visual
    index = HybridIndex()
    for path, video_id in videos:
        say(f"== {video_id}")
        chunks, info = process_video(path, video_id, cache_dir, use_visual=use_visual,
                                     frame_interval=FRAME_INTERVAL,
                                     progress=lambda f, t: say(f"[{int(f * 100):3d}%] {t}"))
        say(f"Indexed: {info['speech_chunks']} spoken + {info['visual_chunks']} visual moments")
        for w in info["warnings"]:
            say(f"WARNING: {w}")
        index.add_chunks(chunks)
    titles = index.videos()
    cache = DiskCache(cache_dir)
    stats = {"new_requests": 0, "from_cache": 0, "failed": 0}
    for q in questions:
        target = match_video_title(q, titles)      # same routing as the app
        results = index.search(q, top_k=top_k_for(1 if target else len(titles)), mode="hybrid", video_id=target)
        try:
            ans = generate_answer(q, results, cache=cache)
        except Exception as e:  # noqa: BLE001
            say(f"  FAILED   {q}\n           {e}")
            stats["failed"] += 1
            if isinstance(e, llm.QuotaExhausted):
                say("Daily quota reached - stop here and run this again after the reset.")
                break
            continue
        if ans is None:
            say(f"  SKIPPED  {q} (no API key)")
            continue
        key = "from_cache" if ans["cached"] else "new_requests"
        stats[key] += 1
        say(f"  {'cached ' if ans['cached'] else 'NEW 1 req'}  {q}")
    say(f"\nDone. New API requests used for answers: {stats['new_requests']}, "
        f"already cached: {stats['from_cache']}, failed: {stats['failed']}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Pre-compute everything needed for a demo")
    ap.add_argument("--video", required=True, action="append")
    ap.add_argument("--id", required=True, action="append",
                    help="the exact title you will give the video in the app (one per --video)")
    ap.add_argument("--questions", required=True, help="text file, one question per line")
    ap.add_argument("--no-visual", action="store_true")
    a = ap.parse_args()
    if len(a.video) != len(a.id):
        ap.error("give one --id for every --video")
    cache_dir = os.environ.get("INSIGHTRAG_CACHE_DIR") or os.path.join("library", "cache")
    print(f"Cache folder: {os.path.abspath(cache_dir)}\n")
    warm([(v, sanitize_video_id(i)) for v, i in zip(a.video, a.id)], read_questions(a.questions), cache_dir,
         use_visual=False if a.no_visual else None)


if __name__ == "__main__":
    main()
