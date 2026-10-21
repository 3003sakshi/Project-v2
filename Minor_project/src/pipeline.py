"""
Steps 1-3 orchestrated: video -> speech chunks (+ visual chunks) with caching.
"""
import os
from typing import Callable, List, Optional, Tuple

from . import llm
from .cache import DiskCache, file_sha256
from .chunking import chunk_segments
from .transcribe import transcribe_video
from .visual import describe_frames, sample_frames, visual_chunks

Progress = Optional[Callable[[float, str], None]]


def process_video(
    video_path: str,
    video_id: str,
    cache_dir: str,
    *,
    whisper_model: str = "base",
    task: str = "translate",
    use_visual: bool = True,
    frame_interval: float = 20.0,
    max_frames: int = 120,
    progress: Progress = None,
) -> Tuple[List[dict], dict]:
    """Return (chunks, info). Never raises for vision failures - falls back to speech-only."""
    cache = DiskCache(cache_dir)
    say = progress or (lambda f, t: None)
    info = {"warnings": [], "speech_chunks": 0, "visual_chunks": 0, "frames_sampled": 0}

    say(0.02, "Hashing video...")
    fhash = file_sha256(video_path)

    say(0.05, "Transcribing speech (Whisper)...")
    segments = transcribe_video(video_path, video_id, model_size=whisper_model, task=task,
                                cache=cache, file_hash=fhash)
    speech = chunk_segments(segments)
    info["speech_chunks"] = len(speech)
    chunks = list(speech)

    if use_visual:
        if not llm.available():
            info["warnings"].append("Visual analysis skipped: GEMINI_API_KEY is not set.")
        else:
            try:
                thumb_dir = os.path.join(cache_dir, "thumbs", fhash[:16])
                say(0.50, "Sampling keyframes...")
                frames, duration = sample_frames(
                    video_path, interval=frame_interval, max_frames=max_frames, thumb_dir=thumb_dir,
                    progress=lambda f, t: say(0.50 + 0.10 * f, t))
                info["frames_sampled"] = len(frames)
                described = describe_frames(
                    frames, fhash, cache=cache,
                    progress=lambda f, t: say(0.60 + 0.35 * f, t))
                vis = visual_chunks(video_id, frames, described, frame_interval, duration)
                info["visual_chunks"] = len(vis)
                chunks.extend(vis)
            except Exception as e:  # noqa: BLE001
                info["warnings"].append(f"Visual analysis failed ({e}); indexed speech only.")

    say(1.0, "Done")
    return chunks, info
