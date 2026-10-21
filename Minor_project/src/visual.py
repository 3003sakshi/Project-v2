"""
Step 1c - vision: sample keyframes, describe them with a vision-language model (Gemini),
and turn the result into timestamped *visual chunks* that are indexed next to speech chunks.

Designed for a CPU-only laptop + Gemini free tier:
  * frames are sampled at a fixed interval, then near-duplicate frames (a slide that stays
    on screen) are dropped with a cheap 32x18 grayscale difference -> far fewer API calls
  * several frames (default 8) are sent per request: free-tier DAILY request caps are small
  * every frame description is cached by (video hash, time, model) so nothing is paid twice
  * a hard cap (max_frames) bounds the worst case for very long videos
"""
import io
import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import llm
from .cache import DiskCache
from .chunking import make_chunk
from .config import GEMINI_VISION_MODEL

PROMPT_VERSION = "v1"

FRAME_PROMPT = (
    "You are analysing keyframes sampled from a video (lecture, meeting, or talk). "
    "For EACH image, in order, produce one JSON object. Return ONLY a JSON array of objects with keys:\n"
    '  "frame": the 1-based index of the image,\n'
    '  "description": 1-2 factual sentences on what is visible (slide topic, diagram, chart, '
    "code, objects, what the person is doing),\n"
    '  "on_screen_text": all legible on-screen text, verbatim (slide titles, bullets, code, formulas, '
    "labels); empty string if there is none.\n"
    "Do not guess at unreadable text and do not invent details."
)


def sample_frames(
    video_path: str,
    interval: float = 20.0,
    max_frames: int = 120,
    diff_threshold: float = 6.0,
    thumb_dir: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[dict], float]:
    """Return ([{t, jpeg, thumb}], duration). Near-duplicate consecutive frames are skipped."""
    from moviepy.editor import VideoFileClip
    from PIL import Image

    clip = VideoFileClip(video_path)
    try:
        duration = float(clip.duration or 0.0)
        times = [i * interval + min(1.0, interval / 2) for i in range(int(duration // interval) + 1)]
        times = [t for t in times if t < duration]
        if len(times) > max_frames:  # uniform subsample
            idx = np.linspace(0, len(times) - 1, max_frames).astype(int)
            times = [times[i] for i in idx]

        kept, last_sig = [], None
        for n, t in enumerate(times):
            if progress:
                progress(n / max(len(times), 1), f"Sampling frame {n + 1}/{len(times)}")
            img = Image.fromarray(clip.get_frame(t))
            sig = np.asarray(img.convert("L").resize((32, 18)), dtype=np.float32)
            if last_sig is not None and float(np.abs(sig - last_sig).mean()) < diff_threshold:
                continue  # same slide / static scene
            last_sig = sig

            img.thumbnail((768, 768))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=70)
            thumb_path = None
            if thumb_dir:
                os.makedirs(thumb_dir, exist_ok=True)
                thumb_path = os.path.join(thumb_dir, f"{int(t):06d}.jpg")
                small = img.copy()
                small.thumbnail((320, 320))
                small.convert("RGB").save(thumb_path, format="JPEG", quality=70)
            kept.append({"t": float(t), "jpeg": buf.getvalue(), "thumb": thumb_path})
        return kept, duration
    finally:
        clip.close()


def _parse_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else []


def describe_frames(
    frames: List[dict],
    file_hash: str,
    cache: Optional[DiskCache] = None,
    batch_size: Optional[int] = None,
    model: str = GEMINI_VISION_MODEL,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[float, dict]:
    """Return {t: {description, on_screen_text}} for every frame (cached where possible)."""
    from google.genai import types

    batch_size = batch_size or int(os.environ.get("GEMINI_FRAME_BATCH", 8))
    out: Dict[float, dict] = {}
    todo = []
    for f in frames:
        key = f"{file_hash}:{f['t']:.1f}:{model}:{PROMPT_VERSION}"
        hit = cache.get("frames", key) if cache else None
        if hit is not None:
            out[f["t"]] = hit
        else:
            todo.append((key, f))

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    for bi, batch in enumerate(batches):
        if progress:
            progress(bi / max(len(batches), 1), f"Vision model: batch {bi + 1}/{len(batches)}")
        contents = [FRAME_PROMPT]
        for n, (_, f) in enumerate(batch, 1):
            contents.append(f"Image {n}:")
            contents.append(types.Part.from_bytes(data=f["jpeg"], mime_type="image/jpeg"))
        parsed = _parse_json_array(llm.generate(contents, model=model, json_mode=True))
        by_idx = {int(p.get("frame", i + 1)): p for i, p in enumerate(parsed) if isinstance(p, dict)}
        for n, (key, f) in enumerate(batch, 1):
            p = by_idx.get(n, {})
            rec = {
                "description": str(p.get("description", "") or "").strip(),
                "on_screen_text": str(p.get("on_screen_text", "") or "").strip(),
            }
            out[f["t"]] = rec
            if cache and p:  # don't cache failures
                cache.put("frames", key, rec)
    return out


def visual_chunks(video_id: str, frames: List[dict], described: Dict[float, dict],
                  interval: float, duration: float) -> List[dict]:
    chunks = []
    for f in frames:
        rec = described.get(f["t"]) or {}
        desc, ost = rec.get("description", ""), rec.get("on_screen_text", "")
        if not desc and not ost:
            continue
        text = desc if not ost else f"{desc} On-screen text: {ost}"
        end = min(f["t"] + interval, duration) if duration else f["t"] + interval
        chunks.append(make_chunk(video_id, "visual", len(chunks), f["t"], end, text,
                                 description=desc, on_screen_text=ost, thumb=f.get("thumb")))
    return chunks
