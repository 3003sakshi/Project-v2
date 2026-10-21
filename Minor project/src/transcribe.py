"""
Step 1a - speech: video/audio -> timestamped transcript segments (faster-whisper, CPU, int8).
"""
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import List, Optional

from .cache import DiskCache


@dataclass
class TranscriptSegment:
    video_id: str
    start: float
    end: float
    text: str


@lru_cache(maxsize=2)
def _load_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _run_whisper(path: str, model_size: str, task: str, beam_size: int):
    model = _load_model(model_size)
    segs, _info = model.transcribe(path, task=task, beam_size=beam_size, vad_filter=True)
    return [(s.start, s.end, s.text.strip()) for s in segs if s.text.strip()]


def transcribe_video(
    video_path: str,
    video_id: str,
    model_size: str = "base",
    task: str = "translate",
    beam_size: int = 1,
    cache: Optional[DiskCache] = None,
    file_hash: Optional[str] = None,
    save_json_path: Optional[str] = None,
) -> List[TranscriptSegment]:
    cache_key = f"{file_hash}:{model_size}:{task}:{beam_size}" if (cache and file_hash) else None
    raw = cache.get("transcripts", cache_key) if cache_key else None

    if raw is None:
        try:
            # faster-whisper decodes the container directly (via PyAV) - fastest path.
            raw = _run_whisper(video_path, model_size, task, beam_size)
        except Exception:
            # Fallback: extract a 16 kHz mono WAV with moviepy, then transcribe that.
            from moviepy.editor import AudioFileClip

            with tempfile.TemporaryDirectory() as td:
                wav = os.path.join(td, "audio.wav")
                clip = AudioFileClip(video_path)
                clip.write_audiofile(wav, fps=16000, nbytes=2, codec="pcm_s16le", logger=None)
                clip.close()
                raw = _run_whisper(wav, model_size, task, beam_size)
        if cache_key:
            cache.put("transcripts", cache_key, raw)

    segments = [TranscriptSegment(video_id, float(s), float(e), t) for s, e, t in raw]
    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)
    return segments


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Transcribe a video to timestamped segments")
    ap.add_argument("video")
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="base")
    a = ap.parse_args()
    segs = transcribe_video(a.video, a.video_id, model_size=a.model, save_json_path=a.out)
    print(f"{len(segs)} segments")
