"""
Thin Gemini wrapper shared by the vision step and the answer step.

Free-tier friendly: a global minimum interval between calls (default 4 s ~= 15 requests/min)
and exponential backoff on 429/503. Tune with GEMINI_MIN_INTERVAL.
"""
import os
import threading
import time
from functools import lru_cache

_lock = threading.Lock()
_last_call = 0.0

class QuotaExhausted(RuntimeError):
    """The free-tier DAILY request quota for a model is used up - retrying will not help."""


_TRANSIENT = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded", "deadline", "timed out")


def api_key():
    return os.environ.get("GEMINI_API_KEY")


def available() -> bool:
    return bool(api_key())


@lru_cache(maxsize=2)
def _client(key: str):
    from google import genai

    return genai.Client(api_key=key)


def generate(contents, *, model: str, json_mode: bool = False, temperature: float = 0.0,
             retries: int = 4, min_interval: float = None) -> str:
    """Call Gemini and return response text. Raises on non-transient errors."""
    global _last_call
    from google.genai import types

    cfg_kwargs = {"temperature": temperature}
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"
    cfg = types.GenerateContentConfig(**cfg_kwargs)

    interval = float(os.environ.get("GEMINI_MIN_INTERVAL", 4.0)) if min_interval is None else min_interval
    delay = 5.0
    for attempt in range(retries + 1):
        with _lock:
            wait = interval - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        try:
            resp = _client(api_key()).models.generate_content(model=model, contents=contents, config=cfg)
            return resp.text or ""
        except Exception as e:  # noqa: BLE001 - SDK raises several exception types
            msg = str(e)
            if "429" in msg and ("PerDay" in msg or "per day" in msg.lower()):
                raise QuotaExhausted(
                    f"Gemini daily free quota for '{model}' is used up (it resets at midnight Pacific time). "
                    "Options: wait, set GEMINI_MODEL / GEMINI_VISION_MODEL to another model "
                    "(each model has its own daily quota), or enable billing."
                ) from e
            transient = any(s in str(e) for s in _TRANSIENT)
            if attempt == retries or not transient:
                raise
            time.sleep(delay)
            delay *= 2
    return ""
