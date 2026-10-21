"""
Find which Gemini models your key can use RIGHT NOW (free-tier quota is counted per model).

    python -m src.check_models

Sends one tiny request to each candidate model (each model has its own quota, so this does not
touch the quota of the model you have already used up) and prints the lines to put in .env.
"""
import time

from . import llm

SKIP = ("embedding", "image", "imagen", "veo", "tts", "audio", "live", "native", "robotics",
        "computer", "gemma", "aqa", "learnlm", "pro")


def candidate_models(client) -> list:
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        name = (getattr(m, "name", "") or "").replace("models/", "")
        if "generateContent" in actions and name.startswith("gemini") and not any(s in name for s in SKIP):
            names.append(name)
    # Lite models first (usually the most generous free quota), then Flash
    return sorted(set(names), key=lambda n: ("lite" not in n, "flash" not in n, n))[:8]


def classify(exc: Exception) -> str:
    msg = str(exc)
    if "limit: 0" in msg:
        return "NO FREE QUOTA for this model"
    if "429" in msg:
        return "RATE LIMITED (wait a minute, run again)"
    if "404" in msg or "not found" in msg.lower():
        return "NOT AVAILABLE"
    return f"ERROR: {msg[:60]}"


def main(pause: float = 13.0):
    if not llm.available():
        print("GEMINI_API_KEY is not set. Put it in .env first.")
        return
    try:
        names = candidate_models(llm._client(llm.api_key()))
    except Exception as e:  # noqa: BLE001
        print(f"Could not list models: {e}")
        return
    if not names:
        print("No usable Gemini text models were listed for this key.")
        return

    print(f"Testing {len(names)} models (one tiny request each)...\n")
    working = []
    for i, name in enumerate(names):
        try:
            llm.generate("Reply with the single word OK", model=name, retries=0, min_interval=0)
            print(f"  WORKS                                {name}")
            working.append(name)
        except llm.QuotaExhausted:
            print(f"  DAILY QUOTA ALREADY USED             {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  {classify(e):<37}{name}")
        if i < len(names) - 1:
            time.sleep(pause)  # stay under the 5-requests-per-minute free limit

    print()
    if not working:
        print("No model worked right now. Wait for the daily reset (midnight Pacific time) or enable billing.")
        return
    answer_model = working[0]
    vision_model = working[1] if len(working) > 1 else working[0]
    print("Put these lines in your .env:\n")
    print(f"GEMINI_MODEL={answer_model}")
    print(f"GEMINI_VISION_MODEL={vision_model}")


if __name__ == "__main__":
    main()
