"""Integration tests: real frame extraction on a synthetic video + mocked Gemini (no network)."""
import json

import numpy as np
import pytest

pytest.importorskip("moviepy")

from src import llm, visual
from src.cache import DiskCache
from src.chunking import make_chunk
from src.evaluate import build_report, evaluate_retrieval, load_questions
from src.library import Library
from src.pipeline import process_video
from tests.test_core import FakeEmbedder


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """60 s video: 0-30 s red slide, 30-60 s blue slide (so exactly 2 distinct scenes)."""
    from moviepy.editor import VideoClip

    def frame(t):
        img = np.zeros((90, 160, 3), dtype="uint8")
        img[..., 0 if t < 30 else 2] = 220
        return img

    p = tmp_path_factory.mktemp("vid") / "slides.mp4"
    VideoClip(frame, duration=60).write_videofile(str(p), fps=2, codec="libx264", logger=None, audio=False)
    return str(p)


def test_sample_frames_dedupes_static_scenes(video, tmp_path):
    frames, duration = visual.sample_frames(video, interval=10, thumb_dir=str(tmp_path / "th"))
    assert 59 <= duration <= 61
    assert len(frames) == 2                      # 6 sampled, only the 2 distinct scenes kept
    assert frames[0]["t"] < 30 <= frames[1]["t"]
    assert frames[0]["jpeg"][:2] == b"\xff\xd8"  # real JPEG
    assert (tmp_path / "th").exists()


def test_describe_frames_batches_parses_and_caches(video, tmp_path, monkeypatch):
    calls = []

    def fake_generate(contents, *, model, json_mode=False, **kw):
        n_images = sum(1 for c in contents if not isinstance(c, str))
        calls.append(n_images)
        return "```json\n" + json.dumps([
            {"frame": i + 1, "description": f"Slide {i + 1} about topic", "on_screen_text": f"TITLE {i + 1}"}
            for i in range(n_images)]) + "\n```"

    monkeypatch.setattr(llm, "generate", fake_generate)
    frames, duration = visual.sample_frames(video, interval=10)
    cache = DiskCache(str(tmp_path / "c"))
    d1 = visual.describe_frames(frames, "hash123", cache=cache, batch_size=4)
    assert calls == [2]                          # both frames in ONE request
    d2 = visual.describe_frames(frames, "hash123", cache=cache, batch_size=4)
    assert calls == [2] and d1 == d2             # second run: served from cache, zero API calls

    chunks = visual.visual_chunks("slides", frames, d1, 10, duration)
    assert len(chunks) == 2 and chunks[0]["modality"] == "visual"
    assert "On-screen text: TITLE 1" in chunks[0]["text"]
    assert chunks[0]["end"] - chunks[0]["start"] == 10


def test_parse_json_array_is_robust():
    assert visual._parse_json_array('Sure! [{"frame": 1, "description": "x"}] done')[0]["frame"] == 1
    assert visual._parse_json_array("not json") == []
    assert visual._parse_json_array('{"frame": 1}') == [{"frame": 1}]


def test_pipeline_degrades_to_speech_only_without_api_key(video, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.pipeline.transcribe_video", lambda *a, **k: [])
    chunks, info = process_video(video, "slides", str(tmp_path), use_visual=True)
    assert chunks == [] and any("GEMINI_API_KEY" in w for w in info["warnings"])


def test_library_mcp_tools_and_eval(tmp_path, monkeypatch):
    from src import mcp_server

    lib = Library(str(tmp_path / "lib"), embedder=FakeEmbedder())
    idx = lib.load_index()
    idx.add_chunks([
        make_chunk("lec", "speech", 0, 0, 40, "gradient descent updates weights using the learning rate"),
        make_chunk("lec", "speech", 1, 40, 60, "overfitting happens when the model memorises training data"),
        make_chunk("lec", "visual", 0, 60, 80, "Slide with a confusion matrix. On-screen text: Precision Recall F1"),
    ])
    idx.save(str(lib.index_dir))
    monkeypatch.setattr(mcp_server, "_lib", lib)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert mcp_server.list_videos()["videos"]["lec"] == {"speech": 2, "visual": 1}
    r = mcp_server.search_video("confusion matrix", top_k=99)           # top_k clamped
    assert r["results"][0]["modality"] == "visual" and r["results"][0]["timestamp"] == "01:00"
    assert len(mcp_server.search_video("data", modality="speech")["results"]) <= 2
    with pytest.raises(ValueError):
        mcp_server.search_video("x", video_id="nope")
    with pytest.raises(ValueError):
        mcp_server.search_video("   ")
    a = mcp_server.ask_video("what is gradient descent")
    assert a["evidence"] and "GEMINI_API_KEY" in a["note"]
    w = mcp_server.get_transcript_window("lec", "00:50", "01:10")
    assert {s["modality"] for s in w["segments"]} == {"speech", "visual"}
    for bad in ("../../etc/passwd", "/etc/passwd"):
        with pytest.raises(ValueError):
            mcp_server.index_video(bad, "x")
    with pytest.raises(ValueError):
        mcp_server.index_video("ok.mp4", "bad/id")

    qf = tmp_path / "q.json"
    qf.write_text(json.dumps([
        {"id": "s1", "type": "speech", "video_id": "lec", "question": "gradient descent learning rate",
         "gold": [{"start": "00:00", "end": "00:40"}]},
        {"id": "v1", "type": "visual", "video_id": "lec", "question": "confusion matrix precision recall",
         "gold": [{"start": "01:00", "end": "01:20"}]},
    ]))
    qs = load_questions(str(qf))
    rows = evaluate_retrieval(idx, qs)
    by = {(r["id"], r["config"]): r for r in rows}
    assert by[("s1", "hybrid | speech")]["hit@1"] == 1.0
    assert by[("v1", "hybrid | speech")]["hit@5"] == 0.0          # speech-only cannot localise the slide
    assert by[("v1", "hybrid | speech+visual")]["hit@1"] == 1.0
    strict = {(r["id"], r["config"]): r for r in evaluate_retrieval(idx, qs, strict=True)}
    assert strict[("s1", "hybrid | speech+visual")]["hit@1"] == 1.0  # speech question, speech chunk first
    assert strict[("v1", "hybrid | speech")]["hit@5"] == 0.0
    report = build_report(rows, None, len(qs))
    assert "hybrid | speech+visual" in report and "Paired comparisons" in report


def test_daily_quota_fails_fast_without_retries(monkeypatch):
    calls = []

    class _Models:
        def generate_content(self, **kw):
            calls.append(1)
            raise Exception("429 RESOURCE_EXHAUSTED ... GenerateRequestsPerDayPerProjectPerModel-FreeTier ...")

    class _Client:
        models = _Models()

    monkeypatch.setattr(llm, "_client", lambda key: _Client())
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_MIN_INTERVAL", "0")
    with pytest.raises(llm.QuotaExhausted):
        llm.generate("hi", model="m")
    assert len(calls) == 1          # no pointless retries against a 22-hour limit


def test_default_frame_batch_is_8(monkeypatch, tmp_path):
    n_calls = []

    def fake_generate(contents, *, model, json_mode=False, **kw):
        n = sum(1 for c in contents if not isinstance(c, str))
        n_calls.append(n)
        return json.dumps([{"frame": i + 1, "description": "d", "on_screen_text": ""} for i in range(n)])

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.delenv("GEMINI_FRAME_BATCH", raising=False)
    frames = [{"t": float(i), "jpeg": b"\xff\xd8x", "thumb": None} for i in range(12)]
    visual.describe_frames(frames, "h", cache=DiskCache(str(tmp_path)))
    assert n_calls == [8, 4]


def test_check_models_picks_working_models(monkeypatch, capsys):
    from src import check_models

    class _M:
        def __init__(self, name, actions=("generateContent",)):
            self.name, self.supported_actions = name, list(actions)

    class _Client:
        class models:
            @staticmethod
            def list():
                return [_M("models/gemini-3.6-flash"), _M("models/gemini-3.5-flash-lite"),
                        _M("models/gemini-embedding-001"), _M("models/gemini-3.5-pro"),
                        _M("models/gemini-3.5-flash", actions=("embedContent",))]

    def fake_generate(prompt, *, model, **kw):
        if model == "gemini-3.6-flash":
            raise llm.QuotaExhausted("used up")
        return "OK"

    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(llm, "_client", lambda key: _Client())
    monkeypatch.setattr(llm, "generate", fake_generate)
    check_models.main(pause=0)
    out = capsys.readouterr().out
    assert "WORKS" in out and "gemini-3.5-flash-lite" in out
    assert "DAILY QUOTA ALREADY USED" in out
    assert "embedding" not in out and "pro" not in out.split("Put these")[0].replace("provider", "")
    assert "GEMINI_MODEL=gemini-3.5-flash-lite" in out


def test_warmup_populates_cache_so_the_demo_needs_no_requests(monkeypatch, tmp_path):
    import re
    from src import warmup
    import src.retrieve as R

    monkeypatch.setattr(R, "get_embedder", lambda: FakeEmbedder())
    chunks = [make_chunk("Demo", "speech", 0, 0, 38, "frameworks like opencv and tensorflow",
                         segments=[{"start": 0, "end": 38, "text": "frameworks like opencv and tensorflow"}])]
    monkeypatch.setattr(warmup, "process_video", lambda *a, **k: (chunks, {
        "speech_chunks": 1, "visual_chunks": 0, "warnings": [], "frames_sampled": 0}))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    calls = []
    monkeypatch.setattr(llm, "generate", lambda prompt, *, model, **kw: calls.append(1) or "Answer [Demo @ 00:10].")

    cache_dir = str(tmp_path / "cache")
    first = warmup.warm([("v.mp4", "Demo")], ["Which frameworks?"], cache_dir, say=lambda *_: None)
    second = warmup.warm([("v.mp4", "Demo")], ["Which frameworks?"], cache_dir, say=lambda *_: None)
    assert first["new_requests"] == 1 and second["new_requests"] == 0 and second["from_cache"] == 1
    assert len(calls) == 1          # the second run (= demo day) made no API call


def test_app_and_warmup_share_settings():
    from src import config
    app = open("app.py", encoding="utf-8").read()
    wu = open("src/warmup.py", encoding="utf-8").read()
    for text in (app, wu):
        assert "top_k_for(" in text and "FRAME_INTERVAL" in text and "match_video_title(" in text
    assert config.top_k_for(1) == 6 and config.top_k_for(3) == 10 and config.top_k_for(50) == 12


def test_match_video_title_routes_questions():
    from src.retrieve import match_video_title as m
    titles = ["Lecture 1", "Lecture 10", "Intro to cv"]
    assert m("In Lecture 1, what is overfitting?", titles) == "Lecture 1"
    assert m("what did lecture 10 say about CNNs", titles) == "Lecture 10"      # longest / whole word
    assert m("compare Lecture 1 and Intro to CV", titles) is None                # two videos -> search all
    assert m("what is a CNN?", titles) is None


def test_abstention_detection_and_cleaning(monkeypatch):
    from src.answer import generate_answer, is_abstention, validate_citations
    res = [make_chunk("v", "speech", 0, 0, 38, "frameworks like opencv", segments=[{"start": 0, "end": 38, "text": "x"}])]
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    def run(reply):
        monkeypatch.setattr(llm, "generate", lambda prompt, *, model, **kw: reply)
        return generate_answer("q", res)

    a = run("NOT_FOUND: The video does not discuss programming languages.")
    assert a["abstained"] and not a["text"].startswith("NOT_FOUND") and "programming" in a["text"]
    assert run("The provided sources do not contain information about that.")["abstained"]       # fallback, no marker
    assert run("I cannot find anything about this in the sources.")["abstained"]
    real = run("Frameworks like OpenCV are mentioned [v @ 00:10].")
    assert not real["abstained"] and real["validation"]["n_valid"] == 1
    # a real answer that merely contains the word 'no' must not be mistaken for an abstention
    assert not run("There is no doubt the video covers OpenCV [v @ 00:10].")["abstained"]
