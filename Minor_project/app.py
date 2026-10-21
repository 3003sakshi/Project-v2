"""
InsightRAG - ask questions about what was SAID and what was SHOWN in a video.

Every answer cites timestamps; the app verifies each citation against the evidence it retrieved
and lets you jump straight to those moments in the video.

Each visitor's videos live only in memory / a private temp folder (nothing is shared between users).

Run with:   python -m streamlit run app.py
"""
import html
import os
import re
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="InsightRAG", page_icon="🎥", layout="wide", initial_sidebar_state="expanded")

from src import llm  # noqa: E402
from src.answer import CITATION_RE, generate_answer  # noqa: E402
from src.cache import DiskCache  # noqa: E402
from src.config import FRAME_INTERVAL, top_k_for  # noqa: E402
from src.chunking import format_timestamp  # noqa: E402
from src.index import HybridIndex  # noqa: E402
from src.library import sanitize_video_id  # noqa: E402
from src.pipeline import process_video  # noqa: E402
from src.retrieve import match_video_title  # noqa: E402

SHOW_TIMELINE = False     # True = also draw the spoken/on-screen match timeline under each answer
SUGGESTIONS = [
    "Summarize this video",
    "What is shown on the slides or screen?",
    "What are the key points?",
]

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
    --ink:#0A0D12; --panel:#12161F; --panel-2:#171C27; --line:#262D3A;
    --timecode:#F2A93B; --signal:#35D0BA; --text:#EDEFF4; --muted:#8A93A6; --bad:#F07A7A;
}
.stApp { background: radial-gradient(ellipse 120% 80% at 50% -10%, #161B26 0%, var(--ink) 55%); color: var(--text); }
html, body, [class*="css"], .stMarkdown p, label { font-family:'Inter',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-0.01em; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display:none !important; }
.block-container { padding-top:2.2rem; max-width:1000px; }
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }

.vr-hero h1 { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.2rem; margin:0;
    background:linear-gradient(90deg,var(--text) 35%,var(--timecode) 130%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.vr-tagline { color:var(--muted); font-size:1rem; margin:4px 0 22px 0; }

.vr-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:8px 0 18px 0; }
.vr-step { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; }
.vr-step-n { width:28px; height:28px; border-radius:50%; background:rgba(242,169,59,.15); color:var(--timecode);
    font-family:'JetBrains Mono',monospace; font-weight:600; display:flex; align-items:center; justify-content:center; margin-bottom:10px; }
.vr-step-t { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; margin-bottom:4px; }
.vr-step-d { color:var(--muted); font-size:.9rem; line-height:1.5; }

.vr-answer-card { background:linear-gradient(155deg,var(--panel-2) 0%,var(--panel) 100%); border:1px solid var(--line);
    border-left:3px solid var(--timecode); border-radius:10px; padding:18px 20px; }
.vr-eyebrow { font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:var(--timecode); margin-bottom:8px; }
.vr-answer-text { font-size:1rem; line-height:1.65; color:var(--text); }
.vr-verify { margin-top:12px; padding-top:10px; border-top:1px solid var(--line); font-family:'JetBrains Mono',monospace; font-size:.74rem; color:var(--muted); }

.vr-cite { font-family:'JetBrains Mono',monospace; font-size:.76rem; border-radius:5px; padding:1px 7px; margin:0 2px; white-space:nowrap; }
.vr-valid { background:rgba(53,208,186,.12); color:var(--signal); border:1px solid rgba(53,208,186,.35); }
.vr-invalid { background:rgba(240,90,90,.12); color:var(--bad); border:1px solid rgba(240,90,90,.4); }

.vr-timeline-wrap { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px 6px 16px; margin:12px 0 6px 0; }
.vr-timeline-label { font-family:'JetBrains Mono',monospace; font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }

.vr-source-card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:8px; }
.vr-source-head { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.vr-badge-time { font-family:'JetBrains Mono',monospace; background:rgba(242,169,59,.12); color:var(--timecode);
    border:1px solid rgba(242,169,59,.35); border-radius:5px; padding:2px 8px; font-size:.78rem; font-weight:600; }
.vr-badge { font-family:'JetBrains Mono',monospace; font-size:.68rem; border-radius:5px; padding:2px 7px; font-weight:600; }
.vr-badge-speech { background:rgba(242,169,59,.12); color:var(--timecode); border:1px solid rgba(242,169,59,.35); }
.vr-badge-visual { background:rgba(53,208,186,.12); color:var(--signal); border:1px solid rgba(53,208,186,.35); }
.vr-badge-video { font-family:'JetBrains Mono',monospace; color:var(--muted); font-size:.75rem; }
.vr-meter { flex:1; height:4px; background:var(--line); border-radius:2px; overflow:hidden; max-width:90px; }
.vr-meter > span { display:block; height:100%; background:var(--timecode); }
.vr-source-text { color:var(--text); font-size:.88rem; line-height:1.55; }

.vr-chip { display:inline-flex; align-items:center; font-family:'JetBrains Mono',monospace; font-size:.78rem;
    background:rgba(53,208,186,.1); color:var(--signal); border:1px solid rgba(53,208,186,.3); border-radius:999px; padding:4px 12px; }
.vr-mini { font-family:'JetBrains Mono',monospace; font-size:.68rem; color:var(--muted); margin:3px 0 8px 6px; }

div[data-testid="stButton"] button[kind="primary"] { background:var(--timecode); border:none; color:#1A1300; font-weight:600; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#FFC061; }
div[data-testid="stButton"] button[kind="secondary"] { border-radius:999px; border:1px solid var(--line); background:var(--panel); }
div[data-testid="stButton"] button[kind="secondary"]:hover { border-color:var(--timecode); color:var(--timecode); }
</style>
"""


# ------------------------------------------------------------------------------ session
def init_session():
    ss = st.session_state
    if "session_id" not in ss:
        ss.session_id = uuid.uuid4().hex[:8]
    if "index" not in ss:
        ss.index = HybridIndex()  # in memory, private to this browser session
    if "data_dir" not in ss:
        ss.data_dir = tempfile.mkdtemp(prefix=f"insightrag_{ss.session_id}_")
    for key, default in (("video_paths", {}), ("messages", []), ("play", None), ("processing", False),
                         ("input_key_counter", 0), ("confirm_delete", None), ("notice", None)):
        if key not in ss:
            ss[key] = default


def cache_dir() -> str:
    return os.environ.get("INSIGHTRAG_CACHE_DIR") or os.path.join(st.session_state.data_dir, "cache")


def delete_video(video_id: str):
    ss = st.session_state
    ss.index.remove_video(video_id)
    path = ss.video_paths.pop(video_id, None)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    ss.messages, ss.play = [], None


# ------------------------------------------------------------------------------ rendering helpers
# Everything user- or model-controlled is html-escaped before it reaches st.html.
def render_answer_html(text: str, validation: dict) -> str:
    cites = iter(validation["citations"])

    def chip(m):
        c = next(cites, None)
        ok = bool(c and c["valid"])
        tip = "Verified against the retrieved evidence" if ok else "Not found in any retrieved moment"
        return f'<span class="vr-cite {"vr-valid" if ok else "vr-invalid"}" title="{tip}">{"✓" if ok else "✕"} {m.group(0)[1:-1]}</span>'

    safe = CITATION_RE.sub(chip, html.escape(text))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return safe.replace("\n", "<br>")


def render_timeline(video_id: str, all_chunks: list, matches: list) -> str:
    chunks = [c for c in all_chunks if c["video_id"] == video_id]
    if not chunks:
        return ""
    duration = max(c["end"] for c in chunks) or 1.0
    width, height, x0, x1 = 1000, 108, 24, 976
    lanes = {"speech": (30, "#F2A93B"), "visual": (62, "#35D0BA")}

    def x_of(t):
        return x0 + (t / duration) * (x1 - x0)

    parts = []
    for m, (y, _) in lanes.items():
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#262D3A" stroke-width="2"/>')
        parts.append(f'<text x="{x0}" y="{y - 12}" fill="#8A93A6" font-family="JetBrains Mono, monospace" font-size="11">{m}</text>')
        for c in chunks:
            if c["modality"] == m:
                parts.append(f'<line x1="{x_of(c["start"]):.1f}" y1="{y - 6}" x2="{x_of(c["start"]):.1f}" y2="{y + 6}" stroke="#262D3A" stroke-width="2"/>')
    for r in matches:
        y, colour = lanes[r["modality"]]
        x, norm = x_of(r["start"]), 0.45 + 0.55 * r["rel"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="{5 + norm * 7:.1f}" fill="{colour}" fill-opacity="{norm:.2f}" '
                     f'stroke="{colour}" stroke-width="1.4"><title>{r["modality"]} · {r["timestamp"]}</title></circle>')
        parts.append(f'<text x="{x:.1f}" y="98" fill="#8A93A6" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="middle">{r["timestamp"]}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:auto;display:block;">{"".join(parts)}</svg>')


def best_lines(m: dict, n: int = 3):
    """Without a written answer: the individual transcript lines / frames that best match the question."""
    from src.bm25 import tokenize

    q = set(tokenize(m["query"]))
    cands = []
    for rank, r in enumerate(m["results"]):
        parts = ([(s["start"], s["text"]) for s in r["segments"]]
                 if r["modality"] == "speech" and r.get("segments") else [(r["start"], r["text"])])
        for start, text in parts:
            cands.append((len(q & set(tokenize(text))), -rank, start, r["video_id"], text, r["modality"]))
    cands.sort(key=lambda c: (-c[0], -c[1], c[2]))
    picked, seen = [], set()
    for overlap, _, start, vid, text, kind in cands:
        if (vid, int(start)) in seen:
            continue
        if overlap == 0 and picked:
            break
        seen.add((vid, int(start)))
        picked.append((vid, start, text, kind))
        if len(picked) >= n:
            break
    return picked


def jump_targets(m: dict):
    """Distinct (video_id, seconds) moments to offer as 'play' buttons: valid citations, else top matches."""
    ss = st.session_state
    out, seen = [], set()
    if m["answer"]:
        for c in m["answer"]["validation"]["citations"]:
            if c["valid"] and c["chunk_id"]:
                vid = c["chunk_id"].rsplit(":", 2)[0]
                key = (vid, int(c["seconds"]))
                if key not in seen and vid in ss.video_paths:
                    seen.add(key)
                    out.append((vid, float(c["seconds"])))
    if not out:
        for vid, start, _, _ in best_lines(m):
            if vid in ss.video_paths:
                out.append((vid, start))
    return sorted(out, key=lambda t: (t[0], t[1]))[:8]


def render_assistant(i: int, m: dict):
    ss = st.session_state
    results, answer = m["results"], m["answer"]
    if not results:
        st.info("I couldn't find anything relevant in your videos. Try rephrasing the question.")
        return
    if m.get("note"):
        st.warning(m["note"])
    if m.get("scope") and len(ss.index.videos()) > 1:
        st.caption(f"Searched only in “{m['scope']}”")

    abstained = bool(answer and answer.get("abstained"))
    if answer:
        v = answer["validation"]
        body, eyebrow = render_answer_html(answer["text"], v), "Answer"
        if abstained:
            verdict = "Not found in your videos"
        elif v["n_citations"]:
            bad = v["n_citations"] - v["n_valid"]
            verdict = f"✓ {v['n_valid']}/{v['n_citations']} timestamps verified against the video evidence"
            verdict += f" · ✕ {bad} could not be verified" if bad else ""
        else:
            verdict = "No timestamps cited in this answer"
        footer = f'<div class="vr-verify">{html.escape(verdict)}</div>'
    else:
        eyebrow = ("Closest moments" if llm.available()
                   else "Closest moments · add a Gemini key for written answers")
        body = "<br><br>".join(
            f"<b>[{format_timestamp(start)}] {html.escape(vid)} · {'said' if kind == 'speech' else 'shown'}</b> — "
            + html.escape(text if len(text) < 260 else text[:257] + "…").replace("\n", "<br>")
            for vid, start, text, kind in best_lines(m))
        footer = ""
    st.html(f'<div class="vr-answer-card"><div class="vr-eyebrow">{eyebrow}</div>'
            f'<div class="vr-answer-text">{body}</div>{footer}</div>')

    if abstained:      # nothing relevant exists, so do not present any "evidence"
        with st.expander("Show what was searched (nothing relevant was found)"):
            for r in results:
                st.caption(f"{r['timestamp']} · {r['video_id']} · {r['modality']} — "
                           + (r["text"][:140] + "…" if len(r["text"]) > 140 else r["text"]))
        return

    jumps = jump_targets(m)
    if jumps:
        st.caption("Jump to a moment in the video")
        cols = st.columns(min(len(jumps), 6))
        for k, (vid, sec) in enumerate(jumps):
            label = f"▶ {format_timestamp(sec)}" if len({j[0] for j in jumps}) == 1 else f"▶ {vid} {format_timestamp(sec)}"
            if cols[k % len(cols)].button(label, key=f"jump_{i}_{k}"):
                ss.play = (i, vid, sec)
                st.rerun()
    if ss.play and ss.play[0] == i and ss.play[1] in ss.video_paths:
        st.video(ss.video_paths[ss.play[1]], start_time=int(ss.play[2]))
        if st.button("✕ Close player", key=f"close_{i}"):
            ss.play = None
            st.rerun()

    for vid in (sorted({r["video_id"] for r in results}) if SHOW_TIMELINE else []):
        svg = render_timeline(vid, ss.index.chunks, [r for r in results if r["video_id"] == vid])
        if svg:
            st.html(f'<div class="vr-timeline-wrap"><div class="vr-timeline-label">Where in “{html.escape(vid)}” '
                    f'· amber = spoken · teal = on screen</div>{svg}</div>')

    with st.expander("See the evidence behind this answer"):
        for j, r in enumerate(results):
            kind = r["modality"]
            left, right = st.columns([6, 1])
            with left:
                st.html(f"""<div class="vr-source-card"><div class="vr-source-head">
                    <span class="vr-badge-time">{r['timestamp']}</span>
                    <span class="vr-badge vr-badge-{kind}">{'🗣 SAID' if kind == 'speech' else '🖼 SHOWN'}</span>
                    <span class="vr-badge-video">{html.escape(r['video_id'])}</span>
                    <span class="vr-meter"><span style="width:{max(0, min(100, r['rel'] * 100)):.0f}%"></span></span>
                    </div><div class="vr-source-text">{html.escape(r['text']).replace(chr(10), '<br>')}</div></div>""")
                thumb = r.get("thumb")
                if kind == "visual" and thumb and os.path.exists(thumb):
                    st.image(thumb, width=240)
            with right:
                if r["video_id"] in ss.video_paths and st.button(f"▶ {r['timestamp']}", key=f"src_{i}_{j}"):
                    ss.play = (i, r["video_id"], r["start"])
                    st.rerun()


# ------------------------------------------------------------------------------ app
init_session()
st.html(THEME)
ss = st.session_state
has_key = llm.available()

# ============================== SIDEBAR
with st.sidebar:
    if ss.notice:
        st.toast(ss.notice, icon="✅")
        ss.notice = None

    st.markdown("### 📥 Add a video")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mkv", "mov", "avi"], label_visibility="collapsed")
    title_input = st.text_input("Title (optional)", key=f"title_{ss.input_key_counter}", placeholder="e.g. Lecture 1")
    if uploaded_file:
        video_id = sanitize_video_id(title_input if title_input.strip() else os.path.splitext(uploaded_file.name)[0])
    else:
        video_id = ""

    use_visual = st.checkbox("Also analyze slides & visuals", value=has_key, disabled=not has_key,
                             help="Lets you ask about diagrams, slides and on-screen text. Takes a bit longer.")

    already = video_id in ss.index.videos()
    replace = st.checkbox("Replace the existing version", value=False) if already else False
    if already and not replace:
        st.caption(f"“{video_id}” is already added.")

    go = st.button("🚀 Process video", type="primary", use_container_width=True,
                   disabled=ss.processing or not uploaded_file or (already and not replace))

    if go:
        ss.processing = True
        info = None
        try:
            os.makedirs(ss.data_dir, exist_ok=True)
            path = os.path.join(ss.data_dir, f"{video_id}{os.path.splitext(uploaded_file.name)[1]}")
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            with st.status("Processing your video…", expanded=True) as status:
                bar = st.progress(0.0, text="Starting…")
                if replace:
                    ss.index.remove_video(video_id)
                chunks, info = process_video(
                    path, video_id, cache_dir(), use_visual=use_visual, frame_interval=FRAME_INTERVAL,
                    progress=lambda fr, tx: bar.progress(min(max(fr * 0.9, 0.0), 0.9), text=tx))
                bar.progress(0.95, text="Building the search index…")
                ss.index.add_chunks(chunks)
                ss.video_paths[video_id] = path
                ss.messages, ss.play = [], None
                bar.progress(1.0, text="Done")
                status.update(label="Ready ✅", state="complete", expanded=False)
            ss.notice = f"“{video_id}” is ready to search"
            ss.input_key_counter += 1
        except Exception as e:  # noqa: BLE001
            st.error(f"Processing failed: {e}")
        finally:
            ss.processing = False
        if info is not None:
            if info["warnings"]:
                for w in info["warnings"]:
                    st.warning(w)
            else:
                st.rerun()

    st.divider()
    videos, stats = ss.index.videos(), ss.index.stats()
    st.markdown(f"### 📚 Your videos ({len(videos)})")
    if videos:
        for v in videos:
            a, b = st.columns([5, 1])
            with a:
                s = stats[v]
                st.html(f'<span class="vr-chip">▸ {html.escape(v)}</span>'
                        f'<div class="vr-mini">{s.get("speech", 0)} spoken · {s.get("visual", 0)} visual moments</div>')
            with b:
                if st.button("🗑", key=f"del_{v}", help=f"Remove “{v}”"):
                    ss.confirm_delete = v
                    st.rerun()
        if ss.confirm_delete:
            st.warning(f"Remove **{ss.confirm_delete}**?")
            c1, c2 = st.columns(2)
            if c1.button("Yes, remove", type="primary", key="del_yes", use_container_width=True):
                delete_video(ss.confirm_delete)
                ss.confirm_delete = None
                st.rerun()
            if c2.button("Cancel", key="del_no", use_container_width=True):
                ss.confirm_delete = None
                st.rerun()
        if ss.messages and st.button("🧹 Clear conversation", use_container_width=True):
            ss.messages, ss.play = [], None
            st.rerun()
    else:
        st.caption("Nothing here yet.")

    if not has_key:
        st.divider()
        st.caption("⚠️ No Gemini key found. Slide analysis and written answers are off; "
                   "you'll still get the closest matching moments.")

# ============================== MAIN
st.html('<div class="vr-hero"><h1>🎥 InsightRAG</h1></div>'
        '<div class="vr-tagline">Ask about what was <b>said</b> and what was <b>shown</b> in your videos. '
        'Every answer points to the exact moment.</div>')

videos = ss.index.videos()
if not videos:
    st.html("""<div class="vr-steps">
        <div class="vr-step"><div class="vr-step-n">1</div><div class="vr-step-t">Upload</div>
            <div class="vr-step-d">Add a lecture, meeting or talk from the sidebar.</div></div>
        <div class="vr-step"><div class="vr-step-n">2</div><div class="vr-step-t">Process</div>
            <div class="vr-step-d">InsightRAG transcribes the speech and reads the slides and on-screen text.</div></div>
        <div class="vr-step"><div class="vr-step-n">3</div><div class="vr-step-t">Ask</div>
            <div class="vr-step-d">Get answers with timestamps you can click to watch that moment.</div></div>
    </div>""")
    st.stop()

scope = None
if len(videos) > 1:
    choice = st.selectbox("Ask about", ["All videos"] + videos)
    scope = None if choice == "All videos" else choice

question = st.chat_input("Ask about what was said or shown…")
if not ss.messages:
    st.caption("Try one of these")
    cols = st.columns(len(SUGGESTIONS))
    for k, text in enumerate(SUGGESTIONS):
        if cols[k].button(text, key=f"sugg_{k}", use_container_width=True):
            question = text

if question and question.strip():
    q = question.strip()
    target = scope or match_video_title(q, videos)          # chosen in the dropdown, or named in the question
    with st.spinner("Searching what was said and shown…"):
        results = ss.index.search(q, top_k=top_k_for(1 if target else len(videos)), mode="hybrid", video_id=target)
        answer, note = None, None
        if results:
            try:
                answer = generate_answer(q, results, cache=DiskCache(cache_dir()))
            except Exception as e:  # noqa: BLE001
                note = f"Couldn't write an answer ({e}). Showing the closest moments instead."
    ss.messages.append({"query": q, "results": results, "answer": answer, "note": note, "scope": target})
    ss.play = None
    st.rerun()

for i, m in enumerate(ss.messages):
    with st.chat_message("user"):
        st.write(m["query"])
    with st.chat_message("assistant", avatar="🎥"):
        render_assistant(i, m)
