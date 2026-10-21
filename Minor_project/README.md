# 🎥 InsightRAG — Multimodal, Timestamp-Grounded Question Answering over Videos

InsightRAG answers natural-language questions about videos by searching **what was said** *and* **what was shown on screen**, then replying with citations to exact timestamps — citations the system **verifies automatically** against the evidence it retrieved.

> **MCP-enabled multimodal retrieval framework for temporal question answering over videos**
> Team: Sakshi Shekhawat, Juhi Purohit

---

## ✨ Key Features

- **Speech understanding** — transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), kept at line level so every sentence has its own timestamp.
- **Visual understanding** — keyframes are sampled, near-duplicate frames are skipped, and a vision-language model (Gemini) describes each frame and reads its on-screen text (slides, diagrams, formulas, labels).
- **Hybrid retrieval** — lexical **BM25** and semantic **dense embeddings** (Sentence-Transformers + FAISS), run separately for speech and visual evidence and merged with **Reciprocal Rank Fusion (RRF)**.
- **Verified citations** — answers cite `[video @ mm:ss]`; each citation is checked against the transcript line or frame it points to. Invented timestamps are flagged.
- **Honest "not found" answers** — if the video does not contain the answer, the system says so instead of guessing, and shows no misleading evidence.
- **Multiple videos** — search across all videos, name a video in the question, or pick one from a dropdown; citations carry the video title.
- **Jump to the moment** — every cited timestamp is a button that plays the video from that point.
- **MCP server** — exposes the video library to AI assistants (e.g. Claude Desktop) through a small set of controlled tools.
- **Evaluation harness** — measures retrieval quality across six configurations and the grounding of generated answers.
- **Runs on a CPU-only laptop** — no GPU required.

### Example

> **Q:** What does the stop sign slide show?
> **A:** The slide is titled "FOOLING THE AI" and shows that stickers placed on a stop sign made an AI system read it as "speed limit 45" `✓ [Intro to cv @ 01:41]`
> `✓ 1/1 timestamps verified against the video evidence`

> **Q:** What is the speaker's favorite programming language?
> **A:** *Not found in your videos.*

---

## 🏗️ Architecture

```
 video ──► Whisper (CPU, int8) ──► speech chunks (~40 s, line-level timing) ──┐
   │                                                                          │    ┌─ dense_speech ─┐
   └────► keyframes ─► de-dup ─► Gemini vision ─► visual chunks              ├─► index ─ bm25_speech  ─┤─► RRF ─► top-k
          (every N s)  (skip      (description + on-screen text)            ─┘    ├─ dense_visual ─┤   evidence
                       static                                                      └─ bm25_visual  ─┘      │
                       slides)                                                                             ▼
                                              Gemini answer  ◄── modality-tagged context ◄─────────────────┘
                                                   │
                                                   ▼
                    citation validator: does [video @ mm:ss] fall on a retrieved transcript line / frame?
```

| Module | Role |
|---|---|
| `src/transcribe.py` | faster-whisper transcription |
| `src/chunking.py` | Merges segments into ~40 s chunks while keeping every line's timing; timestamp helpers; shared chunk schema |
| `src/visual.py` | Keyframe sampling, near-duplicate skipping, batched Gemini vision calls, visual chunks |
| `src/bm25.py` | Okapi BM25 (dependency-free) |
| `src/fusion.py` | Reciprocal Rank Fusion |
| `src/index.py` | `HybridIndex`: per-modality FAISS + BM25; `search(mode, modalities, video_id)`; save/load |
| `src/retrieve.py` | Embedder loader, LLM context builder, video-title detection in questions |
| `src/answer.py` | Answer generation, "not found" detection, **citation validation** |
| `src/pipeline.py` | Orchestrates transcribe → chunk → (vision), with caching |
| `src/cache.py` | On-disk cache for transcripts, frame descriptions and answers |
| `src/llm.py` | Gemini wrapper (rate spacing, retries, clear error messages) |
| `src/library.py` | Persistent single-user library used by the CLI and MCP server |
| `src/mcp_server.py` | MCP server (stdio) |
| `src/cli.py` | Command-line interface |
| `src/evaluate.py` | Evaluation harness |
| `src/warmup.py` | Pre-computes transcripts, slide descriptions and answers for chosen videos/questions |
| `src/check_models.py` | Lists which Gemini models your API key can use |
| `app.py` | Streamlit web app |

### Design choices
- **Hybrid retrieval.** Dense embeddings handle paraphrase ("how does the model learn?" ↔ "gradient descent updates the weights"). BM25 handles exact tokens — names, acronyms, formulas, slide titles and OCR text — that embeddings blur.
- **RRF.** Rank-based, so BM25 scores (unbounded) and cosine scores (0–1), or the speech and visual lists, merge without score normalisation: `score(d) = Σ 1 / (60 + rank)`.
- **Visual chunks are text.** Frame descriptions and on-screen text are indexed exactly like transcript chunks, so one retrieval pipeline serves both modalities and every result keeps its timestamp.
- **Line-level citation checking.** Speech chunks keep their individual transcript lines (about 5 s each). The model is shown each line's time and a citation is *valid* only if it lands on a retrieved line (±2 s) or a sampled-frame window. Checking against whole 40 s chunks would accept almost any timestamp; line-level checking makes the verification meaningful.

---

## 📁 Project Structure

```
insight-rag/
├── app.py                    # Streamlit web app
├── requirements.txt
├── demo_questions.txt        # example questions for src.warmup
├── .env.example              # copy to .env and add GEMINI_API_KEY
├── src/                      # the pipeline (see table above)
├── tests/                    # 28 automated tests
└── eval/
    └── questions.example.json
```

---

## ⚙️ Setup

**Prerequisites:** Python 3.10+. `ffmpeg` on your `PATH` is recommended (macOS `brew install ffmpeg` · Ubuntu `sudo apt install ffmpeg` · Windows `winget install ffmpeg`).

```bash
git clone https://github.com/3003sakshi/insight-rag
cd insight-rag
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env   — then add your GEMINI_API_KEY
```

### Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required for slide analysis and written answers. Without it, speech search still works and the app shows the exact matching transcript lines. |
| `GEMINI_MODEL` | Model that writes answers (optional) |
| `GEMINI_VISION_MODEL` | Model that describes keyframes (optional) |
| `GEMINI_FRAME_BATCH` | Frames sent per vision request (default 8) |
| `GEMINI_MIN_INTERVAL` | Minimum seconds between API calls (default 4) |
| `INSIGHTRAG_CACHE_DIR` | Folder where processed results are kept between runs, e.g. `library/cache` |
| `INSIGHTRAG_LIBRARY` | Folder used by the CLI and MCP server (default `library`) |

---

## 🚀 Usage

### 1. Web app
```bash
python -m streamlit run app.py
```
1. **Upload** a video in the sidebar, give it a title, and click **Process video**. Tick *Also analyze slides & visuals* to make slides, diagrams and on-screen text searchable.
2. **Ask** in the chat box, or click a suggested question.
3. Read the answer: green ✓ chips are timestamps verified against the video, red ✕ chips are not. Use the **▶** buttons to watch each cited moment.
4. Open *See the evidence behind this answer* to see each supporting moment, labelled **SAID** (speech) or **SHOWN** (on screen), with a frame thumbnail for visual matches.

**Several videos:** add each one with its own title. Ask across all of them, name a video in your question ("In Lecture 2, what…"), or choose one from the *Ask about* dropdown.

Each browser session has its own in-memory index and private temp folder, so people using a shared deployment never see each other's videos.

### 2. Command line (persistent local library)
```bash
python -m src.cli add path/to/lecture.mp4 --id lecture1          # --no-visual to skip slide analysis
python -m src.cli list
python -m src.cli search "formula on the loss slide" --modality visual
python -m src.cli ask "How is overfitting explained?" --video lecture1
python -m src.cli delete lecture1
```

### 3. MCP server (Claude Desktop and other MCP clients)
Exposes the **local library** through five controlled tools:

| Tool | Purpose |
|---|---|
| `list_videos` | Indexed videos and chunk counts |
| `search_video` | Hybrid search → timestamped moments (`modality`, `mode`, `top_k` options) |
| `ask_video` | Answer with validated citations and a `found_in_video` flag |
| `get_transcript_window` | Everything indexed between two timestamps (≤ 10 min) |
| `index_video` | Index a file — **only from `library/inbox/`**; other paths are refused |

Safety: tools are read-only except `index_video`; inputs are validated and clamped; paths cannot escape the inbox; logs go to stderr so the stdio channel stays clean.

Add to your client config (e.g. Claude Desktop `claude_desktop_config.json`) using absolute paths:
```json
{
  "mcpServers": {
    "insightrag": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/insight-rag",
      "env": { "GEMINI_API_KEY": "your_key_here", "INSIGHTRAG_LIBRARY": "/absolute/path/to/insight-rag/library" }
    }
  }
}
```
Index videos with the CLI first, then ask the assistant, e.g. *"Search my lecture videos for the slide about the confusion matrix."*

---

## 📊 Evaluation

The harness tests two research questions directly:

- **RQ1** — Does **hybrid** retrieval (BM25 + dense) beat a **dense-only** baseline on *spoken* content?
- **RQ2** — Does adding **visual** evidence improve retrieval for questions that speech alone cannot answer?

**1. Build a question set** (40–60 questions across 2–3 videos; verify each gold time span by watching). Copy `eval/questions.example.json` to `eval/questions.json`:
```json
{"id": "v01", "type": "visual", "video_id": "lecture1",
 "question": "Which slide shows the architecture diagram?",
 "gold": [{"start": "06:40", "end": "07:00"}]}
```
`type` is `speech` (answer only in the audio), `visual` (only on screen) or `combined` (needs both). Write questions *without copying wording* from the transcript or slides, otherwise BM25 gets an unfair advantage.

**2. Run**
```bash
python -m src.evaluate --questions eval/questions.json --library library --out eval/results
python -m src.evaluate --questions eval/questions.json --strict      # stricter relevance (below)
python -m src.evaluate --questions eval/questions.json --answers     # also evaluates generated answers
```
Outputs `eval/results/report.md` and `per_question.csv`.

**Configurations compared:** dense / BM25 / hybrid × speech-only / speech+visual (6 in total).

**Retrieval metrics:** Hit@1/3/5, Recall@5 (fraction of gold spans covered), MRR, nDCG@5, with paired-bootstrap 95% confidence intervals for the RQ1 and RQ2 comparisons.

**Answer metrics (`--answers`):** *citation validity* (cited moments that exist in the retrieved evidence), *grounded sentences* (sentences with at least one valid citation), and *citation accuracy* (valid citations that also fall inside a gold span).

**Relevance definitions — state which one you use:**
- *Default (temporal localisation):* a chunk is relevant if its time window overlaps a gold span, regardless of modality — "did the system point to the right moment?".
- *`--strict`:* it must also be the right evidence type (speech question → speech chunk, visual question → visual chunk). Under `--strict`, speech-only retrieval trivially scores 0 on visual questions, so read RQ2 in the default mode first.

**Reporting notes:** citation validity measures *grounding* (is the cited moment among the retrieved evidence), not factual *correctness*; that is why citation accuracy against gold spans is reported separately. With small question sets the confidence intervals will be wide — report them rather than only the means.

---

## ⚡ Efficiency

- Transcription runs `faster-whisper` (`base`, int8) on the CPU; `tiny` is faster and `small` is more accurate.
- Frames are sampled at a fixed interval (every 5 s in the web app, configurable in the CLI), and **near-identical frames are dropped**, so a slide that stays on screen costs one description rather than many. Frames are sent to the vision model in **batches of 8**, and a hard cap (120 frames) bounds very long videos.
- Transcripts, per-frame descriptions and written answers are **cached by video content hash**, so re-processing the same file or repeating a question is instant. Set `INSIGHTRAG_CACHE_DIR` to keep the cache between runs.

### Pre-computing results (warm-up)
To make a presentation or evaluation run instantly, pre-compute everything once:
```bash
python -m src.warmup --video test.mp4 --id "Intro to cv" --questions demo_questions.txt
# several videos: repeat --video/--id in the order you will upload them
```
Keep `INSIGHTRAG_CACHE_DIR` pointing at the same folder. When you later upload the same video with the same title and ask the same questions, processing and answers come straight from the cache.

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| Speech-to-text | faster-whisper |
| Vision-language model | Google Gemini |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`IndexFlatIP`, cosine similarity) |
| Lexical search | Okapi BM25 (custom implementation) |
| Rank fusion | Reciprocal Rank Fusion |
| Answer generation | Google Gemini |
| Tool interface | Model Context Protocol (MCP) Python SDK |
| UI | Streamlit |
| Video / frame handling | moviepy, Pillow |

---

## ⚠️ Known Limitations

- **Verified citation ≠ true claim.** A green ✓ means the cited timestamp exists in the retrieved evidence, not that the sentence is fully correct. A sentence can combine facts from different moments and still carry valid timestamps.
- **Vision errors.** Frame descriptions and on-screen-text reading can be wrong, and citation checking cannot catch a wrong description that is cited correctly.
- **Sampling gaps.** Slides visible for less than the sampling interval can be missed; a slide that stays on screen is represented by its first appearance only.
- **Transcription.** Speech-to-text can mishear names or numbers. Transcription uses `task="translate"`, so all speech becomes English; change `task` in `process_video` to keep the original language.
- **Scale.** `IndexFlatIP` and the in-process BM25 are exact/brute-force — fine for hundreds to thousands of chunks, not for very large libraries.
- **Retrieval always returns the closest chunks,** even when nothing is truly relevant. The answer step handles this by replying "not found", but some loosely related chunks may be retrieved first.
- **Sessions.** The web app is ephemeral per session by design. The CLI/MCP library is single-user and local, with no authentication.
- BM25 uses a simple tokenizer (stop-words and plural folding); there is no full stemming.

## 🔭 Future Work
- Direct YouTube link ingestion with duration limits
- Claim-level verification: re-check every generated sentence against its cited evidence
- Cross-encoder re-ranking of the fused top-k
- Scene-change-based (rather than fixed-interval) keyframe selection
- Speaker diarization; original-language transcripts
- Approximate FAISS index for large libraries; authenticated, persistent per-user libraries

---

## 🧪 Tests
```bash
python -m pytest -q
```
28 tests cover chunking, BM25, RRF, the hybrid index, line-level citation validation, "not found" detection, video-title routing, retrieval metrics, frame de-duplication on a synthetic video, batching and caching with a mocked vision model, the MCP tools and their path-traversal guard, the warm-up cache, and the evaluation runner. They use a fake embedder and a mocked Gemini client, so they run offline in a few seconds.

## 📄 License
MIT License. See the [LICENSE](LICENSE) file for details.