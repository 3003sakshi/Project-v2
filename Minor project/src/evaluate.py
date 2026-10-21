"""
Evaluation harness for the two research questions in the project:

  RQ1  Does hybrid lexical+semantic retrieval beat a dense-only baseline for SPOKEN content?
  RQ2  Does adding VISUAL information improve retrieval for questions speech alone can't answer?

Questions file (JSON list), one entry per manually verified question:

  {"id": "q01", "question": "...", "type": "speech" | "visual" | "combined",
   "video_id": "lecture1",
   "gold": [{"start": "01:10", "end": "01:40"}]}        # time span(s) that contain the answer

A retrieved chunk is RELEVANT if its [start, end] overlaps a gold span of the same video
(temporal localisation: "did the system point to the right moment?"). With --strict it must
ALSO be of the right evidence modality (speech question -> speech chunk, visual question ->
visual chunk, combined -> either). Report both: the default does not hand the visual index a
free win on visual questions; strict shows whether the right *kind* of evidence was retrieved.

Retrieval metrics (per question, then averaged):
  Hit@k     1 if any relevant chunk is in the top k
  Recall@k  fraction of gold spans covered by at least one top-k chunk
  MRR       1 / rank of the first relevant chunk (0 if none in top max-k)
  nDCG@k    binary-relevance nDCG over the top-k chunks

Answer-grounding metrics (with --answers, needs GEMINI_API_KEY):
  citation validity   cited timestamps that fall inside a retrieved chunk window
  grounded sentences  sentences carrying at least one valid citation
  citation accuracy   valid citations that also fall inside a GOLD span (a proxy for correctness)

    python -m src.evaluate --questions eval/questions.json --library library --out eval/results [--strict] [--answers]
"""
import argparse
import csv
import json
import math
import os
import random
from typing import Dict, List, Optional, Sequence

from .chunking import parse_timestamp

CONFIGS: Dict[str, dict] = {
    "dense | speech": dict(mode="dense", modalities=("speech",)),          # <- baseline for RQ1
    "bm25 | speech": dict(mode="bm25", modalities=("speech",)),
    "hybrid | speech": dict(mode="hybrid", modalities=("speech",)),        # <- baseline for RQ2
    "dense | speech+visual": dict(mode="dense", modalities=("speech", "visual")),
    "bm25 | speech+visual": dict(mode="bm25", modalities=("speech", "visual")),
    "hybrid | speech+visual": dict(mode="hybrid", modalities=("speech", "visual")),  # <- full system
}
# (system, baseline, question type, research question)
COMPARISONS = [
    ("hybrid | speech", "dense | speech", "speech", "RQ1 hybrid vs dense (speech questions)"),
    ("hybrid | speech+visual", "hybrid | speech", "visual", "RQ2 +visual vs speech-only (visual questions)"),
    ("hybrid | speech+visual", "hybrid | speech", "combined", "RQ2 +visual vs speech-only (combined questions)"),
    ("hybrid | speech+visual", "hybrid | speech", "speech", "sanity: +visual must not hurt speech questions"),
]
KS = (1, 3, 5)


# ---------- metrics (pure functions; unit-tested) -----------------------------------------

def overlaps(chunk: dict, span: dict) -> bool:
    return chunk["start"] < span["end"] and chunk["end"] > span["start"]


ACCEPTED_MODALITIES = {"speech": {"speech"}, "visual": {"visual"}, "combined": {"speech", "visual"}}


def is_relevant(chunk: dict, video_id: Optional[str], gold: Sequence[dict],
                accepted_modalities: Optional[set] = None) -> bool:
    if video_id is not None and chunk["video_id"] != video_id:
        return False
    if accepted_modalities is not None and chunk["modality"] not in accepted_modalities:
        return False
    return any(overlaps(chunk, g) for g in gold)


def hit_at_k(rels: Sequence[bool], k: int) -> float:
    return float(any(rels[:k]))


def mrr(rels: Sequence[bool]) -> float:
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: Sequence[bool], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:k], 1) if r)
    n_rel = min(sum(rels), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / ideal if ideal else 0.0


def recall_at_k(chunks: Sequence[dict], video_id: Optional[str], gold: Sequence[dict], k: int,
                accepted_modalities: Optional[set] = None) -> float:
    if not gold:
        return 0.0
    covered = 0
    for g in gold:
        if any(is_relevant(c, video_id, [g], accepted_modalities) for c in chunks[:k]):
            covered += 1
    return covered / len(gold)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 0):
    """Mean(a-b) with a 95% bootstrap CI over questions. Returns (mean, lo, hi)."""
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1]


# ---------- running -----------------------------------------------------------------------

def load_questions(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        qs = json.load(f)
    for q in qs:
        q["gold"] = [{"start": parse_timestamp(g["start"]), "end": parse_timestamp(g["end"])} for g in q["gold"]]
        if q.get("type") not in ("speech", "visual", "combined"):
            raise ValueError(f"{q.get('id')}: type must be speech | visual | combined")
    return qs


def evaluate_retrieval(index, questions: List[dict], configs=CONFIGS, max_k: int = 10,
                       strict: bool = False) -> List[dict]:
    rows = []
    for q in questions:
        for name, cfg in configs.items():
            res = index.search(q["question"], top_k=max_k, video_id=q.get("video_id"), **cfg)
            acc = ACCEPTED_MODALITIES[q["type"]] if strict else None
            rels = [is_relevant(r, q.get("video_id"), q["gold"], acc) for r in res]
            row = {"id": q["id"], "type": q["type"], "config": name, "mrr": mrr(rels)}
            for k in KS:
                row[f"hit@{k}"] = hit_at_k(rels, k)
                row[f"recall@{k}"] = recall_at_k(res, q.get("video_id"), q["gold"], k, acc)
                row[f"ndcg@{k}"] = ndcg_at_k(rels, k)
            rows.append(row)
    return rows


def evaluate_answers(index, questions: List[dict], cache, top_k: int = 5) -> List[dict]:
    from .answer import generate_answer

    rows = []
    for q in questions:
        res = index.search(q["question"], top_k=top_k, video_id=q.get("video_id"))
        ans = generate_answer(q["question"], res, cache=cache)
        if not ans:
            continue
        v = ans["validation"]
        good = 0
        for c in v["citations"]:
            if c["valid"] and any(c["seconds"] >= g["start"] - 2 and c["seconds"] <= g["end"] + 2 for g in q["gold"]):
                good += 1
        rows.append({
            "id": q["id"], "type": q["type"], "n_citations": v["n_citations"],
            "citation_validity": v["validity_rate"], "grounded_sentences": v["grounded_rate"],
            "citation_accuracy": (good / v["n_citations"]) if v["n_citations"] else None,
        })
    return rows


# ---------- reporting ---------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def retrieval_table(rows: List[dict], qtype: Optional[str] = None) -> str:
    cols = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@5"]
    out = ["| config | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name in CONFIGS:
        sel = [r for r in rows if r["config"] == name and (qtype is None or r["type"] == qtype)]
        if sel:
            out.append(f"| {name} | " + " | ".join(f"{_mean([r[c] for r in sel]):.3f}" for c in cols) + " |")
    return "\n".join(out)


def build_report(rows: List[dict], answer_rows: Optional[List[dict]], n_questions: int,
                 strict: bool = False) -> str:
    types = sorted({r["type"] for r in rows})
    counts = {t: len({r["id"] for r in rows if r["type"] == t}) for t in types}
    md = [f"# InsightRAG evaluation ({n_questions} questions: "
          + ", ".join(f"{n} {t}" for t, n in counts.items()) + ")\n",
          "Relevance: " + ("STRICT (time overlap AND matching evidence modality)" if strict
                           else "time overlap with a gold span (temporal localisation)") + "\n",
          "## Overall\n", retrieval_table(rows), ""]
    for t in types:
        md += [f"## {t.capitalize()} questions (n={counts[t]})\n", retrieval_table(rows, t), ""]

    md += ["## Paired comparisons (mean difference, 95% bootstrap CI over questions)\n",
           "| comparison | n | metric | diff | 95% CI |", "|---|---|---|---|---|"]
    for sys_name, base_name, qtype, label in COMPARISONS:
        by = lambda cfg: {r["id"]: r for r in rows if r["config"] == cfg and r["type"] == qtype}
        s, b = by(sys_name), by(base_name)
        ids = sorted(set(s) & set(b))
        for metric in ("mrr", "hit@5"):
            if not ids:
                continue
            m, lo, hi = paired_bootstrap([s[i][metric] for i in ids], [b[i][metric] for i in ids])
            md.append(f"| {label} | {len(ids)} | {metric} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    md.append("\nA CI that excludes 0 suggests a real difference; with few questions CIs will be wide - say so in the report.\n")

    if answer_rows:
        md += ["## Answer grounding (hybrid | speech+visual, top-5)\n",
               "| type | n | citation validity | grounded sentences | citation accuracy (in gold span) |",
               "|---|---|---|---|---|"]
        for t in [None] + types:
            sel = [r for r in answer_rows if t is None or r["type"] == t]
            if sel:
                md.append(f"| {t or 'all'} | {len(sel)} | {_mean([r['citation_validity'] for r in sel]):.3f} | "
                          f"{_mean([r['grounded_sentences'] for r in sel]):.3f} | "
                          f"{_mean([r['citation_accuracy'] for r in sel]):.3f} |")
        md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--answers", action="store_true", help="also evaluate LLM answers (uses Gemini quota)")
    ap.add_argument("--strict", action="store_true",
                    help="relevant chunks must also match the question type's evidence modality")
    a = ap.parse_args()

    from .cache import DiskCache
    from .library import Library

    lib = Library(a.library)
    index = lib.load_index()
    questions = load_questions(a.questions)
    missing = {q["video_id"] for q in questions if q.get("video_id")} - set(index.videos())
    if missing:
        raise SystemExit(f"Questions reference videos not in the library: {sorted(missing)}")

    rows = evaluate_retrieval(index, questions, strict=a.strict)
    answer_rows = evaluate_answers(index, questions, DiskCache(str(lib.cache_dir))) if a.answers else None

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_question.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report = build_report(rows, answer_rows, len(questions), strict=a.strict)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {a.out}/report.md and per_question.csv")


if __name__ == "__main__":
    main()
