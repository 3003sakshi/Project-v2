import zlib
from pathlib import Path

import numpy as np
import pytest

from src.answer import validate_citations
from src.bm25 import BM25, tokenize
from src.chunking import chunk_segments, format_timestamp, make_chunk, parse_timestamp
from src.evaluate import hit_at_k, is_relevant, mrr, ndcg_at_k, paired_bootstrap, recall_at_k
from src.fusion import reciprocal_rank_fusion
from src.index import HybridIndex
from src.library import resolve_in_dir
from src.retrieve import build_context_block
from src.transcribe import TranscriptSegment


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (no model download needed)."""
    dim = 128

    def encode(self, texts, **kw):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def make_index():
    idx = HybridIndex(FakeEmbedder())
    idx.add_chunks([
        make_chunk("v1", "speech", 0, 0, 40, "gradient descent updates weights using the learning rate"),
        make_chunk("v1", "speech", 1, 40, 80, "overfitting happens when the model memorises training data"),
        make_chunk("v1", "visual", 0, 60, 80, "A slide showing a confusion matrix. On-screen text: Precision Recall F1"),
        make_chunk("v2", "speech", 0, 0, 40, "kubernetes schedules containers across the cluster"),
    ])
    return idx


# ---- timestamps / chunking
def test_timestamp_roundtrip():
    assert format_timestamp(75) == "01:15" and format_timestamp(3725) == "1:02:05"
    assert parse_timestamp("01:15") == 75 and parse_timestamp("1:02:05") == 3725 and parse_timestamp(12) == 12.0


def test_chunking_respects_window_and_timing():
    segs = [TranscriptSegment("v", i * 10.0, i * 10.0 + 9.0, f"sentence number {i}") for i in range(10)]
    chunks = chunk_segments(segs, target_seconds=40)
    assert chunks[0]["start"] == 0.0
    assert all(c["end"] - c["start"] <= 40 for c in chunks)
    assert chunks[-1]["end"] == 99.0
    assert sum(len(c["text"].split()) for c in chunks) == 30  # nothing lost
    assert all(c["modality"] == "speech" for c in chunks)


# ---- BM25 / RRF
def test_bm25_exact_term_wins():
    docs = [tokenize(t) for t in ["the cat sat", "quantum entanglement of photons", "dogs and cats"]]
    bm = BM25(docs)
    assert bm.rank(tokenize("entanglement"), 5)[0][0] == 1
    assert bm.rank(tokenize("zzzz"), 5) == []  # no overlap -> no results, not random ones


def test_plural_folding():
    assert tokenize("neural networks") == tokenize("neural network")


def test_rrf_prefers_items_ranked_by_both():
    fused = reciprocal_rank_fusion({"a": ["x", "y", "z"], "b": ["y", "x", "w"]})
    assert [i for i, _ in fused[:2]] == ["x", "y"] or [i for i, _ in fused[:2]] == ["y", "x"]
    assert {i for i, _ in fused} == {"x", "y", "z", "w"}
    only_a = reciprocal_rank_fusion({"a": ["p"], "b": ["q", "p"]})
    assert only_a[0][0] == "p"


# ---- index
def test_hybrid_search_modalities_and_filters():
    idx = make_index()
    res = idx.search("confusion matrix precision", top_k=3)
    assert res[0]["modality"] == "visual"
    assert idx.search("confusion matrix", modalities=("speech",), mode="bm25") == []
    assert all(r["video_id"] == "v2" for r in idx.search("containers", video_id="v2", top_k=5))
    r = idx.search("gradient descent", top_k=1, mode="hybrid")[0]
    assert r["video_id"] == "v1" and r["rel"] == 1.0 and "dense_speech" in r["ranks"] and "bm25_speech" in r["ranks"]


def test_search_modes_validate():
    with pytest.raises(ValueError):
        make_index().search("x", mode="nope")


def test_remove_video_and_stats():
    idx = make_index()
    assert idx.videos() == ["v1", "v2"]
    idx.remove_video("v1")
    assert idx.videos() == ["v2"] and idx.vectors.shape[0] == 1
    assert idx.search("gradient descent", top_k=3)[0]["video_id"] == "v2"
    idx.remove_video("v2")
    assert idx.search("anything") == []


def test_save_load_roundtrip(tmp_path):
    idx = make_index()
    idx.save(str(tmp_path))
    loaded = HybridIndex.load(str(tmp_path), embedder=FakeEmbedder())
    assert loaded.chunks == idx.chunks
    assert loaded.search("kubernetes", top_k=1)[0]["video_id"] == "v2"


# ---- citation validation
def test_validate_citations():
    res = [make_chunk("v1", "speech", 0, 10, 50, "a"), make_chunk("v1", "visual", 0, 120, 140, "b")]
    ans = ("The model updates weights [v1 @ 00:30]. A slide shows a matrix **[v1 @ 02:05]**. "
           "This claim is invented [v1 @ 09:00]. And another unsupported sentence here without any citation.")
    v = validate_citations(ans, res)
    assert [c["valid"] for c in v["citations"]] == [True, True, False]
    assert v["n_valid"] == 2 and abs(v["validity_rate"] - 2 / 3) < 1e-9
    assert v["citations"][1]["modality"] == "visual"
    wrong_video = validate_citations("Fact here is stated [v9 @ 00:30].", res)
    assert wrong_video["n_valid"] == 0
    bare = validate_citations("Fact here is stated [00:30].", res)
    assert bare["n_valid"] == 1
    assert validate_citations("No citations at all in this answer.", res)["validity_rate"] is None


def test_context_block_tags_modality():
    res = [make_chunk("v1", "speech", 0, 10, 50, "spoken"), make_chunk("v1", "visual", 0, 120, 140, "shown")]
    ctx = build_context_block(res)
    assert "type=SPEECH" in ctx and "window 00:10-00:50" in ctx and "type=VISUAL" in ctx and "frame at 02:00" in ctx


def test_segment_level_citation_validation():
    segs = [{"start": 0, "end": 8, "text": "intro"}, {"start": 8, "end": 20, "text": "cnn"},
            {"start": 30, "end": 38, "text": "end"}]
    res = [make_chunk("v", "speech", 0, 0, 38, "intro cnn end", segments=segs)]
    ok = validate_citations("Cnn is covered here in detail [v @ 00:10].", res)
    bad = validate_citations("Invented moment inside the window [v @ 00:25].", res)   # in the silent gap
    assert ok["n_valid"] == 1 and bad["n_valid"] == 0          # window-level validation would accept both
    ctx = build_context_block(res)
    assert "[00:08] cnn" in ctx and "[00:30] end" in ctx


def test_chunks_keep_segments():
    segs = [TranscriptSegment("v", 0, 9, "a b c d"), TranscriptSegment("v", 10, 19, "e f g h")]
    c = chunk_segments(segs, target_seconds=40)[0]
    assert [s["start"] for s in c["segments"]] == [0, 10] and c["segments"][1]["text"] == "e f g h"


# ---- evaluation metrics
def test_metrics():
    assert mrr([False, True, True]) == 0.5 and mrr([False, False]) == 0.0
    assert hit_at_k([False, False, True], 2) == 0.0 and hit_at_k([False, False, True], 3) == 1.0
    assert ndcg_at_k([True, False, False], 3) == 1.0
    assert 0 < ndcg_at_k([False, True, False], 3) < 1.0
    chunks = [make_chunk("v", "speech", 0, 0, 40, "x"), make_chunk("v", "speech", 1, 40, 80, "y")]
    gold = [{"start": 10, "end": 20}, {"start": 200, "end": 220}]
    assert recall_at_k(chunks, "v", gold, 2) == 0.5
    assert is_relevant(chunks[0], "v", gold) and not is_relevant(chunks[1], "v", gold)
    assert not is_relevant(chunks[0], "other", gold)


def test_paired_bootstrap_detects_real_difference():
    a, b = [1.0] * 20, [0.0] * 20
    m, lo, hi = paired_bootstrap(a, b, n=500)
    assert m == 1.0 and lo > 0.9
    m, lo, hi = paired_bootstrap([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], n=200)
    assert m == 0 and lo <= 0 <= hi


# ---- MCP safety guard
def test_inbox_path_guard(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.mp4").write_bytes(b"x")
    assert resolve_in_dir(inbox, "a.mp4").name == "a.mp4"
    for bad in ("../secret.txt", "/etc/passwd", "sub/../../x"):
        with pytest.raises(ValueError):
            resolve_in_dir(inbox, bad)
