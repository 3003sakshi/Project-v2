"""
Step 3 - hybrid multimodal index.

For each modality (speech, visual) we keep two rankers over the same chunks:
    dense : Sentence-Transformer embeddings in a FAISS IndexFlatIP (cosine similarity)
    bm25  : lexical Okapi BM25
A query produces up to four ranked lists (dense_speech, bm25_speech, dense_visual, bm25_visual)
which are merged with Reciprocal Rank Fusion. Ablations (dense-only, BM25-only, speech-only,
visual-only) are just different arguments to `search`, which is what the evaluation uses.
"""
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from .bm25 import BM25, tokenize
from .config import CANDIDATE_DEPTH, RRF_K
from .fusion import reciprocal_rank_fusion

MODES = ("hybrid", "dense", "bm25")


class _ModalityIndex:
    def __init__(self, ids: List[int], vectors: np.ndarray, texts: List[str]):
        import faiss

        self.ids = ids
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.bm25 = BM25([tokenize(t) for t in texts])

    def dense(self, qvec: np.ndarray, depth: int, allowed: Optional[Set[int]]):
        # Exact search over a small flat index; fetch everything and filter by video if asked.
        D, I = self.faiss.search(qvec, len(self.ids))
        out = []
        for score, local in zip(D[0], I[0]):
            if local < 0:
                continue
            gi = self.ids[local]
            if allowed is None or gi in allowed:
                out.append((gi, float(score)))
                if len(out) >= depth:
                    break
        return out

    def lexical(self, qtokens: Sequence[str], depth: int, allowed: Optional[Set[int]]):
        local_allowed = None if allowed is None else {l for l, g in enumerate(self.ids) if g in allowed}
        return [(self.ids[l], s) for l, s in self.bm25.rank(qtokens, depth, local_allowed)]


class HybridIndex:
    def __init__(self, embedder=None):
        self._embedder = embedder
        self.chunks: List[dict] = []
        self.vectors = np.zeros((0, 0), dtype="float32")
        self._sub: Dict[str, _ModalityIndex] = {}

    # ---- embedding -------------------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from .retrieve import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _embed(self, texts: List[str]) -> np.ndarray:
        v = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False, batch_size=32)
        return np.asarray(v, dtype="float32")

    # ---- maintenance -----------------------------------------------------------------
    def add_chunks(self, chunks: List[dict]) -> None:
        if not chunks:
            return
        vecs = self._embed([c["text"] for c in chunks])
        self.chunks.extend(dict(c) for c in chunks)
        self.vectors = vecs if self.vectors.size == 0 else np.vstack([self.vectors, vecs])
        self._rebuild()

    def remove_video(self, video_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c["video_id"] != video_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype="float32")
        self._rebuild()

    def videos(self) -> List[str]:
        return sorted({c["video_id"] for c in self.chunks})

    def stats(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for c in self.chunks:
            out.setdefault(c["video_id"], {}).setdefault(c["modality"], 0)
            out[c["video_id"]][c["modality"]] += 1
        return out

    def _rebuild(self) -> None:
        self._sub = {}
        for m in sorted({c["modality"] for c in self.chunks}):
            ids = [i for i, c in enumerate(self.chunks) if c["modality"] == m]
            self._sub[m] = _ModalityIndex(ids, self.vectors[ids], [self.chunks[i]["text"] for i in ids])

    # ---- search ----------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        modalities: Iterable[str] = ("speech", "visual"),
        video_id: Optional[str] = None,
        depth: int = CANDIDATE_DEPTH,
        rrf_k: int = RRF_K,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not self.chunks:
            return []

        allowed = None if video_id is None else {i for i, c in enumerate(self.chunks) if c["video_id"] == video_id}
        qtokens = tokenize(query)
        qvec = None
        lists: Dict[str, list] = {}
        for m in modalities:
            sub = self._sub.get(m)
            if sub is None:
                continue
            if mode in ("dense", "hybrid"):
                if qvec is None:
                    qvec = self._embed([query])
                lists[f"dense_{m}"] = sub.dense(qvec, depth, allowed)
            if mode in ("bm25", "hybrid"):
                lists[f"bm25_{m}"] = sub.lexical(qtokens, depth, allowed)

        fused = reciprocal_rank_fusion({n: [g for g, _ in r] for n, r in lists.items()}, k=rrf_k, weights=weights)
        if not fused:
            return []
        top_score = fused[0][1]
        rank_maps = {n: {g: (pos, s) for pos, (g, s) in enumerate(r, 1)} for n, r in lists.items()}

        results = []
        for gi, score in fused[:top_k]:
            r = dict(self.chunks[gi])
            r["score"] = score
            r["rel"] = score / top_score if top_score else 0.0
            r["ranks"] = {n: rm[gi][0] for n, rm in rank_maps.items() if gi in rm}
            dense = rank_maps.get(f"dense_{r['modality']}", {}).get(gi)
            r["dense_score"] = dense[1] if dense else None
            results.append(r)
        return results

    # ---- persistence (used by the CLI / MCP library, not by the per-session web app) ----
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        tmp_c, tmp_v = os.path.join(directory, "chunks.json.tmp"), os.path.join(directory, "vectors.tmp.npy")
        with open(tmp_c, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        np.save(tmp_v, self.vectors)
        os.replace(tmp_c, os.path.join(directory, "chunks.json"))
        os.replace(tmp_v, os.path.join(directory, "vectors.npy"))

    @classmethod
    def load(cls, directory: str, embedder=None) -> "HybridIndex":
        idx = cls(embedder)
        cp, vp = os.path.join(directory, "chunks.json"), os.path.join(directory, "vectors.npy")
        if os.path.exists(cp) and os.path.exists(vp):
            with open(cp, encoding="utf-8") as f:
                idx.chunks = json.load(f)
            idx.vectors = np.load(vp)
            idx._rebuild()
        return idx
