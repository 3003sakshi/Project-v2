"""
Okapi BM25 lexical ranker (no external dependency).

Lexical retrieval matters for video because exact terms - names, acronyms, formulas,
slide titles and OCR'd on-screen text - are often poorly captured by dense embeddings.
"""
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this that "
    "these those i you he she we they them his her our your their my me do does did so than then there here "
    "what which who whom when where why how about into over under up down out not no yes can could would should "
    "will just also very more most some any each other such".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    toks = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural folding: "networks" -> "network"
        toks.append(t)
    return toks


class BM25:
    def __init__(self, docs_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs_tokens):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def rank(self, query_tokens: Sequence[str], depth: int, allowed: Optional[Set[int]] = None
             ) -> List[Tuple[int, float]]:
        """Return [(doc_index, score)] best-first; documents with no term overlap are omitted."""
        scores: Dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            for i, tf in self.postings.get(term, ()):
                if allowed is not None and i not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
