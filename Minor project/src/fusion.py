"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009)."""
from typing import Dict, Hashable, List, Optional, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Dict[str, Sequence[Hashable]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[Hashable, float]]:
    """
    rankings: {ranker_name: [item ids, best first]}.  score(d) = sum_r w_r / (k + rank_r(d)).
    Rank-based, so it needs no score normalisation between BM25 (unbounded) and cosine (0..1),
    or between the speech and visual lists.
    """
    scores: Dict[Hashable, float] = {}
    best_rank: Dict[Hashable, int] = {}
    for name, items in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], str(kv[0])))
