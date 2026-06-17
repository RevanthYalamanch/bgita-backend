# backend/rerank.py
"""Maximal Marginal Relevance (MMR) reranking for RAG candidates.

Pure-Python cosine math over the embeddings we already retrieve — no extra model
call or dependency. Given a larger candidate set pulled by vector distance, MMR
picks a final top-k that is both relevant to the query AND diverse, so we don't
hand the model three near-identical passages and waste the context window.

score(c) = lambda * sim(query, c) - (1 - lambda) * max sim(c, already_selected)

lambda=1 reduces to plain similarity ranking; lower lambda favors diversity.
"""
import math


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(sum(x * x for x in a))


def cosine(a, b):
    if not a or not b:
        return 0.0
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def mmr_select(query_vec, candidates, k, lambda_mult=0.7):
    """Reorder/select up to `k` candidates by MMR.

    `candidates` is a list of dicts with "content" and "embedding" (a float list,
    in query-distance order). Returns the selected "content" strings. Falls back
    to the input order if embeddings are missing/unparseable, so retrieval still
    works even if reranking can't run.
    """
    if not candidates:
        return []

    usable = bool(query_vec) and all(c.get("embedding") for c in candidates)
    if not usable:
        return [c["content"] for c in candidates[:k]]

    relevance = {id(c): cosine(query_vec, c["embedding"]) for c in candidates}
    selected = []
    remaining = list(candidates)

    while remaining and len(selected) < k:
        best, best_score = None, None
        for c in remaining:
            if not selected:
                score = relevance[id(c)]
            else:
                redundancy = max(cosine(c["embedding"], s["embedding"]) for s in selected)
                score = lambda_mult * relevance[id(c)] - (1 - lambda_mult) * redundancy
            if best_score is None or score > best_score:
                best, best_score = c, score
        selected.append(best)
        remaining = [c for c in remaining if c is not best]

    return [c["content"] for c in selected]
