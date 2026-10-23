"""Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion (RRF)."""

from functools import lru_cache
from typing import TYPE_CHECKING

import bm25s
import numpy as np

from rag.index import Index, load_embedder, load_index

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# bge models expect this instruction on queries (not on passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RRF_K = 60
CANDIDATES = 100


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked lists of row indices with reciprocal rank fusion.

    Args:
        rankings: Ranked lists of row indices, best first.
        k: RRF smoothing constant; higher flattens the influence of top ranks.

    Returns:
        Row indices ordered by fused score, best first.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


@lru_cache(maxsize=1)
def _resources() -> tuple[Index, "SentenceTransformer"]:
    """Load the index and query embedder once per process.

    Returns:
        The index and the embedding model.
    """
    return load_index(), load_embedder()


def retrieve(question: str, k: int, mode: str = "hybrid") -> list[dict]:
    """Retrieve the top-k corpus chunks for a question.

    Args:
        question: The user question.
        k: Number of chunks to return.
        mode: ``hybrid`` (BM25 + dense via RRF), ``bm25`` or ``dense``.

    Returns:
        Chunks (``id``, ``text``, ``url``) ordered by relevance, best first.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode not in {"hybrid", "bm25", "dense"}:
        raise ValueError(f"Unknown retrieval mode: {mode!r}")
    index, embedder = _resources()
    rankings = []
    if mode in {"hybrid", "bm25"}:
        tokens = bm25s.tokenize([question], stopwords="en", show_progress=False)
        rows, _ = index.bm25.retrieve(tokens, k=CANDIDATES, show_progress=False)
        rankings.append(rows[0].tolist())
    if mode in {"hybrid", "dense"}:
        query = embedder.encode(QUERY_INSTRUCTION + question, normalize_embeddings=True)
        rankings.append(np.argsort(-(index.embeddings @ query))[:CANDIDATES].tolist())
    return [index.chunks[row] for row in rrf(rankings)[:k]]
