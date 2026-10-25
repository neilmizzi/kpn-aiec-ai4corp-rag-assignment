"""Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion (RRF)."""

import re
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


def question_language(question: str) -> str | None:
    """Detect one explicitly named language family without guessing from APIs.

    Args:
        question: User question, including any code.

    Returns:
        ``python``, ``javascript`` (including TypeScript), or None for ambiguous/absent cues.
    """
    python = bool(re.search(r"\bpython\b", question, re.IGNORECASE))
    javascript = bool(
        re.search(r"\b(?:javascript|typescript|js|ts|langchainjs)\b", question, re.IGNORECASE)
    )
    return ("python" if python else "javascript") if python != javascript else None


def chunk_language(doc_id: str) -> str | None:
    """Identify clear source-language families from corpus paths.

    Args:
        doc_id: Corpus chunk ID containing its source path.

    Returns:
        Language family, or None when the path does not establish one.
    """
    path = doc_id.rsplit("_", 2)[0].lower()
    if path.startswith("langchainjs/") or path.endswith((".js", ".jsx", ".ts", ".tsx")):
        return "javascript"
    if path.startswith("langchain/") or path.endswith((".py", ".ipynb")):
        return "python"
    return None


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
        mode: ``hybrid`` (BM25 + dense via RRF), ``bm25``, ``dense``, or experimental
            ``hybrid-language`` (demote clearly mismatched language families).

    Returns:
        Chunks (``id``, ``text``, ``url``) ordered by relevance, best first.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode not in {"hybrid", "bm25", "dense", "hybrid-language"}:
        raise ValueError(f"Unknown retrieval mode: {mode!r}")
    index, embedder = _resources()
    rankings = []
    if mode in {"hybrid", "hybrid-language", "bm25"}:
        tokens = bm25s.tokenize([question], stopwords="en", show_progress=False)
        rows, _ = index.bm25.retrieve(tokens, k=CANDIDATES, show_progress=False)
        rankings.append(rows[0].tolist())
    if mode in {"hybrid", "hybrid-language", "dense"}:
        query = embedder.encode(QUERY_INSTRUCTION + question, normalize_embeddings=True)
        rankings.append(np.argsort(-(index.embeddings @ query))[:CANDIDATES].tolist())
    rows = rrf(rankings)
    language = question_language(question) if mode == "hybrid-language" else None
    if language:
        rows.sort(key=lambda row: chunk_language(index.chunks[row]["id"]) not in (None, language))
    return [index.chunks[row] for row in rows[:k]]
