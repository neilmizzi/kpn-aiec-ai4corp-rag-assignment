"""Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion."""


def retrieve(question: str, k: int) -> list[dict]:
    """Retrieve the top-k corpus chunks for a question.

    Args:
        question: The user question.
        k: Number of chunks to return.

    Returns:
        Chunks ordered by fused relevance score, best first.

    Raises:
        NotImplementedError: Not implemented yet.
    """
    raise NotImplementedError
