"""Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion."""


def retrieve(question: str, k: int):
    """Retrieve the top-k corpus chunks for a question.

    Args:
        question: The user question.
        k: Number of chunks to return.
    """
    raise NotImplementedError
