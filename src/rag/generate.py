"""Answer generation grounded in retrieved chunks, with doc-ID citations and abstention."""


def answer(question: str, chunks: list[dict]) -> str:
    """Answer a question from retrieved chunks, or state that the corpus lacks the information.

    Args:
        question: The user question.
        chunks: Retrieved corpus chunks to ground the answer in.

    Returns:
        The answer text with doc-ID citations, or an explicit abstention.

    Raises:
        NotImplementedError: Not implemented yet.
    """
    raise NotImplementedError
