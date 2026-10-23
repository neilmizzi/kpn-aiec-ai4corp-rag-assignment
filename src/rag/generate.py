"""Answer generation grounded in retrieved chunks, with doc-ID citations and abstention."""


def answer(question: str, chunks):
    """Answer a question from retrieved chunks, or state that the corpus lacks the information.

    Args:
        question: The user question.
        chunks: Retrieved corpus chunks to ground the answer in.
    """
    raise NotImplementedError
