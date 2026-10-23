"""Loading the FreshStack LangChain corpus and queries from Hugging Face."""

CORPUS_DATASET = "freshstack/corpus-oct-2024"
QUERIES_DATASET = "freshstack/queries-oct-2024"
SUBSET = "langchain"


def load_corpus() -> list[dict]:
    """Load the corpus chunks, the only material the assistant may search.

    Returns:
        Corpus chunks, each with a document ID and text.

    Raises:
        NotImplementedError: Not implemented yet.
    """
    raise NotImplementedError


def load_queries() -> list[dict]:
    """Load the evaluation queries; never used as assistant input.

    Returns:
        Queries with reference answers, nuggets and relevance judgments.

    Raises:
        NotImplementedError: Not implemented yet.
    """
    raise NotImplementedError


def split_queries(
    queries: list[dict], dev_fraction: float = 0.3, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split queries deterministically into a dev set and a held-out test set.

    Args:
        queries: The full query set.
        dev_fraction: Fraction of queries used during development.
        seed: Random seed for a reproducible split.

    Returns:
        The dev and test query lists.

    Raises:
        NotImplementedError: Not implemented yet.
    """
    raise NotImplementedError
