"""Loading the FreshStack LangChain corpus and queries from Hugging Face."""

CORPUS_DATASET = "freshstack/corpus-oct-2024"
QUERIES_DATASET = "freshstack/queries-oct-2024"
SUBSET = "langchain"


def load_corpus():
    """Load the corpus chunks (doc id + text), the only material the assistant may search."""
    raise NotImplementedError


def load_queries():
    """Load queries with reference answers, nuggets and qrels, for evaluation only."""
    raise NotImplementedError


def split_queries(queries, dev_fraction: float = 0.3, seed: int = 42):
    """Split queries deterministically into a dev set and a held-out test set.

    Args:
        queries: The full query set.
        dev_fraction: Fraction of queries used during development.
        seed: Random seed for a reproducible split.
    """
    raise NotImplementedError
