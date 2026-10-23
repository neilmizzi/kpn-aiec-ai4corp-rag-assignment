"""Building and persisting the BM25 and dense (embedding) indexes."""


def build_index() -> None:
    """Embed the corpus, build BM25, and persist both under DATA_DIR.

    Raises:
        NotImplementedError: Not implemented yet.
    """
    raise NotImplementedError


def load_index() -> object:
    """Load the persisted BM25 and dense indexes from DATA_DIR.

    Returns:
        The loaded hybrid index.

    Raises:
        NotImplementedError: Not implemented yet.
    """
    raise NotImplementedError
