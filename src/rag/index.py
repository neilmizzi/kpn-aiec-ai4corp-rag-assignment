"""Building and persisting the BM25 and dense (OpenAI embeddings) indexes."""


def build_index():
    """Embed the corpus, build BM25, and persist both under DATA_DIR."""
    raise NotImplementedError


def load_index():
    """Load the persisted BM25 and dense indexes from DATA_DIR."""
    raise NotImplementedError
