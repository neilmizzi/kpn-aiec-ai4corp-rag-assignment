"""Building and loading the hybrid search index over the corpus.

The index covers corpus chunks only (never queries, reference answers or nuggets), and every row
keeps its corpus document ID so answers can cite it. Layout under ``DATA_DIR/index``:

* ``ids.json``: chunk IDs in index row order.
* ``bm25/``: ``bm25s`` sparse keyword index.
* ``embeddings.npy``: L2-normalised dense vectors (float32), one row per chunk.
* ``parts/``: embedding shards, so an interrupted build resumes where it stopped.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s
import numpy as np

from rag.config import DATA_DIR, EMBEDDING_MODEL
from rag.data import load_corpus

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

INDEX_DIR = DATA_DIR / "index"
SHARD_SIZE = 2000


@dataclass
class Index:
    """Loaded hybrid index, aligned row-by-row with ``chunks``.

    Attributes:
        chunks: Corpus chunks (``id``, ``text``, ``url``) in index row order.
        bm25: Sparse keyword index.
        embeddings: L2-normalised dense vectors, one row per chunk.
    """

    chunks: list[dict]
    bm25: bm25s.BM25
    embeddings: np.ndarray


def doc_text(chunk: dict) -> str:
    """Build the text that gets indexed for a chunk: its source path, then its content.

    The path (e.g. ``langchain/docs/how_to/document_loader_json.mdx``) is part of the corpus and
    carries strong signal for both keyword and semantic matching.

    Args:
        chunk: Corpus chunk with ``id`` and ``text``.

    Returns:
        Text to index.
    """
    path = chunk["id"].rsplit("_", 2)[0]
    return f"{path}\n{chunk['text']}"


def load_embedder() -> "SentenceTransformer":
    """Load the local sentence-transformers embedding model (imports torch lazily).

    Returns:
        A ``SentenceTransformer`` for ``EMBEDDING_MODEL``.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def build_index(index_dir: Path = INDEX_DIR) -> None:
    """Build the BM25 and dense indexes over the corpus and save them; skips finished parts.

    Args:
        index_dir: Output directory.
    """
    corpus = load_corpus()
    texts = [doc_text(c) for c in corpus]
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "ids.json").write_text(json.dumps([c["id"] for c in corpus]))

    if not (index_dir / "bm25").exists():
        print(f"Building BM25 over {len(texts)} chunks...")
        bm25 = bm25s.BM25()
        bm25.index(bm25s.tokenize(texts, stopwords="en"))
        bm25.save(index_dir / "bm25")

    if (index_dir / "embeddings.npy").exists():
        return
    parts = index_dir / "parts"
    parts.mkdir(exist_ok=True)
    model = load_embedder()
    n_shards = -(-len(texts) // SHARD_SIZE)
    for shard in range(n_shards):
        path = parts / f"{shard:04d}.npy"
        if path.exists():
            continue
        print(f"Embedding shard {shard + 1}/{n_shards}...", flush=True)
        batch = texts[shard * SHARD_SIZE : (shard + 1) * SHARD_SIZE]
        vectors = model.encode(batch, batch_size=32, normalize_embeddings=True)
        np.save(path, vectors.astype(np.float32))
    shards = [np.load(parts / f"{s:04d}.npy") for s in range(n_shards)]
    np.save(index_dir / "embeddings.npy", np.concatenate(shards))
    print("Index built.")


def load_index(index_dir: Path = INDEX_DIR) -> Index:
    """Load the saved index and the corpus chunks it was built from.

    Args:
        index_dir: Directory written by ``build_index``.

    Returns:
        The loaded index.

    Raises:
        FileNotFoundError: If the index has not been built (run ``rag index``).
        ValueError: If the saved index does not match the current corpus.
    """
    if not (index_dir / "embeddings.npy").exists():
        raise FileNotFoundError(f"No index in {index_dir}; run `uv run rag index` first.")
    chunks = load_corpus()
    if json.loads((index_dir / "ids.json").read_text()) != [c["id"] for c in chunks]:
        raise ValueError("Index does not match the corpus; delete it and run `rag index`.")
    return Index(
        chunks=chunks,
        bm25=bm25s.BM25.load(index_dir / "bm25"),
        embeddings=np.load(index_dir / "embeddings.npy"),
    )
