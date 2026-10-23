"""Loading the FreshStack LangChain corpus and queries from Hugging Face."""

import random

from datasets import load_dataset

CORPUS_DATASET = "freshstack/corpus-oct-2024"
QUERIES_DATASET = "freshstack/queries-oct-2024"
SUBSET = "langchain"


def load_corpus() -> list[dict]:
    """Load the corpus chunks, the only material the assistant may search.

    Returns:
        Chunks as dicts with ``id`` (e.g. ``langchain/docs/x.mdx_0_996``), ``text`` and ``url``.
    """
    rows = load_dataset(CORPUS_DATASET, SUBSET, split="train")
    return [{"id": r["_id"], "text": r["text"], "url": r["metadata"]["url"]} for r in rows]


def load_queries() -> list[dict]:
    """Load the evaluation queries; never used as assistant input.

    Returns:
        Queries as dicts with ``id``, ``question`` (title + body), ``answer`` (accepted Stack
        Overflow answer), ``nuggets`` (key facts) and ``relevant_ids`` (union of the nuggets'
        relevant corpus chunk IDs).
    """
    rows = load_dataset(QUERIES_DATASET, SUBSET, split="test")
    return [
        {
            "id": r["query_id"],
            "question": f"{r['query_title']}\n\n{r['query_text']}",
            "answer": r["answer_text"],
            "nuggets": [n["text"] for n in r["nuggets"]],
            "relevant_ids": sorted({i for n in r["nuggets"] for i in n["relevant_corpus_ids"]}),
        }
        for r in rows
    ]


def split_queries(
    queries: list[dict], dev_fraction: float = 0.3, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split queries deterministically into a dev set and a held-out test set.

    Queries are sorted by ID before a seeded shuffle, so the split does not depend on load order.

    Args:
        queries: The full query set, each with an ``id``.
        dev_fraction: Fraction of queries used during development.
        seed: Random seed for a reproducible split.

    Returns:
        The dev and test query lists.

    Raises:
        ValueError: If ``dev_fraction`` is not strictly between 0 and 1.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError(f"dev_fraction must be between 0 and 1, got {dev_fraction}")
    shuffled = sorted(queries, key=lambda q: q["id"])
    random.Random(seed).shuffle(shuffled)
    n_dev = round(len(shuffled) * dev_fraction)
    return shuffled[:n_dev], shuffled[n_dev:]
