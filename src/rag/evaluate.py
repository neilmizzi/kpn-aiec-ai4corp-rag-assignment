"""Retrieval comparison against corpus relevance judgments, without generating answers."""

import json
import math
import time
from datetime import datetime
from pathlib import Path

from rag.data import load_queries, split_queries
from rag.retrieve import retrieve


def retrieval_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    """Score a ranked result list using binary relevance judgments.

    Args:
        ranked_ids: Retrieved document IDs in ranking order, without duplicates.
        relevant_ids: Corpus document IDs judged relevant to the query.
        k: Rank cutoff.

    Returns:
        Recall, hit rate and nDCG at the requested cutoff.

    Raises:
        ValueError: If the cutoff is not positive or no relevant IDs are supplied.
    """
    if k < 1 or not relevant_ids:
        raise ValueError("Metrics require a positive cutoff and at least one relevant document.")
    hits = [doc_id in relevant_ids for doc_id in ranked_ids[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant_ids))))
    return {
        f"recall@{k}": sum(hits) / len(relevant_ids),
        f"hit@{k}": float(any(hits)),
        f"ndcg@{k}": dcg / ideal,
    }


def main(split: str = "dev", language_aware: bool = False) -> None:
    """Compare retrieval modes and save per-query measurements for reproduction.

    Model/index startup is excluded from timing by warming up before measuring. Binary relevance
    is the union of each query's nugget-level relevant IDs. Unjudged chunks count as non-relevant.

    Args:
        split: ``dev`` for tuning, or ``test`` for the final held-out evaluation.
        language_aware: Compare baseline hybrid and experimental language-aware hybrid on dev.

    Raises:
        ValueError: If the split is unknown or the experiment requests held-out data.
    """
    if split not in {"dev", "test"}:
        raise ValueError(f"Unknown evaluation split: {split!r}")
    if language_aware and split != "dev":
        raise ValueError("Language-aware retrieval is a dev-only experiment.")
    dev, test = split_queries(load_queries())
    queries = dev if split == "dev" else test
    print(f"Retrieval evaluation: {split}, {len(queries)} queries. Loading index...", flush=True)
    retrieve("warm up", 1)
    output = Path("results") / f"retrieval-{split}.jsonl"
    if language_aware:
        output = (
            Path("results")
            / "raw"
            / datetime.now().strftime("retrieval-language-aware-%Y%m%d-%H%M%S.jsonl")
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    print("mode       recall@8  hit@8  ndcg@8  recall@20  seconds/query", flush=True)
    with output.open("w", encoding="utf-8") as stream:
        for mode in (
            ("hybrid", "hybrid-language") if language_aware else ("bm25", "dense", "hybrid")
        ):
            measurements = []
            for query in queries:
                start = time.perf_counter()
                ranked_ids = [c["id"] for c in retrieve(query["question"], 100, mode=mode)]
                elapsed = time.perf_counter() - start
                metrics = {}
                for k in (8, 10, 20, 100):
                    metrics.update(retrieval_metrics(ranked_ids, set(query["relevant_ids"]), k))
                metrics["seconds"] = elapsed
                stream.write(
                    json.dumps(
                        {
                            "query_id": query["id"],
                            "split": split,
                            "mode": mode,
                            "retrieved_ids": ranked_ids,
                            "metrics": metrics,
                        }
                    )
                    + "\n"
                )
                measurements.append(metrics)
            means = {
                key: sum(row[key] for row in measurements) / len(measurements)
                for key in measurements[0]
            }
            print(
                f"{mode:<10} {means['recall@8']:.3f}     {means['hit@8']:.3f}  "
                f"{means['ndcg@8']:.3f}   {means['recall@20']:.3f}      "
                f"{means['seconds']:.3f}",
                flush=True,
            )
    print(f"Per-query results saved to {output}")
