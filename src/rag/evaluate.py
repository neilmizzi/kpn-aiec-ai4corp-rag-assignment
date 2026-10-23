"""Evaluation of retrieval, answer quality and citation support.

Retrieval is scored against the qrels (recall@k, nDCG), answer quality via nugget coverage, and
citation support by checking whether the cited chunks back the answer's claims.
"""


def main():
    """Run the evaluation on the held-out test split."""
    raise NotImplementedError
