from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rag.retrieve import chunk_language, question_language, retrieve, rrf


def test_rrf_rewards_agreement_between_rankers():
    assert rrf([[1, 2, 3], [4, 2, 5]])[0] == 2


def test_rrf_keeps_single_ranking_order():
    assert rrf([[5, 1, 9]]) == [5, 1, 9]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("bm25", [0, 1]), ("dense", [1, 2]), ("hybrid", [1, 0])],
)
def test_retrieval_modes_fuse_disagreeing_rankings_and_limit_results(monkeypatch, mode, expected):
    chunks = [{"id": str(i), "text": f"Chunk {i}", "url": f"url-{i}"} for i in range(3)]
    sparse = Mock(return_value=(np.array([[0, 1, 2]]), None))
    embedder = Mock(encode=Mock(return_value=np.array([1])))
    index = SimpleNamespace(
        chunks=chunks,
        bm25=SimpleNamespace(retrieve=sparse),
        embeddings=np.array([[0], [3], [2]]),
    )
    monkeypatch.setattr("rag.retrieve._resources", lambda: (index, embedder))
    assert retrieve("question", 2, mode) == [chunks[i] for i in expected]
    assert sparse.call_count == (mode != "dense")
    if mode == "bm25":
        embedder.encode.assert_not_called()
    else:
        embedder.encode.assert_called_once_with(
            "Represent this sentence for searching relevant passages: question",
            normalize_embeddings=True,
        )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Load a PDF in Python", "python"),
        ("Use TypeScript", "javascript"),
        ("JavaScript or JS", "javascript"),
        ("Compare Python and JavaScript", None),
        ("Load a PDF", None),
        ("What is a timestamp?", None),
    ],
)
def test_question_language(question, expected):
    assert question_language(question) == expected


def test_chunk_language():
    assert chunk_language("langchainjs/docs/pdf.mdx_0_12") == "javascript"
    assert chunk_language("langchain/docs/pdf.ipynb_0_12") == "python"
    assert chunk_language("other/src/pdf.ts_0_12") == "javascript"
    assert chunk_language("other/docs/pdf.md_0_12") is None


def test_language_retrieval_demotes_mismatch_only(monkeypatch):
    chunks = [
        {"id": "langchainjs/docs/pdf.mdx_0_12"},
        {"id": "other/docs/pdf.md_0_12"},
        {"id": "langchain/docs/pdf.ipynb_0_12"},
        {"id": "other/src/pdf.ts_0_12"},
    ]
    index = SimpleNamespace(
        chunks=chunks,
        bm25=Mock(retrieve=Mock(return_value=(np.array([[0, 1, 2, 3]]), None))),
        embeddings=np.array([[4], [3], [2], [1]]),
    )
    embedder = Mock(encode=Mock(return_value=np.array([1])))
    monkeypatch.setattr("rag.retrieve._resources", lambda: (index, embedder))
    assert retrieve("PDF in Python", 4) == chunks
    assert retrieve("PDF in Python", 4, "hybrid-language") == [
        chunks[1],
        chunks[2],
        chunks[0],
        chunks[3],
    ]
    assert retrieve("PDF in JavaScript", 4, "hybrid-language") == [
        chunks[0],
        chunks[1],
        chunks[3],
        chunks[2],
    ]
    for question in ("Load PDF", "Python and TypeScript"):
        assert retrieve(question, 4, "hybrid-language") == chunks
