from unittest.mock import Mock

import bm25s
import numpy as np
import pytest

from rag import data, index


def test_index_uses_only_corpus_and_resumes_without_reembedding(monkeypatch, tmp_path):
    rows = [
        {"_id": "docs/alpha.py_0_10", "text": "alphaterm", "metadata": {"url": "alpha"}},
        {"_id": "docs/beta.py_0_10", "text": "betaterm", "metadata": {"url": "beta"}},
    ]

    def corpus_only(dataset, subset, *, split):
        assert (dataset, subset, split) == (data.CORPUS_DATASET, "langchain", "train")
        return rows

    monkeypatch.setattr(data, "load_dataset", corpus_only)
    monkeypatch.setattr(index, "SHARD_SIZE", 1)
    embedder = Mock()
    embedder.encode.side_effect = [np.array([[1, 0]]), RuntimeError("Interrupted")]
    load_embedder = Mock(return_value=embedder)
    monkeypatch.setattr(index, "load_embedder", load_embedder)

    with pytest.raises(FileNotFoundError, match="rag index"):
        index.load_index(tmp_path)
    with pytest.raises(RuntimeError, match="Interrupted"):
        index.build_index(tmp_path)
    assert (tmp_path / "parts" / "0000.npy").exists()
    assert not (tmp_path / "embeddings.npy").exists()
    embedder.encode.assert_any_call(
        ["docs/alpha.py\nalphaterm"], batch_size=32, normalize_embeddings=True
    )

    embedder.encode.reset_mock()
    embedder.encode.side_effect = [np.array([[0, 1]])]
    index.build_index(tmp_path)
    embedder.encode.assert_called_once_with(
        ["docs/beta.py\nbetaterm"], batch_size=32, normalize_embeddings=True
    )
    loaded = index.load_index(tmp_path)
    assert loaded.chunks == [
        {"id": row["_id"], "text": row["text"], "url": row["metadata"]["url"]} for row in rows
    ]
    np.testing.assert_array_equal(loaded.embeddings, np.eye(2, dtype=np.float32))
    tokens = bm25s.tokenize(["betaterm"], stopwords="en", show_progress=False)
    ranked, _ = loaded.bm25.retrieve(tokens, k=2, show_progress=False)
    assert loaded.chunks[ranked[0][0]]["id"] == rows[1]["_id"]

    load_embedder.reset_mock()
    index.build_index(tmp_path)
    load_embedder.assert_not_called()
    rows.reverse()
    with pytest.raises(ValueError, match="does not match the corpus"):
        index.load_index(tmp_path)
