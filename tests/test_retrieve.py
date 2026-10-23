from rag.index import doc_text
from rag.retrieve import rrf


def test_rrf_rewards_agreement_between_rankers():
    assert rrf([[1, 2, 3], [4, 2, 5]])[0] == 2


def test_rrf_keeps_single_ranking_order():
    assert rrf([[5, 1, 9]]) == [5, 1, 9]


def test_doc_text_prepends_source_path():
    chunk = {"id": "langchain/docs/how_to/json.mdx_0_996", "text": "body"}
    assert doc_text(chunk) == "langchain/docs/how_to/json.mdx\nbody"
