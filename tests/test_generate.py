from rag.generate import ABSTAIN, answer, extract_citations, format_context

IDS = ["langchain/docs/a.mdx_0_10", "langchainjs/src/b.ts_5_99"]


def test_extract_citations_keeps_only_retrieved_ids_in_order():
    text = f"X [{IDS[1]}]. Y [{IDS[0]}; made/up.py_0_1]. Z [{IDS[1]}]."
    assert extract_citations(text, IDS) == [IDS[1], IDS[0]]


def test_format_context_labels_and_truncates():
    ctx = format_context([{"id": IDS[0], "text": "x" * 10_000}])
    assert ctx.startswith(f"### Source 1: [{IDS[0]}]")
    assert len(ctx) < 4100


def test_answer_abstains_without_chunks():
    assert answer("anything?", []) == {"answer": ABSTAIN, "citations": [], "abstained": True}
