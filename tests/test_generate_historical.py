"""Checks for the free-text generator retained for historical experiments."""

import pytest

from rag.generate import baseline_answer, extract_citations, format_context

IDS = ["langchain/docs/a.mdx_0_10", "langchainjs/src/b.ts_5_99"]


def test_extract_citations_keeps_only_retrieved_ids_in_order():
    text = f"X [{IDS[1]}]. Y [{IDS[0]}; made/up.py_0_1]. Z [{IDS[1]}]."
    assert extract_citations(text, IDS) == [IDS[1], IDS[0]]


@pytest.mark.parametrize("limit", [None, 4000])
def test_format_context_labels_and_respects_explicit_limit(limit):
    ctx = format_context([{"id": IDS[0], "text": "x" * 10_000}], max_chunk_chars=limit)
    assert ctx.startswith(f"### Source 1: [{IDS[0]}]")
    assert ctx.split("\n", 1)[1] == "x" * (limit or 10_000)


def test_experiment_context_and_instruction_do_not_change_defaults(monkeypatch):
    messages = []

    def fake_chat(request):
        messages.append(request)
        return "A supported answer."

    monkeypatch.setattr("rag.generate.chat", fake_chat)
    chunks = [{"id": IDS[0], "text": "x" * 6000}]
    baseline_answer("question", chunks, max_chunk_chars=4000, extra_instruction="\nBe brief.")
    baseline_answer("question", chunks)
    assert "x" * 4001 not in messages[0][1]["content"]
    assert messages[0][0]["content"].endswith("Be brief.")
    assert "x" * 6000 in messages[1][1]["content"]
    assert not messages[1][0]["content"].endswith("Be brief.")
