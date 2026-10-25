import json
from unittest.mock import Mock

import pytest

from rag.generate import ABSTAIN, GroundingError, answer

IDS = ["langchain/docs/a.mdx_0_10", "langchainjs/src/b.ts_5_99"]
CHUNKS = [
    {"id": IDS[0], "text": "x" * 6000, "url": "https://example.org/first"},
    {"id": IDS[1], "text": "Other fact.", "url": "https://example.org/second"},
    {"id": "unused.py_0_10", "text": "Unused fact.", "url": "https://example.org/unused"},
]


def test_answer_abstains_without_chunks(monkeypatch):
    completion = Mock()
    monkeypatch.setattr("rag.generate._json_completion", completion)
    assert answer("anything?", []) == {"answer": ABSTAIN, "citations": [], "abstained": True}
    completion.assert_not_called()


@pytest.mark.parametrize(
    "draft",
    [
        {},
        {"claims": [], "incomplete": "true"},
        {"claims": [], "incomplete": 1},
        {"claims": {}, "incomplete": True},
        {"claims": [], "incomplete": True, "explanation": "extra"},
        {"claims": [], "incomplete": False},
    ],
)
def test_invalid_top_level_schema_fails_closed(monkeypatch, draft):
    monkeypatch.setattr("rag.generate._json_completion", Mock(return_value=draft))
    with pytest.raises(GroundingError):
        answer("question", CHUNKS)


@pytest.mark.parametrize(
    "claim",
    [
        None,
        {},
        {"text": "Claim.", "source": "S9"},
        {"text": "Claim.", "source": IDS[0]},
        {"text": "Claim.", "source": ["S1", "S2"]},
        {"text": "Claim.", "source": None},
        {"text": "", "source": "S1"},
        {"text": " \n ", "source": "S1"},
        {"text": 1, "source": "S1"},
        {"text": "Claim.", "source": "S1", "other": True},
        {"text": "Claim.", "citations": ["S1"]},
    ],
)
def test_invalid_claim_fails_whole_answer_not_just_one_claim(monkeypatch, claim):
    completion = Mock(
        return_value={
            "claims": [{"text": "Valid claim.", "source": "S1"}, claim],
            "incomplete": False,
        }
    )
    monkeypatch.setattr("rag.generate._json_completion", completion)
    with pytest.raises(GroundingError, match="one valid source label"):
        answer("question", CHUNKS)
    assert completion.call_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "Claim [S1].",
        "Claim [S1, S2].",
        f"Claim [{IDS[0]}].",
        "Claim [invented/path.py_0_9].",
        f"The document is {IDS[0]}.",
        "Read [this](https://example.org).",
    ],
)
def test_generated_citation_markup_is_rejected(monkeypatch, text):
    monkeypatch.setattr(
        "rag.generate._json_completion",
        Mock(return_value={"claims": [{"text": text, "source": "S1"}], "incomplete": False}),
    )
    with pytest.raises(GroundingError, match="inline citation or link"):
        answer("question", CHUNKS)


def test_code_paths_and_indexing_are_not_mistaken_for_citations(monkeypatch):
    text = '```python\npaths = ["docs/report.pdf"]\nprint(paths[0], matrix[0][1])\n```'
    monkeypatch.setattr(
        "rag.generate._json_completion",
        Mock(return_value={"claims": [{"text": text, "source": "S1"}], "incomplete": False}),
    )
    result = answer("question", CHUNKS)
    assert result["answer"] == f"{text}\n[{IDS[0]}]"


@pytest.mark.parametrize(
    "chunks",
    [
        [{"id": IDS[0], "text": ""}],
        [{"id": IDS[0], "text": None}],
        [{"id": "", "text": "Evidence."}],
        [{"text": "Evidence."}],
        [{"id": IDS[0], "text": "A"}, {"id": IDS[0], "text": "B"}],
    ],
)
def test_invalid_source_records_fail_before_model_call(monkeypatch, chunks):
    completion = Mock()
    monkeypatch.setattr("rag.generate._json_completion", completion)
    with pytest.raises(GroundingError, match="unique IDs and nonempty text"):
        answer("question", chunks)
    completion.assert_not_called()


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [
        pytest.param(None, None, id="empty-choices"),
        ("{broken", "stop"),
        ("[]", "stop"),
        (None, "stop"),
        ('{"claims":[]}', "length"),
        ('{"claims":[]}', "content_filter"),
        ('{"incomplete":false,"incomplete":true}', "stop"),
        ('{"claims":[{"text":"A","text":"B","source":"S1"}],"incomplete":false}', "stop"),
    ],
)
def test_malformed_completion_is_explicit_failure(monkeypatch, content, finish_reason):
    from rag.generate import _json_completion

    client = Mock()
    choice = Mock(finish_reason=finish_reason)
    choice.message.content = content
    client.chat.completions.create.return_value.choices = [] if finish_reason is None else [choice]
    monkeypatch.setattr("rag.generate.get_client", lambda: client)
    with pytest.raises(GroundingError):
        _json_completion([])


def test_model_errors_propagate_without_application_fallback(monkeypatch):
    completion = Mock(side_effect=RuntimeError("gateway failed"))
    monkeypatch.setattr("rag.generate._json_completion", completion)
    with pytest.raises(RuntimeError, match="gateway failed"):
        answer("question", CHUNKS)
    assert completion.call_count == 1


def test_empty_draft_explicitly_abstains(monkeypatch):
    completion = Mock(return_value={"claims": [], "incomplete": True})
    monkeypatch.setattr("rag.generate._json_completion", completion)
    result = answer("question", CHUNKS)
    assert result == {"answer": ABSTAIN, "citations": [], "abstained": True}
    assert completion.call_count == 1


def test_partial_answer_uses_fixed_warning_without_claiming_verification(monkeypatch):
    completion = Mock(
        return_value={
            "claims": [{"text": "Source-backed API.", "source": "S1"}],
            "incomplete": True,
        }
    )
    monkeypatch.setattr("rag.generate._json_completion", completion)
    result = answer("question", CHUNKS)
    assert "Partial answer: the model reports" in result["answer"]
    assert "verified" not in result["answer"]
    assert not result["abstained"]
    assert result["citations"] == [IDS[0]]


def test_cli_reports_structural_failure_and_keeps_prompting(monkeypatch, capsys):
    from rag.cli import repl

    inputs = iter(["question", ""])
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    monkeypatch.setattr("rag.retrieve.retrieve", lambda *a: CHUNKS)
    monkeypatch.setattr("rag.generate.answer", Mock(side_effect=GroundingError("Bad structure.")))
    repl()
    output = capsys.readouterr().out
    assert "Error: Bad structure." in output
    assert "No answer was displayed." in output


def test_cli_parses_sdk_response_and_displays_only_cited_sources(monkeypatch, capsys):
    from rag.cli import main
    from rag.config import LLM_MODEL, TOP_K

    inputs = iter(["question", ""])
    monkeypatch.setattr("sys.argv", ["rag", "ask"])
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))
    retrieval = Mock(return_value=CHUNKS)
    monkeypatch.setattr("rag.retrieve.retrieve", retrieval)
    client = Mock()
    choice = Mock(finish_reason="stop")
    choice.message.content = json.dumps(
        {
            "claims": [
                {"text": "Second fact.", "source": "S2"},
                {"text": "First fact.", "source": "S1"},
                {"text": "Another second fact.", "source": "S2"},
            ],
            "incomplete": False,
        }
    )
    client.chat.completions.create.return_value.choices = [choice]
    monkeypatch.setattr("rag.generate.get_client", lambda: client)
    main()
    retrieval.assert_any_call("question", TOP_K)
    client.chat.completions.create.assert_called_once()
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == LLM_MODEL
    assert request["response_format"] == {"type": "json_object"}
    assert "json" in request["messages"][0]["content"].lower()
    assert json.loads(request["messages"][1]["content"]) == {
        "question": "question",
        "sources": {"S1": "x" * 6000, "S2": "Other fact.", "S3": "Unused fact."},
    }
    output = capsys.readouterr().out
    assert (
        f"Second fact.\n[{IDS[1]}]\n\nFirst fact.\n[{IDS[0]}]"
        f"\n\nAnother second fact.\n[{IDS[1]}]" in output
    )
    assert output.count(f"[{IDS[1]}] https://example.org/second") == 1
    assert output.index("https://example.org/second") < output.index("https://example.org/first")
    assert "unused.py_0_10" not in output
    assert "Partial answer" not in output
