import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag.answer_eval import main, validate_grade
from rag.cli import main as cli_main


def test_experiment_variants_cannot_be_combined():
    for first, second in (
        ("full_context_short", "evidence_first"),
        ("evidence_first", "language_aware"),
        ("full_context_short", "language_aware"),
    ):
        with pytest.raises(ValueError, match="Choose one experiment"):
            main(**{first: True, second: True})


@pytest.mark.parametrize(
    "args",
    [
        ["eval", "--full-context-short"],
        ["eval", "--evidence-first"],
        ["eval", "--answers", "--evidence-first", "--full-context-short"],
        ["eval", "--answers", "--evidence-first", "--split", "test"],
        ["eval", "--language-aware", "--split", "test"],
        ["eval", "--answers", "--language-aware", "--evidence-first"],
    ],
)
def test_cli_rejects_invalid_experiment_args(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["rag", *args])
    with pytest.raises(SystemExit) as error:
        cli_main()
    assert error.value.code == 2


def test_grade_validation():
    grade = {
        "covered_nuggets": [True, False],
        "citation_checks": [
            {
                "claim": "Python example",
                "citation_id": "js",
                "supported": False,
                "reason": "JS only",
            }
        ],
        "uncited_claims": [],
        "refused": False,
    }
    validate_grade(grade, 2)
    with pytest.raises(ValueError):
        validate_grade(grade, 3)
    with pytest.raises(ValueError):
        validate_grade({**grade, "covered_nuggets": [1, 0]}, 2)
    with pytest.raises(ValueError):
        validate_grade({**grade, "citation_checks": [{"supported": True}]}, 2)


@pytest.mark.parametrize("variant", ["full_context_short", "evidence_first", "language_aware"])
def test_test_split_rejects_experiments(variant):
    with pytest.raises(ValueError, match="dev-only"):
        main(split="test", **{variant: True})


def test_cli_dispatches_frozen_test_answers(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["rag", "eval", "--answers", "--split", "test"])
    monkeypatch.setattr("rag.answer_eval.main", lambda **kwargs: calls.append(kwargs))
    cli_main()
    assert calls == [
        {
            "full_context_short": False,
            "evidence_first": False,
            "language_aware": False,
            "split": "test",
        }
    ]


def test_structured_evaluation_records_current_variant_and_grades_display_only(
    monkeypatch, tmp_path
):
    import rag.answer_eval as evaluation
    from rag.config import TOP_K

    monkeypatch.chdir(tmp_path)
    query = {"id": "dev-1", "question": "Question?", "nuggets": ["Fact."]}
    sources = [{"id": "doc", "text": "Fact."}]
    monkeypatch.setattr(evaluation, "load_queries", lambda: [])
    monkeypatch.setattr(evaluation, "split_queries", lambda _: ([query], []))
    monkeypatch.setattr(evaluation, "CONTROLS", [])
    retrieval = Mock(return_value=sources)
    monkeypatch.setattr(evaluation, "retrieve", retrieval)
    generate = Mock(
        return_value={
            "answer": "Fact.\n[doc]",
            "citations": ["doc"],
            "abstained": False,
            "internal_metadata": "Do not send to judge",
        }
    )
    monkeypatch.setattr(evaluation, "answer", generate)
    grade = {
        "covered_nuggets": [True],
        "citation_checks": [],
        "uncited_claims": [],
        "refused": False,
    }
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(grade)))]
    )
    monkeypatch.setattr(evaluation, "get_client", lambda: client)
    evaluation.main()
    retrieval.assert_any_call("Question?", TOP_K, mode="hybrid")
    generate.assert_called_once_with("Question?", sources)
    settings_file = next((tmp_path / "results" / "raw").glob("*\\settings.json"))
    settings = json.loads(settings_file.read_text())
    assert settings["variant"] == "structured-claims-post-test"
    assert settings["verification_model"] is None
    assert settings["verification_prompt"] is None
    payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert payload == {
        "question": "Question?",
        "nuggets": ["Fact."],
        "sources": sources,
        "answer": "Fact.\n[doc]",
    }
    run_dir = settings_file.parent
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["dev_questions"] == 1
    assert summary["mean_nugget_coverage"] == 1.0
    assert json.loads((run_dir / "answers.jsonl").read_text())["sources"] == sources
    assert json.loads((run_dir / "grades.jsonl").read_text())["covered_nuggets"] == [True]
