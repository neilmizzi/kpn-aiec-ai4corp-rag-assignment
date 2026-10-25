"""Interactive command-line interface."""

import argparse


def main() -> None:
    """Parse command-line arguments and dispatch to the index, ask or eval command."""
    parser = argparse.ArgumentParser(prog="rag", description="Ask LangChain questions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="Download the corpus and build the search index")
    sub.add_parser("ask", help="Start an interactive Q&A session")
    evaluation = sub.add_parser("eval", help="Compare retrieval modes against relevance judgments")
    evaluation.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluation.add_argument(
        "--answers", action="store_true", help="Evaluate 13 dev cases or all held-out test answers"
    )
    variants = evaluation.add_mutually_exclusive_group()
    variants.add_argument(
        "--full-context-short",
        action="store_true",
        help="With --answers: compare full chunks and a 150-word answer instruction",
    )
    variants.add_argument(
        "--evidence-first",
        action="store_true",
        help="With --answers: test strict claim-level evidence and abstention rules",
    )
    variants.add_argument(
        "--language-aware",
        action="store_true",
        help="Dev only: compare language-aware retrieval, optionally with --answers",
    )
    args = parser.parse_args()
    if (
        args.command == "eval"
        and args.split == "test"
        and (args.full_context_short or args.evidence_first or args.language_aware)
    ):
        parser.error("Experimental variants are dev-only")
    if args.command == "eval" and args.full_context_short and not args.answers:
        parser.error("--full-context-short requires --answers")
    if args.command == "eval" and args.evidence_first and not args.answers:
        parser.error("--evidence-first requires --answers")
    if args.command == "eval" and args.language_aware and args.split != "dev":
        parser.error("--language-aware is a dev-only experiment")

    if args.command == "index":
        from rag.index import build_index

        build_index()
    elif args.command == "ask":
        repl()
    elif args.command == "eval":
        if args.answers:
            from rag.answer_eval import main as run_answers

            run_answers(
                full_context_short=args.full_context_short,
                evidence_first=args.evidence_first,
                language_aware=args.language_aware,
                split=args.split,
            )
        else:
            from rag.evaluate import main as run_eval

            run_eval(split=args.split, language_aware=args.language_aware)


def repl() -> None:
    """Run the interactive question-answer loop until an empty line, EOF or Ctrl+C."""
    from rag.config import TOP_K
    from rag.generate import GroundingError, answer
    from rag.retrieve import retrieve

    print("Loading index...")
    retrieve("warm up", 1)
    print("Ask a LangChain question (empty line or Ctrl+C to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        chunks = retrieve(question, TOP_K)
        try:
            result = answer(question, chunks)
        except GroundingError as exc:
            print(f"\nError: {exc} No answer was displayed.")
            continue
        print(f"\n{result['answer']}")
        urls = {c["id"]: c["url"] for c in chunks}
        if result["citations"]:
            print("\nSources:")
            for doc_id in result["citations"]:
                print(f"  [{doc_id}] {urls[doc_id]}")
