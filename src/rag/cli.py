"""Interactive command-line interface."""

import argparse


def main() -> None:
    """Parse command-line arguments and dispatch to the index, ask or eval command."""
    parser = argparse.ArgumentParser(prog="rag", description="Ask LangChain questions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="Download the corpus and build the search index")
    sub.add_parser("ask", help="Start an interactive Q&A session")
    sub.add_parser("eval", help="Run the evaluation on the held-out test split")
    args = parser.parse_args()

    if args.command == "index":
        from rag.index import build_index

        build_index()
    elif args.command == "ask":
        repl()
    elif args.command == "eval":
        from rag.evaluate import main as run_eval

        run_eval()


def repl() -> None:
    """Run the interactive question-answer loop until an empty line, EOF or Ctrl+C."""
    from rag.config import TOP_K
    from rag.generate import answer
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
        result = answer(question, chunks)
        print(f"\n{result['answer']}")
        urls = {c["id"]: c["url"] for c in chunks}
        if result["citations"]:
            print("\nSources:")
            for doc_id in result["citations"]:
                print(f"  [{doc_id}] {urls[doc_id]}")
