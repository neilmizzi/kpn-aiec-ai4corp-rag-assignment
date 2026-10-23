# LangChain RAG Assistant

Retrieval-augmented assistant that answers LangChain questions using the
[FreshStack](https://huggingface.co/freshstack) October 2024 LangChain corpus, cites supporting
chunks by document ID, and says so when the corpus does not contain the answer.

> Status: scaffold — implementation in progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```powershell
uv sync
uv run pre-commit install
```

Create a `.env` file in the repo root (never commit it) to override any default:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `kpn_gateway` | `kpn_gateway` (az login, no key) or `openai` (needs `OPENAI_API_KEY`) |
| `OPENAI_API_KEY` | — | Only for `LLM_PROVIDER=openai` |
| `EMBEDDING_MODEL` | `text-embedding-3-large-1` | Embedding model / deployment |
| `LLM_MODEL` | `gpt-54-mini-2026-03-17` | Answer generation model |
| `JUDGE_MODEL` | `gpt-54-2026-03-05` | LLM judge used in evaluation |
| `TOP_K` | `8` | Chunks retrieved per question |
| `DATA_DIR` | `data` | Local storage for corpus and indexes |

### KPN GenAI Gateway (default)

Requires the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) and GenAI
Gateway model access for your (BHR) account, requested via IAM. Then:

```powershell
az login --allow-no-subscriptions
$env:RUN_LIVE_TESTS = "1"; uv run pytest tests/test_llm_live.py   # connectivity check
```

KPN's private root CA is trusted through the Windows certificate store via `truststore`, so no
certificate file is needed.

### Outside KPN

Set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and public model names, e.g.
`EMBEDDING_MODEL=text-embedding-3-large` and `LLM_MODEL=gpt-4o-mini`.

Code checks (black, isort, flake8, pydocstyle, docformatter, gitleaks) run on commit via
pre-commit; run them manually with `uv run pre-commit run --all-files`.

## Usage

```powershell
uv run rag index   # download corpus + build indexes (one-off)
uv run rag ask     # interactive CLI
uv run rag eval    # evaluate on the held-out test split
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/rag/llm.py` | Chat/embedding client (KPN GenAI Gateway or OpenAI) |
| `src/rag/data.py` | Load corpus/queries from Hugging Face, dev/test split |
| `src/rag/index.py` | Build/persist BM25 + OpenAI embedding indexes |
| `src/rag/retrieve.py` | Hybrid retrieval (BM25 + dense, reciprocal rank fusion) |
| `src/rag/generate.py` | Grounded answers with doc-ID citations and abstention |
| `src/rag/evaluate.py` | Retrieval, answer-quality and citation-support metrics |
| `src/rag/cli.py` | Command-line entry point |

## Approach and key decisions

_TBD._

## Results

_TBD._

## Limitations and lessons learned

_TBD._
