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
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers embedding model |
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
uv run rag eval --split dev   # compare BM25, dense and hybrid retrieval during development
uv run rag eval --split test  # reserve for final evaluation after choices are frozen
```

Retrieval evaluation reports mean per-question recall, hit rate and binary nDCG at ranks
8, 10, 20 and 100. Relevant documents are the union of nugget-level relevance judgments;
unjudged documents count as non-relevant. Per-query results are written to
`results/retrieval-dev.jsonl` (or `retrieval-test.jsonl`). No LLM calls are made.
Startup is excluded from query timings. Answer quality and citation support are not yet evaluated.

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

See [docs/DECISIONS.md](docs/DECISIONS.md) for the decision log and timeline.

## Results

### Retrieval baseline (development only)

Measured on all 61 dev questions using the existing corpus index. Cutoff 8 matches the number
of chunks passed to answer generation. Values are macro-averages over questions.

| Mode | Recall@8 | Hit@8 | nDCG@8 | Recall@20 | Seconds/query |
|------|----------|-------|--------|-----------|---------------|
| BM25 | 0.152 | 0.672 | 0.226 | 0.218 | 0.003 |
| Dense (`bge-small`) | 0.163 | 0.607 | 0.215 | 0.235 | 0.056 |
| Hybrid (RRF) | 0.170 | 0.705 | 0.255 | 0.261 | 0.058 |

Hybrid is the best of these three baselines on this dev set, but misses all judged-relevant
chunks in the top 8 for about 30% of questions. This supports keeping hybrid as the current
baseline, not a claim that it is optimal. Timings are from one warm run on the development
laptop, excluding model/index startup and answer generation. The 142 test questions remain
held out; answer quality and citation support have not yet been measured.

## Limitations and lessons learned

_TBD._
