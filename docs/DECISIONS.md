# Decisions and timeline

Short log of key decisions (why, and what we gave up) and progress. Newest last.

## Decisions

| # | Date | Decision | Why | Trade-off |
|---|------|----------|-----|-----------|
| 1 | 2026-10-23 | CLI only, no HTTP API | Assignment asks for an interactive CLI; extra layers earn no credit | No web/API access |
| 2 | 2026-10-23 | Python 3.11+, `uv`, `src/` layout | Fast, reproducible installs with a lockfile | Reviewers need `uv` |
| 3 | 2026-10-23 | Pre-commit hooks copied from `amanda-mab-models` (black, isort, flake8, pydocstyle, docformatter, gitleaks), hooks pinned to Python 3.12 | Team standard; docformatter's `untokenize` dep fails to build on 3.14 | Older hook versions |
| 4 | 2026-10-23 | Google-style docstrings (description, Args, Returns, Raises) enforced with `pydoclint` | Consistent, checkable documentation | More boilerplate on small functions |
| 5 | 2026-10-23 | LLM via KPN GenAI Gateway (acc), auth with `az login` token; `LLM_PROVIDER=openai` fallback | No API key or credits needed; OpenAI fallback keeps it reproducible outside KPN | Gateway only works for KPN accounts |
| 6 | 2026-10-23 | KPN private CA trusted via `truststore` (OS cert store), no cert file in repo | Cert files are blocked by content policy; OS store already trusts it | Relies on the OS trust store |
| 7 | 2026-10-23 | Gateway token from `az account get-access-token` instead of `azure-identity` | Ponytail audit: same result, ~15 fewer packages | Hand-rolled 5-min expiry cache |
| 8 | 2026-10-23 | Answer model `gpt-54-mini`, judge model `gpt-54` | Cheap/fast for answers; stronger model for grading | Judge cost scales with eval size |
| 9 | 2026-10-23 | Dev/test split 30/70 (61/142 queries), seeded (42), sorted by ID before shuffle | Assignment requires held-out test; keeps most queries for final eval | Small dev set for tuning |
| 10 | 2026-10-23 | Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion | Corpus is code + docs: BM25 catches identifiers, dense catches paraphrase | Two indexes to build and keep in sync |
| 11 | 2026-10-23 | Dense embeddings: local `BAAI/bge-small-en-v1.5` (CPU) | Free and reproducible; embedding 49.5k chunks (~50M tokens) via gateway costs ~$6.50 and 1-3 h | Truncates at 512 tokens (median chunk ~1k tokens); weaker than `text-embedding-3-large`; ~1 GB PyTorch install |

## Timeline

| Date | Time | Milestone |
|------|------|-----------|
| 2026-10-23 | 08:25 | Start: read assignment, chose stack |
| 2026-10-23 | 08:40 | KPN GenAI Gateway connected (chat + embeddings verified) |
| 2026-10-23 | 08:51 | Scaffold committed (`a633c74`) |
| 2026-10-23 | 08:55 | Docstring standard enforced (`b0107f5`) |
| 2026-10-23 | 10:55 | Python + AI/ML `.gitignore` (`ac58b85`) |
| 2026-10-23 | 11:39 | Data loading + dev/test split working (49,514 chunks, 203 queries) |
| 2026-10-23 | 11:40 | Chose local `bge-small` embeddings |
| 2026-10-23 | 11:50 | Benchmarked `bge-small` on CPU: 500 chunks in 27 s, ~45 min for the full corpus |
