# Decisions and timeline

Short log of key decisions (why, and what we gave up) and progress. Newest last.

## Observed failures

### F01: Python PDF example cites a TypeScript-only chunk

- **Observed:** 2026-10-23, during a manual CLI demo (not a dev/test evaluation question).
- **Question:** "How do I load a PDF in LangChain using Python?"
- **Answer excerpt:** Python code importing `PyPDFLoader` from
  `langchain_community.document_loaders`, constructing it with a PDF path, and calling `load()`.
- **Citation attached to that example:**
  `langchainjs/docs/core_docs/docs/how_to/document_loader_pdf.mdx_0_2919`.
- **Verification:** Read the exact chunk from the downloaded FreshStack corpus on 2026-10-23
  at 16:17. It documents TypeScript `PDFLoader` and `WebPDFLoader`, with no Python or
  `PyPDFLoader` example.
- **Finding:** Confirmed citation-support error. The cited ID exists and was retrieved, but
  that chunk does not support the Python code attributed to it. This does not establish
  whether the Python code itself is incorrect.
- **Follow-up verification (16:41):** The user printed
  `langchain/docs/docs/how_to/document_loader_pdf.ipynb_0_7516` from the corpus. It shows the
  exact `PyPDFLoader` import, construction with `file_path`, and explicitly describes one
  document per page. Its example uses `alazy_load()`, not `load()`. This supports the loader
  choice and page behavior, but does not directly demonstrate the answer's synchronous
  `loader.load()` call.
- **Status:** Logged, not fixed. The Python guide provides partial support; the TypeScript
  citation attached to the code example remains incorrect. Other cited chunks have not
  yet been checked.
- **Lesson:** Checking citation IDs against retrieved IDs is not evidence that the cited
  text supports the answer's claim. Citation-support evaluation must inspect the content.

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
| 12 | 2026-10-23 | Index contains corpus chunks only, keyed by corpus doc ID; queries/answers/nuggets never indexed | Assignment: corpus is the only searchable material; IDs are needed for citations | None |
| 13 | 2026-10-23 | Index the source path + chunk text (e.g. `langchain/docs/how_to/document_loader_json.mdx`) | Path is part of the corpus and names the topic/API; helps both BM25 and dense | Slightly longer texts |
| 14 | 2026-10-23 | `bm25s` instead of `rank-bm25`; English stopwords, no stemming | Sparse matrices: builds in 16 s, fast queries, saves to disk | Default tokenizer keeps `snake_case`/`CamelCase` identifiers whole (good for exact API names, misses sub-word matches) |
| 15 | 2026-10-23 | RRF fusion (k=60) of top-100 from each ranker; exact dense search with NumPy (no vector DB) | RRF needs no score calibration; 49.5k x 384 floats = 76 MB, brute force is milliseconds | No ANN; would not scale to millions of chunks |
| 16 | 2026-10-23 | Embeddings built in 2,000-chunk shards that resume on restart | 45-min CPU job should survive interruption | A few extra lines |
| 17 | 2026-10-23 | Prompt: answer only from numbered sources, cite `[doc_id]` inline, fixed abstention sentence; chunks capped at 4,000 chars, top 8 | Citations map 1:1 to corpus IDs; cap keeps prompt ~8k tokens | Long chunks lose their tail |
| 18 | 2026-10-23 | Cited IDs are validated against the retrieved set; invented IDs are dropped | Prevents hallucinated citations reaching the user | A real but mistyped ID is lost |
| 19 | 2026-10-23 | Abstain only when sources are unrelated; for debugging questions, explain what the sources indicate. Abstention detected by regex on the first line + no citations | First prompt over-abstained on a dev question with 4 relevant chunks retrieved; the model also paraphrased the fixed sentence | More partial answers; risk of answering from loosely related sources when retrieval misses |
| 20 | 2026-10-23 | Keep hybrid as the current baseline after comparing all three modes on 61 dev questions | Hybrid leads in recall@8 (0.170), hit@8 (0.705) and nDCG@8 (0.255); see README for all results | Still no relevant chunk in the top 8 for about 30% of questions; comparison is not proof of optimality |

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
| 2026-10-23 | 13:48 | BM25 index built (16 s). Dev baseline, BM25 only: recall@10 0.171, recall@100 0.371, any-hit@10 67% |
| 2026-10-23 | 13:48 | Full embedding build started |
| 2026-10-23 | 14:03 | Laptop entered standby; embedding paused until 14:48 (shards resumed fine) |
| 2026-10-23 | 15:01 | Answer generation with citations + abstention working (tried on 5 dev questions with BM25) |
| 2026-10-23 | 15:16 | Full embedding index finished, including the standby interruption |
| 2026-10-23 | 15:28 | User verified the hybrid CLI end to end; Python answer cited a JS document, motivating citation-support evaluation |
| 2026-10-23 | After 15:32 | Compared BM25, dense and hybrid on all 61 dev questions; no test questions evaluated and no LLM calls made |
| 2026-10-23 | 16:18 | Logged F01: verified that a TypeScript-only chunk was incorrectly cited for a Python PDF-loading example |
| 2026-10-23 | 16:41 | F01 follow-up: manually checked the Python guide; import, construction and page behavior supported, but its example uses `alazy_load()` rather than `load()` |
