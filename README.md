# LangChain RAG Assistant

A CLI answering LangChain questions from the FreshStack October 2024 corpus, with citations.

**Status:** working PoC. Better citation support does not mean better answers.

**Overall concern: partial requirement satisfaction.** The system produces cited answers,
but does not consistently deliver complete, source-supported solutions. The retained setup
improved judged citation support while reducing answer coverage. We accept this limitation;
we have not resolved it or demonstrated an overall improvement.
Further investigation is needed to distinguish architectural limitations from model,
tokenization, retrieval or tooling limitations; the experiments do not isolate the root cause.

## 1. Run it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.
The default KPN provider also requires Azure CLI and approved gateway access.
Verified on Windows/Python 3.14.6.

```powershell
uv sync --locked
az login --allow-no-subscriptions
uv run rag index
uv run rag ask
```

**First run:** indexing downloads the corpus/model; the original embedding build took ~45 minutes.
Interrupted embedding shards resume. **Exit:** submit an empty question.

**Optional configuration:** use `.env`; never commit secrets.

| Setting | Default |
|---------|---------|
| `LLM_PROVIDER` | `kpn_gateway`; alternatively `openai` with `OPENAI_API_KEY` |
| `LLM_MODEL` / `JUDGE_MODEL` | `gpt-54-mini-2026-03-17` / `gpt-54-2026-03-05` |
| `EMBEDDING_MODEL` | Local `BAAI/bge-small-en-v1.5` |
| `TOP_K` / `DATA_DIR` | `8` / `data` |

**OpenAI alternative:** set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and available
`LLM_MODEL` / `JUDGE_MODEL` names in `.env`; skip Azure CLI installation and `az login`.
Changing embeddings requires a fresh index.

## 2. How it works

```text
49,514 corpus chunks
  -> BM25 + dense search -> rank fusion -> eight full chunks
  -> one structured LLM call -> Python-rendered citations
```

**Data design:** index the corpus; develop on 61 dev queries; evaluate on 142 test queries.
No model training or fine-tuning. Questions enter retrieval/generation; reference answers,
expected facts and relevance labels are used only for scoring, never as answer evidence.

**Boundaries:** corpus-only search; no live web search. The loader selects only the supplied
corpus; retrieval searches its local index, with no web-search tool.
`tests\test_index.py` guards the dataset input and fails
if indexing requests anything other than the corpus. Initial data/model downloads are setup,
not live search. This verifies the search boundary, **not** that every generated statement
comes from the retrieved evidence; the model can still introduce unsupported knowledge.

**Checks:** reject malformed output/unknown source labels; empty claims abstain;
partial answers get a warning. Python validates structure, **not factual support**.
No runtime semantic verifier: in E08, drafting plus verification improved judged support
on eight successful paired cases (69% -> 94%) but reduced fact coverage (50% -> 30%).
Across all ten dev requests, two failed and mean latency roughly doubled (4.7 -> 9.2 s).
The verifier also made semantic mistakes. We kept deterministic checks and offline judging
rather than add a second fallible model to each answer.

### Why these choices

| Choice | Reason |
|--------|--------|
| Use resources at hand | Existing KPN GenAI Gateway for generation/judging; local laptop for embeddings, indexing and retrieval. No new hosted infrastructure for this prototype. |
| Python/uv CLI | Matches the interactive assignment; locked dependencies, no extra API layer |
| BM25 + local BGE + RRF | Combine identifier matching and semantic search without score calibration |
| Exact NumPy search | Compare each query against all 49,514 vectors (~76 MB): practical locally, no approximate index or database service. Scale/simplicity choice; alternatives were not benchmarked. |
| Mini generator, stronger offline judge | Keep interactive inference lightweight; separate generation from assessment |
| Structural checks, no semantic verifier | Deterministic source handling; runtime verifier added failures/latency without a satisfactory quality trade-off |

## 3. Final results

61 dev / 142 test questions, deterministic seed-42 split. Current answers were evaluated
after test exposure: **post-test evidence, not a fresh held-out result**.

**Original baseline:** free-text answers; the model writes citations, and Python extracts
recognized IDs. No enforced claim schema.

**Current (retained):** structured claims with known source labels; Python validates the
schema and renders citations. No runtime semantic verifier.

Both use the same mini generator, hybrid retrieval and eight full chunks.

| Measurement | Original: free text | Current: structured claims |
|-------------|-------------------|---------|
| Answers / attempts | 142/142 | 141/142 |
| Accepted grades | 141 | 141 |
| Mean expected-fact coverage | 33.9% | 16.4% |
| Judge-supported citation pairs | 1,232/1,692 (72.8%) | 938/1,018 (92.1%) |
| Answers covering no expected facts | 61 | 94 |

**Takeaway:** higher judged citation-support rate, less complete answers—not an overall improvement.
Both runs used the same retrieved evidence and graded question IDs.

**Retrieval:** at least one judged-relevant top-eight chunk for 64.8% of questions.
**Speed:** 2.21 seconds average warmed retrieval + generation.

**Method:** check whether search finds the relevant documents and puts them first.
A separate model checks whether answers include the expected facts and whether their
cited sources support what they say.

**Examples from the current run:**

| Case | What the answer did | Assessment |
|------|---------------------|------------|
| `77644218` (Pinecone) | Cited the recommendation to pass an Embeddings object instead of a callable | Useful supported content, but only 1/2 expected facts; not a fully successful answer |
| `76288582` (streaming) | Listed LlamaIndex streaming facts without the requested FastAPI solution | Poor task resolution; 0/3 expected facts |

**Manual development-answer reviews** (earlier saved answers, not the final test run):

| Case | Reviewer assessment |
|------|-----------------|
| `76244938` (StructuredTool) | Useful structured-chat direction, but universal agent compatibility and toolkit typing unresolved |
| `76451997` (token limit) | Useful `max_tokens_limit` guidance; clearer application preferred, not a confirmed factual failure |
| `76779817` (Whisper) | Related facts instead of diagnosis; missing traceback/version clarification |
| `76111075` (Chroma) | Did not directly answer what was needed beyond npm installs |
| `77465301` (Azure Search) | Mixed Python vector-store, retriever and JavaScript evidence without establishing compatibility |

These qualitative reviews assess usefulness, not new scores.
[All reviewed cases and evidence](docs/DEV_EVIDENCE_REVIEW.md).

## 4. Requirement status

| Requirement | Status |
|-------------|--------|
| Search the supplied corpus; interactive CLI | Implemented; no live web search |
| Answer questions with supporting citations | Partial: structure enforced, complete source-supported solutions inconsistent |
| State insufficient evidence instead of guessing | Partial: abstention/warnings implemented, sufficiency decisions unreliable |
| Explain choices and separate dev/test questions | Documented; original held-out and later post-test results distinguished |
| Report results, examples and trade-offs | Above; detailed evidence linked below, monetary cost unmeasured |
| Reproducible repository | Clean-copy/wheel checks passed; user confirmed the public-repository clone runs; compact evaluation records included |

## 5. Limitations and future work

| Limit | Future work |
|-------|-------------|
| Complete, source-supported solutions are inconsistent; the failure may be in retrieval, evidence attribution or answer generation. | Review a small sample to locate failures, then test targeted changes: embeddings/retrieval for missing evidence, evidence selection for wrong citations, or generator/prompt/architecture changes when available evidence is not applied. Measure completeness and support together. |
| None of the 141 generated answers fully declined to answer. Related facts can replace a solution; we have not established when to report insufficient evidence. | Review cases with sufficient, partial and missing evidence; assess whether to answer, qualify or decline. Test an explicit evidence-sufficiency stage against the current architecture, including added errors and latency. |
| Prototype, not production-hardened: stronger prompt-injection defenses and broader LLM/RAG security controls were not evaluated. Prompt instructions and citation checks do not establish injection resistance. | Assess applicable [OWASP LLM risks](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/), including malicious instructions in user questions/retrieved chunks, sensitive-data exposure and resource limits, before production use |
| The model judge and reference answers can contain mistakes. We experimented with running example code separately, but code execution was not part of the answer evaluation. | Manually review a sample and run selected code examples in isolated environments with pinned dependencies, timeouts and no credentials/network. |
| Qualitative review is subjective: open-ended debugging questions can have ambiguous intent and several reasonable answers. Assistant and user judgments of usefulness or completeness may be too generous or too strict. | Define a shared review rubric and have independent reviewers assess a sample, recording disagreements separately from factual/source-support checks. |
| One generation request was policy-blocked; excluded from quality averages, not retried | Investigate why through available gateway diagnostics and provider support, so similar issues can be understood and handled without bypassing filters |
| Dense embeddings truncate at 512 tokens; full chunks reach generation; impact not isolated | Compare chunk sizes or a longer-context embedder with other settings fixed |
| Monetary inference cost unmeasured; local embeddings use CPU/storage and judging adds calls | Record token usage, applicable prices and local build resources; compare cost, speed and quality together |
| Technology choices reflect resources at hand, not proven best overall. The tested stronger generator took ~2.5x longer without a decisive improvement. | When different resources become available, compare a small set of model, embedding, chunking, retrieval and prompt combinations across development folds, like hyperparameter search with cross-validation. Assess usefulness, source support, speed and cost; evaluate the selected setup once on fresh, untouched test questions. |

Future work is proposed, not implemented or a guaranteed remedy.
The existing experiments reused a small dev sample; they are not cross-validation results.

## 6. Check it

```powershell
uv run pytest -q
uv run pre-commit install
uv run pre-commit run --all-files
uv run rag eval --split dev
uv run rag eval --answers
```

**Tests:** 81 offline checks pass. An earlier 83-check suite also passed in
clean-copy/wheel installs with shared caches, before the test review removed/merged checks.
They protect deterministic rules, not answer quality. Hooks require Python 3.12.

**Evaluation:** retrieval evaluation runs locally. `--answers` makes generation and judge
calls; add `--split test` for all 142 exposed test questions.

**Evaluation records:** [summary](results/published/summary.json),
[per-question scores](results/published/per_question.jsonl) and
[failures](results/published/failures.jsonl), about 71 KB total.
Full answers, source excerpts and raw responses remain local/ignored.

[Full findings and limitations](docs/FINDINGS.md) |
[Experiments](docs/EXPERIMENTS.md) |
[Audit trail](docs/AUDIT_TRAIL.md)
