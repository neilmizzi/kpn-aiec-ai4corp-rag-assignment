# Final findings: retained configuration

Decision date: 2026-10-25. This document is the consolidated results summary;
[AUDIT_TRAIL.md](AUDIT_TRAIL.md) preserves the investigation and rejected experiments.

## Configuration evaluated

| Component | Retained choice |
|-----------|-----------------|
| Searchable data | FreshStack October 2024 LangChain corpus, 49,514 chunks; no live web search |
| Sparse retrieval | BM25 over source paths and chunk text |
| Dense retrieval | Local `BAAI/bge-small-en-v1.5`, normalized embeddings, exact NumPy search |
| Fusion/context | RRF, smoothing constant 60; top 100 candidates per ranker; eight full chunks |
| Generator | `gpt-54-mini-2026-03-17`, one structured-claim generation call |
| Output enforcement | Exact JSON schema, known source labels, Python-rendered document-ID citations |
| Insufficient evidence | Empty claims abstain; partial claims get a model-attributed warning |
| Offline judge | `gpt-54-2026-03-05`; not part of interactive answering |
| Interface | Interactive CLI |

There is no runtime semantic verifier, language-aware retrieval default, stronger-model
switch or task-focused answer/gap/question contract. The user retained this configuration
with its limitations, not because it was demonstrated to be best overall.

## Evaluation scope and provenance

The 203 supplied questions were split deterministically into 61 dev and 142 test questions
(sorted IDs, seed 42). Reference answers, nuggets and relevance labels are evaluation data,
not indexed content or generator evidence.

Original retrieval measurements used the held-out test split. Retrieval defaults did not
change afterward; the final answer run confirmed identical retrieved text/order to the
original baseline for all 142 questions.

The current answer architecture changed after test exposure. Its final evaluation is
therefore **post-test evaluation on previously exposed questions**, not fresh held-out
evidence. One output per question, unchanged prompts/settings during the run; all generation
attempts saved before judging. The judge saw question, nuggets, exact sources and displayed
answer, without internal metadata. No application-level retries or repair; production SDK
retry behavior remained unchanged.

## Final results for the retained configuration

| Measurement | Result | Meaning |
|-------------|--------|---------|
| Retrieval recall@8 | 0.157 | Mean fraction of judged-relevant chunks recovered in the first eight |
| Retrieval hit@8 | 0.648 | Fraction of questions with at least one judged-relevant top-eight chunk |
| Retrieval nDCG@8 | 0.209 | Rank-sensitive relevance score using binary judgments |
| Retrieval recall@20 | 0.242 | Mean fraction of judged-relevant chunks recovered in the first twenty |
| Generation attempts | 142 | All test questions attempted |
| Successful answers | 141/142 (99.3%) | One generation request blocked by gateway content policy |
| Accepted grades | 141 | Every successful answer graded; no judge failures in this run |
| Mean reference-nugget coverage | 16.4% | Mean fraction of expected key facts conveyed per graded answer |
| Supported claim/citation pairs | 938/1,018 (92.1%) | Judge-estimated support, pooled across extracted pairs |
| Unsupported claim/citation pairs | 80/1,018 (7.9%) | Valid source identity did not establish semantic support |
| Judge-extracted uncited items | 4 | All four were fixed partial-answer disclosures on inspection |
| Zero / full nugget coverage | 94 / 4 of 141 | 94 answers conveyed none of the reference nuggets; four conveyed all |
| Judge-labelled refusals | 9/141 | Includes partial declines; not an appropriate-refusal success rate |
| Full structural abstentions | 0/141 | No successful answer returned the empty-claims abstention |
| Mean displayed answer length | 112.9 words | Successful answers only |
| Mean warmed retrieval + generation time | 2.21 seconds | Includes failed attempt; excludes startup and judging |

These are different measurements, not interchangeable accuracy scores. In particular,
92.1% supported pairs does not mean 92.1% of questions were solved correctly, and zero
nugget coverage does not prove every statement in an answer was false.

## What changed relative to the original baseline?

| Measurement | Original free-text baseline | Retained structured pipeline |
|-------------|-----------------------------|-------------------------------|
| Mean reference-nugget coverage | 33.9% | 16.4% |
| Supported claim/citation pairs | 1,232/1,692 (72.8%) | 938/1,018 (92.1%) |
| Zero-nugget answers | 61/141 | 94/141 |
| Successful answers / attempts | 142/142 | 141/142 |

Both quality summaries use the same 141 question IDs and identical retrieved evidence,
but different generated answers and extracted claim sets. Citation support rose by
19.3 percentage points while nugget coverage fell by 17.5 points. This is an observed
trade-off, not a controlled same-claim accuracy result or proof of a schema-only effect.

**Conclusion:** The retained system produces more consistently supported statements but
less complete answers. It can correctly describe an API without explaining the remedy
the user needs. It is not demonstrated to be an overall improvement over the baseline.

## Concrete examples

**Useful supported content, not necessarily a complete solution:** In current case
`77644218`, the answer identifies Pinecone's warning against passing an embedding callable
and its recommendation to pass an Embeddings object. It cites the relevant constructor.
However, the answer mainly lists constructor/deprecation facts and earns only 1/2 nuggets.

**Task-resolution failure despite related facts:** In current case `76288582`, the answer
describes LlamaIndex's `streaming=True` and `response_gen`, then lists ServiceContext
parameters. It does not provide the requested FastAPI solution and earns 0/3 nuggets.

These are selected output inspections, not a comprehensive human audit. Generated code in
the final evaluation was not executed.

## Limits on the conclusions

- Retrieval has no judged-relevant top-eight chunk for 35.2% of questions. Unjudged chunks
  count as non-relevant; retrieval metrics alone do not establish evidence sufficiency.
- Dense embeddings truncate inputs at 512 tokens while generation sees full chunks.
  This is a plausible retrieval limitation, not an experimentally isolated root cause.
- Shape/source checks cannot prove relevance, entailment, completeness or code correctness.
  No full abstentions were observed; that does not establish that every question was
  answerable. A partial warning is a model assessment, not verified evidence sufficiency.
- Gateway policy blocked generation for `78429932`; the original run blocked judging
  for the same ID. This is an availability/assessment limitation, not a semantic
  abstention or zero-coverage grade. No filtered request was reworded or retried.
  Quality means exclude ungraded cases and may be selection-biased.
- Model judges and reference nuggets are imperfect. The four uncited disclosures show
  rubric errors; support is estimated over different claim sets. Single stochastic runs
  and exposed test data limit generalization. No significance, billing-cost or
  production-reliability claim is made.

## Testing versus answer evaluation

The 81 passing offline tests protect deterministic behavior: corpus-only indexing,
resumption/alignment, ranking/fusion, split and evaluation-data separation, JSON/source
validation, citation rendering and successful/error CLI flows. External services are
mocked; tiny fixtures use real index persistence where practical.

They do not prove answer grounding or usefulness. No line/branch coverage percentage has
been measured. The test suite makes no live gateway calls.

## Evidence and remaining delivery work

Local ignored artifacts:

- Current answers: `results\raw\answer-test-structured-final-post-test-20261025-100315`
  (settings/source hashes, exact evidence, attempts, grades, summary and runner).
- Original baseline: `results\raw\answer-test-full-context-20261024-114350`
  (answer and retrieval summaries, original outputs, grades and exclusion).

The decision log records the rejected verifier, model and output-contract experiments;
their development results are not substitutes for the final current-configuration scores.
Compact [summaries](../results/published/summary.json),
[per-question scores](../results/published/per_question.jsonl) and
[failure records](../results/published/failures.jsonl) are included in the repository.
Full raw artifacts remain local/ignored.

### Clean-copy setup check (2026-10-25)

Before the final test review removed/merged checks, verified the submission code in a
separate directory and newly created virtual
environment on Windows with Python 3.14.6 and uv 0.11.21. Source copies were hash-checked;
no existing project environment, index, results or local credentials were copied.

- `uv sync --locked --python 3.14` installed successfully using the existing shared uv cache.
- All 83 offline tests passed in the new editable installation, including tiny real index
  persistence/resumption checks. CLI help and pre-commit hook installation succeeded.
- Built a wheel, replaced the editable installation with that wheel, confirmed imports
  came from `site-packages`, and passed all 83 offline tests again. The installed CLI worked.
- With no index present, loading reports the required `uv run rag index` setup command.

This is a clean-copy installation check, not a remote-clone or fresh-machine verification.
The restricted environment template was not accessed or copied. Package caches and installed
Python interpreters were reused. No full corpus/model download, full index rebuild, live
gateway request, alternate Python version or other operating system was tested in this check.

The configuration is retained and findings documented. Remote-clone/published-repository
verification and approval of final repository contents/evidence delivery remain.
