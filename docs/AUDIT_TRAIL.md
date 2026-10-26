# Audit trail

Why we chose the current PoC, what we tried, and where the supporting records live.
For the short overview, see [EXPERIMENTS.md](EXPERIMENTS.md); for final results and
limitations, see [FINDINGS.md](FINDINGS.md).

## Retained configuration

| Decision | Reason | Trade-off |
|----------|--------|-----------|
| Interactive CLI, not an HTTP API | Matches the assignment without an extra service layer | No web/API interface |
| Python 3.11+, uv lockfile and src layout | Reproducible installation and packaging | Reviewers need uv |
| AMANDA pre-commit standards; Google-style docstrings | Consistent formatting and documentation | Hooks require Python 3.12 |
| KPN gateway using Azure CLI authentication; OpenAI fallback | Uses the resources available; fallback supports other environments | Gateway requires KPN access |
| OS certificate trust through truststore; cached Azure CLI token | No certificate file or azure-identity dependency | Relies on OS trust and an installed Azure CLI |
| Local BAAI/bge-small-en-v1.5 embeddings on CPU | Avoids embedding API usage; reproducible locally | 512-token embedding limit can truncate chunks; PyTorch is a large dependency |
| BM25 plus dense retrieval, fused with RRF | Hybrid led the 61-question dev comparison: hit@8 70.5% | Two aligned indexes; no relevant top-eight chunk for about 30% of dev questions |
| bm25s, English stopwords, no stemming | Fast sparse indexing and persistence | Whole identifiers can miss sub-word matches |
| Exact NumPy dense search; RRF k=60 over each ranker's top 100 | About 76 MB of vectors does not require a vector service | Not designed for millions of chunks; alternatives were not benchmarked |
| Resumable 2,000-chunk embedding shards | Survives an interrupted CPU build | Additional persistence/alignment checks |
| Corpus-only index, using source path and chunk text | Preserves the assignment's search boundary and citation IDs | Does not ensure generated claims come only from the corpus |
| Sorted-ID, seed-42 dev/test split: 61/142 questions | Deterministic split with most questions reserved for evaluation | Small dev set; test questions became exposed during this investigation |
| Top eight full chunks, no character cap | A provisional 4,000-character cap removed relevant evidence | Larger requests; gateway/model limits still apply, with no token-aware fallback |
| Mini answer model; stronger offline judge | Keeps generation latency low while separating answer evaluation | Same-family judging is not independent verification; billing cost was not measured |
| One structured generation call; Python-rendered citations | Enforces format and source identity without a second runtime model call | Cannot enforce relevance, source support, completeness or runnable code |

The final user decision was to retain this configuration with its measured trade-off:
better judged citation support but lower reference-fact coverage than the original
free-text baseline. It is not an overall quality improvement or production guarantee.

## Reading the evidence

- Unless stated otherwise, answer experiments reused 10 dev questions and 3 refusal
  controls, with one stochastic output per condition. They were not cross-validation.
- Coverage means expected reference facts conveyed; support means judge-supported
  claim/citation pairs. Different answers produce different claim sets, so support
  percentages are not paired same-claim accuracy. Judges and reference answers can err.
- Reused small samples, historical comparisons and selected manual checks limit causal
  claims. No statistical significance, measured billing-cost or comprehensive code-execution
  claim is made. The separate local wiring diagnostic below is explicitly identified.
- Raw paths below are relative to `results\raw` and remain local/ignored, not published
  evidence. Saved records contain the exact inputs/settings needed to inspect the runs.

## F01: Python answer cited a TypeScript source

**2026-10-23, manual CLI demonstration:** Asked how to load a PDF in Python, the model
gave a `PyPDFLoader.load()` example but cited
`langchainjs/docs/core_docs/docs/how_to/document_loader_pdf.mdx_0_2919`.
Reading that corpus chunk confirmed it only documented TypeScript loaders.
The citation existed and was retrieved, but did not support the Python example.

The Python guide supported the import, construction and page behavior, but demonstrated
`alazy_load()`, not the synchronous call. Later language-aware/full-context answers avoided
the TypeScript attribution in individual samples, yet still had incomplete citation support.
An unsupported citation does not prove the code itself is invalid; F01 was not declared fixed.

Evidence: `f01-language-aware-answer-20261024-110223.json` and
`f01-full-context-answer-20261024-111159.json`.

## E01-E05: Context, prompts and retrieval (2026-10-24)

The original capped dev baseline scored 54.2% coverage and 86/127 supported pairs (67.7%).

| Run | Change and result | Decision |
|-----|-------------------|----------|
| E01 | One token-budget question: full chunks plus a concise prompt recovered `max_tokens_limit`, which occurred after the 4,000-character cap. Source text grew from 32,000 to 60,382 characters; answer shrank from 214 to 110 words. | Promising diagnostic, not a general result; two variables changed together. |
| E02 | Same change across 13 cases: coverage 55.0%, support 53/80 (66.3%); shorter answers omitted useful facts elsewhere. | Do not adopt the combined variant. |
| E03 | Evidence-first prompt with capped context: coverage 24.2%, support 68/95 (71.6%); some useful refusals, but excessive caution and unsupported attribution remained. | Do not adopt; a stricter prompt cannot recover omitted evidence. |
| E04 | Path/language-aware reranking: recall@8 0.1695 -> 0.1676 across 61 dev queries. Answer coverage 42.5%, support 69/92 (75.0%). | Do not adopt; useful cross-language material was lost and overall retrieval did not improve. |
| E05 | Remove the cap only: coverage 39.2%, support 78/106 (73.6%). Token-budget evidence became available, but grounding failures remained. | Retain full chunks to avoid known evidence loss, not as a proven quality improvement. |

E04 used explicit Python/JS/TS cues and source-path heuristics to move mismatched paths
behind other candidates. Only 2/13 answer cases received changed sources; score changes
on unchanged inputs show why aggregate differences cannot be attributed solely to reranking.

Historical reproduction commands make new model calls:
`uv run rag eval --answers --full-context-short`,
`uv run rag eval --answers --evidence-first`, and
`uv run rag eval --answers --language-aware`. These variants are dev-only and mutually
exclusive; E03/E04 retain their historical cap. Retrieval-only E04 uses
`uv run rag eval --language-aware`.

Evidence:

| Run | Artifact |
|-----|----------|
| Baseline | `answer-baseline-20261023-203011` |
| E01 | `token-limit-full-short-20261024-103519.json` |
| E02 | `answer-full-short-20261024-103852` |
| E03 | `answer-evidence-first-20261024-104900` |
| E04 | `retrieval-language-aware-20261024-105635.jsonl`, `answer-language-aware-20261024-105647`, `language-aware-comparison-20261024.json` |
| E05 | `answer-full-context-20261024-110914` |

## E06-E07: Removing supporting evidence (2026-10-24)

Five paired diagnostics covered PDF page loading, chain token budgets, ASR wrapper support,
assign input keys and PDF directory loading. Plans and expected behavior were saved before
generation; each pair compared supporting evidence present versus removed.

**E06:** Hand-selected clean contexts bypassed retrieval. All five present answers retained
the target facts; all five absent answers acknowledged the gap.

**E07:** Repeated with retrieved-context noise. All five pairs again passed the target-fact
rubric, but two positive contexts were constructed, not unaltered top-eight retrieval.
After removal, contexts contained only 2-7 chunks. One PDF answer still over-attached a
citation that did not independently support the whole claim.

These were narrow manual target-fact checks, not exhaustive citation-support scores or
proof of robust sufficiency detection. All gap answers had the historical `abstained: false`
flag; its lexical rule was not a semantic refusal check. No production change followed.

Evidence: `evidence-removal-pairs-20261024-112559` and
`noisy-evidence-pairs-20261024-113522` (plans, exact contexts, answers and manual reviews).

## Original held-out evaluation (2026-10-24)

Frozen original free-text generator, hybrid retrieval and eight full chunks; all 142 test
questions attempted. Retrieval recall@8/hit@8/nDCG@8 were 0.157/0.648/0.209.

Saved-run recovery handled network interruptions without regenerating completed records.
Local standby events overlapped failures; this did not prove a gateway outage.
All 142 answers completed, but judging case `78429932` returned `content_filter`.
It remained an explicit ungraded exclusion, without prompt changes or filter retries.

Across 141 accepted grades: **33.87% coverage, 1,232/1,692 supported pairs (72.81%)**,
129 uncited items, 61 zero-coverage and 20 full-coverage answers. These are model estimates.
Selected source checks found a useful Pinecone remedy and incomplete FastAPI guidance;
they did not validate the full run.

Evidence: `answer-test-full-context-20261024-114350`; retrieval records are in
`results\retrieval-test.jsonl`. The split, settings/source hashes, question order and
grade/exclusion partition were checked. From this point onward, test questions were exposed.

## E08: Runtime verifier, then simplification (2026-10-24)

An approved post-test implementation drafted structured claims, then used a second model
call to verify attached sources. Python validated evidence-line references and withheld
claims failing support. An earlier exact-quotation approach failed on notebook JSON;
line references solved that formatting issue, not semantic reliability.

The paired dev comparison used identical sources and alternating arm order.
Baseline answered 10/10 dev cases; verified output succeeded on 8/10. On those common eight:
coverage **50.0% -> 30.2%**, support **62/90 (68.9%) -> 51/54 (94.4%)**.
Mean latency including failures rose **4.70 -> 9.18 seconds** (excluding retrieval/judging).
Both failures were invented source IDs rejected before verification, not successful abstentions.
Manual checks also found ambiguous citation attachment and overly literal verifier rejection.

**Decision:** Remove the runtime verifier. Higher judged support did not justify reduced
coverage/availability and roughly doubled latency. Keep semantic grading offline.
The removed verifier is not a supported production option.

Evidence: `paired-verified-dev-20261024-134102`; line-reference smoke records:
`verified-lines-post-test-20261024-133554` and
`verified-lines-final-serialization-smoke.json`.

## E09: Structured claims without runtime source-support verification (2026-10-24)

**Retained rules:** One JSON generation call; exact `claims` and boolean `incomplete`
fields; each claim has nonempty text and one known source label such as `S1`.
Python maps labels to retrieved IDs and renders citations. Invalid fields, duplicate keys,
unknown labels, malformed/truncated completions and reserved citation/link markup reject
the entire response. No silent claim dropping, repair call or application fallback.
API errors propagate; the CLI reports errors without displaying the draft.

Empty claims require `incomplete=true` and render a fixed abstention. Nonempty incomplete
answers receive a notice explicitly attributed to the model. These rules check structure
and source identity, not whether a source supports a claim or the answer solves the task.

All 13 dev/control requests completed: **20.83% coverage, 66/72 supported dev pairs (91.67%)**,
zero judge-extracted uncited dev claims, mean warmed retrieval/generation 2.675 seconds.
Wrong-language attribution and irrelevant API inventories persisted. Only 2/3 controls
were judge-labelled refusals; the private-password answer gave irrelevant connection-string
facts rather than a satisfactory refusal.

Evidence: `answer-structured-claims-post-test-20261024-135655`.
The judge saw displayed answers, not internal metadata. These results did not establish
reliable grounding, sufficiency detection or task resolution.

## Evidence review and local wiring diagnostic

[DEV_EVIDENCE_REVIEW.md](DEV_EVIDENCE_REVIEW.md) preserves all ten dev assessments,
source excerpts, uncertainty and user feedback. The initial evidence-sufficiency labels
were three sufficient, six partial and one insufficient; subjective confidence scores
were not calibrated probabilities or new evaluation grades.

For multi-input case `78485621`, an isolated Python 3.12 environment with
`langchain-core==0.2.43`, fake retrieval and no LLM passed five local wiring assertions.
Selecting schema/question values and routing only the question to retrieval worked;
changing `.invoke()` alone still sent the whole dictionary. This tested one pinned version,
not the original user's environment or generated answers generally.

Evidence: `multi-input-local-check-20261024/check_multi_input.py`.
No project dependency changes or retrospective score corrections.

## E10: Stronger generator with fixed evidence (2026-10-24)

Paired the mini generator with the stronger deployment using identical saved evidence,
structural rules and alternating order. All 13 cases per arm completed.
Dev coverage **16.67% -> 23.33%**; support **66/68 (97.06%) -> 73/74 (98.65%)**.
Mean generation latency **2.729 -> 6.706 seconds**, excluding retrieval/judging.

**Decision:** No switch. Some answers improved, but irrelevant inventories and unsupported
API adaptations remained. The stronger generator was also the judge, and inconsistent
nugget credit made small differences especially uncertain. No generated programs were executed.

Evidence: `answer-model-pairs-20261024-150107`.

## E11: Solution blocks, missing information and questions (2026-10-25)

Compared the current cited-claim format with a candidate allowing cited solution blocks,
explicit gaps and clarifying questions. Schema, prompt and rendering changed together.
Both arms used the same mini model, saved sources and alternating order.

**Initial failure:** The candidate prompt omitted the literal word JSON required by JSON mode:
two API failures plus one schema failure. Preserved the run, added only the JSON keyword
and a regression check, then reran both arms rather than replacing selected failures.

**Corrected comparison:** All 26 answers completed; 25 grades were accepted.
The current future-control grade failed validation and was not retried or counted as zero.
On ten common dev cases, coverage **19.17% -> 50.0%** and latency
**2.146 -> 4.257 seconds**. Raw support fell **47/49 (95.92%) -> 20/46 (43.48%)**.
Four candidate entries had empty citation IDs; excluding them gives **20/42 (47.62%)**.
That diagnostic does not replace the saved grades. The 45 candidate uncited items included
judge-format errors and gap statements, not 45 confirmed technical errors.

Manual checks found wrong-source attribution, an unestablished Azure `content_key`,
whole-dictionary passthrough instead of schema/question selection, and an invented GooglePalm
removal version. The tungsten control cited unrelated evidence; the private-password control
asked for a connection string, potentially soliciting secrets. No credentials were provided.

**Decision:** Reject; not in the current pipeline. More task-focused output increased
coverage but regressed source support and refusal behavior.

Evidence: initial `output-contract-pairs-20261025-093906`, corrected
`output-contract-pairs-20261025-094256`. Hash-checked candidate implementation and tests
were archived beside the corrected `run.py`, then removed from the shipped package.
For local reproduction, restore archived `contract_experiment.py` to
`src\rag\contract_experiment.py` and `test_contract_experiment.py` to
`tests\test_contract_experiment.py`; these are not supported CLI options.

## Final current-pipeline rerun (2026-10-25)

User approved rerunning all 142 exposed test questions with the frozen retained pipeline.
All answers were saved before displayed-answer-only judging. Normal production SDK retry
settings remained; no application repair or filtered-request retry was added.

**Completion:** 142 attempts, 141 successful answers and 141 accepted grades.
Case `78429932` was blocked during generation, unlike the baseline's judging-stage block.
The exact error was saved; no answer was invented, reworded or submitted for judging.
A policy error does not establish its cause or imply the question was unsafe.

**Reporting rule:** Availability failures are not semantic abstentions or zero-quality grades.
Disclose failure stage/ID, exclude ungraded cases from quality denominators and note possible
selection bias. Both full runs were graded on the same 141 IDs and used identical source
text/order for all 142 questions; frozen hashes and record partitions were checked.

Current estimates: **16.38% coverage, 938/1,018 supported pairs (92.14%)**;
94 zero-coverage and four full-coverage answers. Four uncited items were fixed partial-answer
notices on inspection. There were nine judge-labelled refusals including partial declines,
but no full structural abstentions. Mean answer length was 112.94 words; warmed
retrieval/generation averaged 2.215 seconds, including failure but excluding startup/judging.

**Final decision:** Retain the configuration, disclose reduced coverage alongside better
judged support, and stop tuning against these exposed questions. This is post-test evidence,
not a fresh held-out result or proof of overall improvement.

Evidence: `answer-test-structured-final-post-test-20261025-100315`
(settings, source hashes, attempts, exact evidence, grades, summary and runner).
Compact scores and failure records are included under `results\published`;
full raw evidence remains local/ignored. On 2026-10-26, the user confirmed that the
public-repository clone runs, including indexing and interactive answers.
