# Experiment log

Coverage = expected facts conveyed. Support = judge-supported citation pairs.
Usually 10 reused dev questions + 3 controls; estimates, not ground truth.

| Run | Change / comparison | Result / decision |
|-----|-------|-------------------|
| Retrieval | BM25 vs dense vs hybrid; 61 dev queries | Hybrid led: hit@8 70.5%. Retained. |
| Baseline | Free-text, eight capped chunks | Coverage 54.2%; support 67.7%. |
| E01 | Full chunks + concise prompt; one case | Recovered cut token-limit evidence. |
| E02 | Full chunks + concise prompt; 13 cases | Coverage 55.0%; support 66.3%. Rejected. |
| E03 | Evidence-first prompt | Coverage 24.2%; support 71.6%. Rejected. |
| E04 | Language-aware retrieval | Recall@8 slightly worse; coverage 42.5%. Rejected. |
| E05 | Remove character cap | Coverage 39.2%; support 73.6%. Retained to avoid cutting evidence. |
| E06 | Five evidence-removal pairs | All target checks passed; diagnostic only. |
| E07 | Same checks with noisy context | All passed; attribution problems remained. |
| E08 | Runtime semantic verifier | Common-eight coverage 50% -> 30%; support 69% -> 94%; two failures, ~2x latency. Removed. |
| E09 | Structured claims with JSON/schema and source-label checks; no runtime check that sources support claims | Coverage 20.8%; support 91.7%. Retained with limitations. |
| E10 | Stronger generator, fixed evidence | Coverage 16.7% -> 23.3%; ~2.5x latency. No switch. |
| E11 | Changed the answer format to allow solution blocks, missing-information statements and clarifying questions, instead of only cited claims | Coverage 19.2% -> 50%; support ~96% -> ~48%; some generated code was incorrect. Rejected; not in the current pipeline. |
| Final | Current pipeline; 142 exposed test questions | 141 answers/grades; coverage 16.4%, support 92.1%. |

**Retained trade-off:** better citation support, lower completeness than the original
full-test baseline (33.9% coverage, 72.8% support). Not an overall improvement.

Supporting checks: F01 confirmed wrong-language attribution; evidence audit found mixed
sufficiency; five local input-routing assertions passed; clean-copy/wheel tests passed.
E11's first run had API/schema errors; its corrected support excludes four invalid judge entries.

Different samples/claims and fallible judges limit comparisons. Final results are post-test,
not fresh held-out evidence. [Changes, failures and decisions](AUDIT_TRAIL.md) |
[Final findings](FINDINGS.md).
