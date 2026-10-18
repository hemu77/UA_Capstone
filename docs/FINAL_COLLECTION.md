# Final Collection Contract

October 18, 2026. This supersedes the **candidate decisions**, not the evidence,
in [the October 17 review](PRECOLLECTION_REVIEW.md). The execution entry point is
`python -X utf8 -B -m final_study`. No main-study spending has been authorized.

**Launch gate passed:** 226 offline tests, 112 engine preflight combinations,
and eight received Hindi global graphs verified. The live check used an
estimated conservative **$0.02740295** and **30.626 serial API seconds**. All eight
parsed; one valid empty graph is preserved. The next study action is the owner's
explicit main-generation launch. See [verification evidence](FINAL_COLLECTION_VERIFICATION.md).

## Decisions Made

- [x] Adopt the narrow claim: sensitivity to seven exact instruction contracts
  on one fictional 50-adult roster and four fixed model configurations.
- [x] Adopt four models: GPT-4.1, GPT-5.6-Luna, GPT-6-Luna, GPT-6-Sol. No silent
  replacement with a newer model, and no old graphs counted as fresh data.
- [x] Adopt four methods: global, local, sequential, iterative.
- [x] Adopt seven conditions: US/English, India/English, Japan/English,
  Brazil/English, US/Hindi, US/Japanese, US/Portuguese. These are **not** the
  complete 4-country by 4-language cross-product; Spanish is not included.
- [x] Adopt 32 fresh schedule blocks, seeds 44000 through 44031: **3,584 planned
  network attempts**, 896 per method. This is not 896 in total. A failed attempt
  remains a failure; the plan does not promise 3,584 valid PNGs.
- [x] Fix the primary endpoints before collection: density for global;
  average local clustering over all 50 nodes for the other three methods.
- [x] Fix 24 primary comparisons: six contrasts against US/English per method.
  Give the four models equal weight within each schedule; models do not count
  as independent repetitions. Use paired schedule-block diagnostics, Holm
  adjustment, and simultaneous Bonferroni intervals, with assumptions stated.
- [x] Preserve missingness. An incomplete model panel cannot become a three-model
  average. Suppress inference for that comparison; show available-block
  summaries and numerical-completion bounds, not fabricated zero values.
- [x] Keep homophily, age assortativity, model-specific contrasts and other
  topology measures descriptive. They do not establish causal demographic
  dominance or population-wide language/culture effects.
- [x] Adopt the global parser's deterministic duplicate-pair normalization.
  Validate every line first; preserve raw text and normalization counts.
  Invalid IDs, malformed lines and self-links invalidate the response.
- [x] Adopt the clarified Hindi **global** instruction for the targeted live
  check. It says realistic/plausible rather than merely possible. This is not
  the unsuccessful earlier Hindi local-alternative wording.
- [x] Choose a conservative transport policy: SDK retries zero, no automatic
  paid replacement after an uncertain request, full reservation retained.
  This is the selected execution behavior, not a promised retry feature.
- [x] Complete and replay the eight-graph Hindi live check with a process-local key.
- [ ] Authorize and execute the full main batch after that live check passes.

The authoritative frozen identity is [contract.json](../final_study/contract.json).
Readiness is recomputed by `status`; a checked box or an old report is not a
substitute for its checks. The main run remains blocked until all eight Hindi
checks finish and parse. Valid empty graphs remain valid and are not rerolled.

## What Changes, What Stays

Earlier work produced reliable graph files but left an analysis proposal rather
than an executable final study. The new package connects the adopted prompts,
parser, paid runner, cached replay, saved PNG/adjacency files and analysis.
It uses the same reviewed engine, with an isolated adapter, rather than editing
the files that generated earlier evidence.

All 104 V8 observations remain intact: 100 valid graphs, four invalid responses,
including four valid empty global graphs. They are calibration, not confirmation.
The three duplicate-related old failures are not relabeled successful. The new
parser applies prospectively. No wholesale calibration regeneration is needed.

The completed Hindi check used two separate schedules, 45000 and 45001, across
the four models: eight global graphs. It tests execution/compliance of the changed text;
it does not prove translation equivalence or supply confirmation observations.

| Model | Edges, seed 45000 | Edges, seed 45001 |
|---|---:|---:|
| GPT-4.1 | 64 | 57 |
| GPT-5.6-Luna | 0 | 99 |
| GPT-6-Luna | 68 | 117 |
| GPT-6-Sol | 83 | 114 |

Every graph contains all 50 personas. Three responses contained one redundant
undirected pair each; the frozen parser recorded that fact and collapsed only
the duplicate. It did not buy corrected responses or repair invalid IDs.
The empty GPT-5.6-Luna realization is real received output, not a missing graph.

The frozen analysis specification is embedded unchanged from the earlier
candidate. The new enclosing contract explicitly adopts it. Analysis output
retains both its analysis-version identifier and the actual execution-contract
hash, so the former candidate label cannot mix old observations into this run.

## Cost And Time

Official Standard short-context input/output prices, checked October 18, per
million tokens:

| Model | Input | Output | Evidence |
|---|---:|---:|---|
| GPT-4.1 | $2.00 | $8.00 | [Model documentation](https://developers.openai.com/api/docs/models/gpt-4.1) |
| GPT-5.6-Luna | $0.20 | $1.20 | [Model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| GPT-6-Luna | $0.10 | $0.50 | [Model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna) |
| GPT-6-Sol | $2.00 | $10.00 | [Model documentation](https://developers.openai.com/api/docs/models/gpt-6-sol) |

Rates match the historical forecast. At V8 usage, 3,584 attempts project to
**$409.77 and 105.22 serial API hours**. A 25% cost allowance is **$512.21**;
a 50% time allowance is **157.82 hours**. These are estimates, not an invoice,
upper bounds, or a promise of completion. Corrections, provider latency and
uncertain requests can increase consumption. The ledger conservatively applies
a 1.25 input multiplier to cover cache-write premiums, rather than promising
cache discounts. An explicit launch allowance still controls spending.

The old ~$102.44 estimate was for **896 total attempts/eight schedules**. The
32-schedule choice improves precision, but does not guarantee power: a
simultaneous interval half-width near 0.03 assumes a paired-block SD near 0.05.
Calibration has not established that variance. Missing responses can still
leave bounds-only results. No significant finding is an acceptance criterion.

GPT-4.1 is deprecated, with shutdown scheduled January 14, 2027. It remains the
legacy comparator; the runner checks access and refuses to substitute a model.
[Official notice](https://developers.openai.com/api/docs/deprecations).
Price freshness is checked before new calls; stale evidence blocks spending.

## Run And Inspect

Use the pinned Python environment in the repository. Configure `OPENAI_API_KEY`
locally; never put a key in a command, Git, a notebook, or a PR comment.

```powershell
# No API calls. Verify the frozen code and exercise all 112 combinations.
python -X utf8 -B -m final_study preflight
python -X utf8 -B -m final_study status

# Targeted live check only, separate from the main study. Requires approval.
# Already completed. In the canonical workspace this exact command resumes
# from receipts; do not run it in a fresh clone to recreate existing evidence.
python -X utf8 -B -m final_study launch --stage hindi_check --approve-usd 1 --approval-reference owner-hindi-launch-check

# Public verification of the eight actual received observations; no API calls.
python -X utf8 -B -m final_study verify --stage hindi_check
python -X utf8 -B -m final_study status

# Do NOT execute until the owner authorizes the full collection and allowance.
python -X utf8 -B -m final_study launch --stage main --approve-usd 520 --approval-reference owner-final-collection

# Replay all saved final observations without sending or buying responses.
python -X utf8 -B -m final_study verify --stage main
```

The $520 example is a rounded allowance above the forecast, **not authorization
and not a guarantee that every planned cell fits**. Repeating the exact launch
command resumes from received evidence. Changing its cap/reference is rejected
rather than creating a new budget. A missing ledger, unmatched journal, altered
receipt, wrong service tier, changed source, or uncertain response stops before
further model calls. Inspect and reconcile that state; do not delete it to retry.

Outputs live only under `outputs/final_study/<contract-id>/<stage>/`. A successful
observation has a JSON receipt, `.adj` and `.png`; failures have a receipt and
no fabricated graph. Main completion writes `network_metrics.csv`,
`analysis_rows.json`, `primary_contrasts.json` and `primary_contrasts.csv`.
Private ledgers, request journals, authorization and raw transport files are
ignored by Git. Public receipts include submitted synthetic prompts, responses,
usage and parser decisions, never authentication headers.

The canonical paid workspace is `UA_Captsone_SocN`. `UA_Captsone_publish` and
GitHub receive public artifacts, not a second spendable ledger. Team members
can run `verify --stage hindi_check` on the public checkout. `status` and paid
launch additionally require the canonical private accounting evidence and a
local credential. Do not copy a ledger into multiple independently running
checkouts or delete it to bypass a resume failure.
The supplied key was used through hidden terminal input in process memory only.
It was not saved in the repository or Windows environment. A later launch still
needs a process-local key; that is a credential prerequisite, not unfinished
prompt, parser, analysis or collection-runner work.

## Limits

AI-assisted translation review is not native-speaker certification. Fixed
nomination quotas constrain quota-method density. One fictional roster, four
chosen model configurations and one wording per language do not represent
national populations, all models or all translations. Provider aliases may
change; request order is interleaved and response metadata is retained.
Offline tests verify software behavior, not scientific truth or acceptance by
a conference. The professor's approval has not been claimed or fabricated.
