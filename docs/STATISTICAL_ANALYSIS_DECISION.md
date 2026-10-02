# Statistical Decision and Wording-Probe Gate

October 2, 2026. **Technical specification, not independent scientific approval.**

## What the Owner Authorized

The owner explicitly accepted proceeding without human bilingual review and
authorized **up to $1 additional usage for 72 wording-probe calls only**. Human
validation remains absent, not completed. The probe permission did not itself
authorize calibration. The owner subsequently approved **104 revised calibration
graphs with a separate $10 additional ceiling**. Neither permission authorizes
translation-variant arms, the main study or a final sample allocation.
Calibration authorization is bound to contract `5fb3715550db`, source, runtime,
prompts and a $6.481868475 starting balance in the original ledger. The cumulative
ceiling is $16.481868475; previous spending is not reset. See the
[calibration runner](CALIBRATION_V6.md) for scope and safeguards.

## Wording Screen

### Completed Paid Result

The authorized screen completed on October 2: **72 first responses, 72 unique
paid requests, no replacements, no unresolved attempts, and zero new networks**.
The recorded conservative charge was **$0.00852425 of the $1 allowance**. Usage:
62,850 input tokens and 1,336 output tokens. Summed recorded request latency was
76.923 seconds; this excludes offline checks and the review pause and is not a
forecast for generating full networks.

| Instruction language / requested choices | Original failures | Revised failures |
| --- | ---: | ---: |
| English / 1 | 0/6 | 0/6 |
| English / 2 | 0/6 | 0/6 |
| English / 8 | 0/6 | 0/6 |
| Portuguese / 1 | **4/6** | **0/6** |
| Portuguese / 2 | 0/6 | 0/6 |
| Portuguese / 8 | 0/6 | 0/6 |
| Total | **4/36** | **0/36** |

**Engineering decision: `NO_FAILURES_OBSERVED` in the revised arm.** This supports
testing the revised instructions in bounded calibration. It does not validate
Hindi/Japanese behavior, other models/methods, translation equivalence, population
realism or a final sample size. These are compliance counts, not language-accuracy
scores. The wording bundle changed several details; no isolated grammatical
cause has been established.

Request 61, an original Portuguese one-choice prompt, reached the 512-token
limit while repeating IDs. The shared caller stopped and retained its charge.
After inspection and read-only review, `resume_wording_probe.py` verified the exact
61 received / 11 missing original IDs before opening an API client. It counted
the truncated response as a failure and purchased only the remaining 11 replies.
No partial response became a graph, no prompt/cap/cache identity changed, and no
historical charge was reset. The recovery wrapper is deliberately restricted to
that inspected checkpoint, not a general automatic retry mechanism.

The complete report and all 72 hashed reply files are in
`outputs/wording_probe_v6/`. Their request IDs, charges, response contents and
hashes were checked against the original private ledger. The stopped report is
retained as `stopped_before_resume.json`. The preparation-only export still labels
its manifest unexecuted; this separate paid report is the execution record.

### Predeclared Design

The screen uses GPT-6-Luna, English/Portuguese, counts 1/2/8, six fixed actors,
and original/revised wording: 72 first replies. Every actor sees the other 49
members of the same 50-person roster. Candidate order/data are matched across
wordings. Call order is deterministically shuffled before any results are seen.

The revised arm must have **zero first-response parse failures across 36 calls**
to receive `NO_FAILURES_OBSERVED`. Otherwise the engineering status is
`REVISE_BEFORE_CALIBRATION`. This is a conservative engineering screen, not a
statistical demonstration that error rates are zero or languages equivalent.
Six observations with zero failures still give a two-sided exact 95% binomial
upper endpoint around **46%**. The fixed actors are not a population sample;
the binomial intervals assume independent equal-probability trials and are
descriptive diagnostics, not national or roster-generalization evidence.

There are no paid correction retries. Malformed first replies are saved as
failures, with their usage charged. A transport ambiguity or abnormal finish stops
the batch. Restarting uses the exact cached reply and never silently buys another.
The original ledger remains authoritative; the $1 allowance cannot reset on resume.

OpenAI's [GPT-6-Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
was checked October 2: standard short-context prices are $0.10 input and $0.50
output per million tokens. The existing caller additionally reserves a 25% input
allowance and caps output at 512 tokens. No cache discounts are assumed. Provider
request IDs follow the [official API guidance](https://developers.openai.com/api/reference/overview).
This is conservative accounting, not a provider invoice guarantee.

## Analysis Rules Now Implemented

- [x] Network generations, not individual edges/nodes/decisions, are repetitions.
- [x] Keep all predeclared hypotheses in the multiplicity family, including those
  without complete results. Missingness must not reduce the correction penalty.
- [x] Missing paired observations suppress inferential output for that contrast.
  Do not drop failures and describe the remaining subset as unbiased.
- [x] Zero-variance samples do not produce fabricated certainty or a spurious
  infinite t-statistic from floating-point roundoff.
- [x] Holm adjustment and Bonferroni simultaneous intervals are implemented
  independently of observed significance. They serve different purposes; a Holm
  rejection need not be identical to a Bonferroni interval excluding zero.
- [x] Preserve assumption-dependent paired-t results separately from conservative
  bounded-mean sensitivity results. Neither method fixes dependent generations.
- [x] Distinguish a null result from equivalence. No equivalence claims without
  an approved raw-unit margin and a suitable procedure.

Primary outcome proposal remains global density/degree-one share, and
clustering/modularity for local/sequential/iterative. Within each method, RQ1
and RQ4 each have 24 primary hypotheses; RQ3 has 84. These are separately declared
families, not study-wide error control. RQ2 remains exploratory unless its own
confirmatory family is approved before collection.

`statistical_review.py` accepts a complete hypothesis-by-repetition matrix.
The repetition columns must represent genuinely recorded matching schedules,
not seed labels alone. It cannot verify those identities from an array; any main
analysis integration must verify them from receipts first. Per-contrast means,
distributions and counts remain the first report, not just a significance label.

Paired-t inference requires credible distribution/independence assumptions;
small skewed or mixture samples may violate them. The conservative sensitivity
uses the [Hoeffding bound for independent bounded observations](https://nowak.ece.wisc.edu/SLT07/lecture7.pdf)
and a union bound within a declared family. Its wider intervals are not evidence
that the study has adequate power. Bounds must come from the metric definition,
not the observed range: differences in density, degree-one share and clustering
are within [-1, 1]; use the conservative [-2, 2] difference bound for modularity.
Undefined modularity remains missing. Bounded homophily assumptions must be
derived separately; never reuse a [-1, 1] bound blindly for every score.

The implementation follows the [documented Holm step-down procedure](https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html)
without adding a dependency. Probe and Monte Carlo intervals use SciPy's
[Clopper-Pearson method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

### Stress-Test Result

The saved offline experiment uses 1,000 simulated worlds for each of four
bounded-null scenarios at n=8 and n=32: 8,000 worlds total, 24 hypotheses per
world. These are numeric test fixtures, not real model or network observations.
At n=8, independent uniform bounded differences yielded a **12.5%** family false
positive rate for paired-t plus Holm (Monte Carlo 95% interval about 10.5%-14.7%),
above the nominal 5%. Its normal-difference assumption is not universally safe.
The conservative bounded method yielded 0% there; in the n=32 bimodal scenario
it yielded 0.6%. These limited scenarios do not certify general performance.

**Decision:** do not grant blanket approval to small-sample t-test significance
claims. Preserve paired-t outputs as assumption-dependent diagnostics. Use
estimation-first reporting and show conservative uncertainty sensitivity; choose
any confirmatory procedure and precision target before the main data are seen.
The conservative method can be too wide to resolve useful effects, so low false
positive rates do not establish adequate power or justify the spending plan.
Evidence: `outputs/statistical_review_v6/report.json`. The review deliberately
does not claim that a successful engineering test creates scientific approval.

## What Cannot Be Approved by Running Tests

- [ ] Scientifically meaningful raw-unit effects and equivalence margins.
- [ ] Final numbers of repetitions for each method, based on a declared precision
  objective, not a favorable p-value or the old arbitrary total of 896.
- [ ] Whether inference will be estimation-only, assumption-dependent testing,
  or another reviewed procedure. The new sensitivity code is not permission to
  silently substitute methods after examining main-study results.
- [ ] Model-by-condition heterogeneity, translation-variant robustness, and whether
  a second roster is needed for the intended generalization.
- [x] Engineering calibration-runner integration is implemented and tested offline;
  see [the runner handoff](CALIBRATION_V6.md). This is not scientific approval.
- [ ] Acceptance of the revised calibration scope and finite budget, followed by
  a separately reviewed main-study protocol and budget.

The owner's waiver removes the operational human-review requirement, not the
scientific limitation. A code agent can test implementation and document a
defensible proposal; it cannot invent a professor's approval, scientific effect
threshold, independent human validation or a guarantee of conference acceptance.

## Reproduction

```powershell
# Offline, no credential:
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_wording_probe.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_statistical_review.py
.\.venv\Scripts\python.exe -B statistical_review.py
```

The probe uses `wording_probe.py --execute-probe` only after explicit hash-bound
authorization. Never expose a key in a command, notebook, report or Git commit.
This screen is now complete: do not execute it again as a new experiment. The
bounded recovery script refuses the completed 72-receipt state before transport.
Results belong in `outputs/wording_probe_v6/`; they are not 72 new networks and
are never inserted into the viewer's observed-network dataset.
