# Completed Calibration: Cost and Runtime Estimates

**October 2 update:** these are preserved V5 historical results/forecasts. The
new 104-graph calibration is complete; read [its separate report](docs/CALIBRATION_V6_RESULTS.md)
and [plan.md](plan.md). Main collection still requires scientific and budget approval.
The 896 target, calibration-reuse assumption and remaining-cost/runtime forecasts
below must not be applied unchanged to revised prompts or allocation.

Inspection date: 2026-09-30. **68/68 calibration networks verified. Paid generation
stopped. The 896-network main batch is not authorized.**

## Decision Summary

- [x] All 68 planned calibration networks collected on the full 50-person roster.
- [x] 68 receipts, 68 adjacency files and 68 PNG files verified.
- [x] Recorded decisions replayed; topology and homophily recomputed from graphs.
- [x] 204 matched offline controls generated; these are not extra LLM networks.
- [x] All 88 tests and 896 full-roster offline fixture checks passed, plus 12 control fixtures.
- [x] Original six receipts preserved byte-for-byte; 194 cached replies reused without repayment.
- [x] No unresolved ledger attempts. Two retired unknown-billing reservations remain counted.
- [x] Completed within the original $5 additional allowance. No budget increase used.
- [ ] Human bilingual signoff, final design acceptance and main-budget approval.
- [ ] Remaining 828 networks, conditional on keeping this protocol unchanged.

This completes **calibration, inspection and estimates**, not a zero-error
research certification or a guarantee of conference acceptance.

## Coverage

Models: GPT-4.1, GPT-5.6-Luna, GPT-6-Luna and GPT-6-Sol.
Methods: global, local, sequential and iterative. Each graph uses the same 50
fictional adults. Seeds identify randomization, not population size.

Seven settings avoid double-counting the shared baseline: US, India, Japan and
Brazil framing in English; US framing in Hindi, Japanese and Brazilian Portuguese.
Spanish is excluded from this fresh study.

| Calibration scope | Networks |
| --- | ---: |
| GPT-6-Luna: 7 settings x 4 methods x 2 repetitions | 56 |
| Other 3 models: US-English x 4 methods x 1 repetition | 12 |
| **Total** | **68** |

Each method has 17 graphs. This is not multilingual calibration of every model.
The full design is 4 models x 4 methods x 7 settings x 8 repetitions = 896.
Calibration reuse requires unchanged reviewed prompts, protocol and generation
contract; otherwise affected runs need fresh generation, not relabeling.

## Actual Accounting

Amounts are conservative token-usage estimates and retained reservations,
**not an OpenAI invoice**. Input uses 1.25x standard pricing; output uses standard
pricing. Cache discounts are not assumed. Missing responses are never free.

| Item | USD |
| --- | ---: |
| Historical ledger before calibration authorization | 2.018819775 |
| Superseded V1, including retained reservation | 0.207082400 |
| Superseded V2 correction attempts | 0.019651000 |
| Superseded V3 calibration | 0.011369400 |
| Superseded V4 calibration | 1.838349300 |
| Completed V5, including retained failed-request reservation | 2.378072350 |
| **Additional calibration accounting, all versions** | **4.454524450** |
| **Cumulative ledger** | **6.473344225** |
| Original cumulative ceiling | 7.018819775 |
| **Unused original allowance** | **0.545475550** |

Two retired requests retain $0.017300875 total; actual provider billing is unknown.
The V5 lost reply retained $0.000969375 and received one explicitly authorized,
separately charged replacement. Neither original was marked received or refunded.
The recovered graph reused all 194 prior replies.

Logging/recovery compatibility was reviewed against archived wrapper source.
Original generation hashes and actual execution hashes are separately recorded
and checked. Offline SDK tests confirmed equal request JSON and parsed replies.
Prompts, engine, parser, metrics, roster and runtime did not change in recovery.

## Updated Completion Estimate

| Scope | Cost estimate | API-only serial time |
| --- | ---: | ---: |
| Full 896-network design, including these 68 | $103.37 | 30.81 hours |
| **Remaining 828 networks** | **$101.00** | **28.44 hours** |
| Remaining with planning allowances | **$121.20** | **42.66 hours** |
| Cumulative ledger after remaining work, without allowance | $107.47 | Not applicable |
| Cumulative ledger after remaining work, with cost allowance | **$127.67** | Not applicable |

Cost allowance: 20%. Runtime allowance: 50%. A **$128 cumulative ledger ceiling**
would cover this planning scenario, not guarantee completion. It is not yet
authorized. Runtime excludes rendering, local analysis, pauses and manual review;
wall-clock completion can take longer.

| Method | Planned graphs | Full-study cost | API hours |
| --- | ---: | ---: | ---: |
| Global | 224 | $0.91 | 0.30 |
| Local | 224 | $14.46 | 3.98 |
| Sequential | 224 | $15.82 | 3.67 |
| Iterative | 224 | $72.18 | 22.85 |
| **Total** | **896** | **$103.37** | **30.81** |

Measured Luna treatment-to-baseline ratios are transferred to the other models'
measured US-English baselines. This is an assumption, not measured multilingual
performance for all four models. Only two treatment repetitions and one
other-model baseline each are available. Allowances are not confidence intervals.
Rates and official source links are frozen in `calibration_review.json`.

The independent read-only reviewer recomputed the forecast and accounting,
finding no material arithmetic errors. That review did not independently repeat
all graph/control checks or establish scientific validity.

## Inspection Findings

All graphs have exactly 50 roster nodes, no self-loops, nonzero edges and matching
saved/replayed edge sets. Edge counts range from 24 to 227. No metric columns in
the exported per-network table are undefined. All 204 control generation targets
completed; this does not establish rewiring mixing convergence.

Calibration is `COMPLETE`; full-study analysis is correctly `PARTIAL` at 68/896.
These are different scopes, not contradictory statuses.

There were **7,749 received requests and 140 rejected parse attempts (1.81%)**.
Rejected replies remain charged and recorded. All received replies ended with
`stop`. Malformed answers were handled by the bounded correction policy, not
accepted as graph data. No graph exhausted that policy in the completed
calibration. The resumed batch had no further lost transport response.

| Method | Rejected parse attempts |
| --- | ---: |
| Global | 1 |
| Local | 49 |
| Sequential | 28 |
| Iterative | 62 |

Most rejections were cardinality mismatches when exactly one person was required
(127 attempts). Others involved duplicate choices, malformed ID-only output,
self-selection or invalid removal. Passing final graphs do not mean first-prompt
compliance. Correction rates belong in the methods and limitations discussion.

PNG verification checks file integrity and hashes, not visual readability.
Spring-layout coordinates are algorithmic, not geographic or measured social
distance. One-color nodes do not encode demographics; dense clusters can overlap.
These are inspection artifacts, not finished conference figures.

Scientific limits remain: a designed fictional roster is not a national sample;
country framing is a prompt intervention; candidate attributes remain English
in every condition; AI language review is not human bilingual signoff. Eight
repetitions do not establish power or eliminate prompt/model bias. Calibration
does not provide final RQ significance conclusions or empirical realism evidence.

## Evidence and Reproduction

- Receipts and graph/PNG artifacts: `outputs/revision896_retry_v5/`.
- Cost/coverage: `outputs/revision896_retry_v5_preflight/calibration_report.json`.
- Timing/integrity: `stats/revision896_retry_v5/calibration_inspection.json`.
- Metrics, homophily, paired contrasts and controls: `stats/revision896_retry_v5/`.
- Recovery: `source_compatibility.json` and `recovery_replay_audit.json` in the preflight directory.
- Earlier report: `source_snapshot/CALIBRATION_RESULTS_before_recovery.md`.
- Failure history: [CALIBRATION_FINDINGS.md](CALIBRATION_FINDINGS.md).

Run from the repository root; both commands are offline:

```powershell
rtk proxy .\.venv\Scripts\python.exe -B -m unittest discover -s tests
rtk proxy .\.venv\Scripts\python.exe -B inspect_calibration.py
```

The notebook's fresh-study section loads these reports and checks preserved
generation and actual execution hashes. It never substitutes historical graphs
or calls OpenAI.

Notebook JSON validation and syntax checks for all 42 code cells passed. The
fresh-study data-loading cell executed offline and showed 68 verified graphs,
zero unresolved requests and the completed forecasts. Earlier historical cells
were not rerun during this calibration closeout.

**Next decision:** review these estimates and scientific limitations, then
explicitly approve a finite main-study ceiling. No further paid generation is
authorized by this calibration report.
