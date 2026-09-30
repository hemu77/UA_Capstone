# Eight-Repetition Study: Calibration First

## Calibration Status

**Complete and inspected.** V5 has 68 verified graphs and 204 offline controls.
The reviewed transport recovery retained the failed request's full reservation
and reused 194 cached replies. No further transport failure occurred. Additional
charges and reservations total $4.454524450 of the original $5 allowance.
[CALIBRATION_RESULTS.md](CALIBRATION_RESULTS.md) records exact evidence and the
current-calibration forecast: $103.37 for the full 896 design; $101.00 remaining
for 828 graphs, or $121.20 with 20% allowance. Remaining serial API time is
28.44 hours, or 42.66 with 50% allowance, excluding offline work and pauses.
These estimates transfer Luna treatment ratios to other models; they are not
guarantees. Main generation remains unapproved.

Live checks exposed retry-completeness, example contamination, empty-response and
output-capacity issues; see [CALIBRATION_FINDINGS.md](CALIBRATION_FINDINGS.md).
Superseded protocols remain preserved and excluded. V5 uses 8192 completion
tokens for global responses and 512 for per-person responses, sufficient for the
declared 50-person numeric reply schema. Abnormal finishes still stop collection.
The interrupted original request was explicitly abandoned with its full charge
retained, not refunded or falsely marked received. Current v5 calibration remains
under the original **$7.018819775 cumulative ceiling**, including every earlier
attempt. There is no main-study approval.

## Current Decision

The author selected all four models, all four methods, all seven settings and
eight generations per condition: **896 networks total**, each with 50 fictional
adults. The historical $50 maximum is removed for the fresh study. This is not
permission for unlimited spending: calibration is authorized for **$5 additional
usage only**, and the main batch requires a separate finite approved ceiling.

The author reports Astra reviewed the preceding implementation. Changes for
this allocation have their own source hashes and bounded independent reviews;
the old review cannot authorize an edited protocol automatically.

## Exact Scope

| Dimension | Values |
| --- | --- |
| Models | GPT-4.1, GPT-5.6-Luna, GPT-6-Luna, GPT-6-Sol |
| Methods | Global, local, sequential, iterative |
| Country framing | US, India, Japan, Brazil |
| Instruction language | English, Hindi, Japanese, Brazilian Portuguese |
| Conditions | Four countries in English; US in three additional languages |
| Repetitions | Eight per model/method/condition; seeds 11000-11007 |
| Roster | Same declared 50 fictional adults, ages 18-67 |

US-English is shared, not counted twice. Thus 4 x 4 x 7 x 8 = 896,
or 224 graphs per model and 224 per method. Seeds control local randomization,
not population size or guaranteed provider determinism.

## What Calibration Does

The frozen first-stage selection contains **68 graphs**, not 68 extra graphs:

- GPT-6-Luna: all four methods and seven settings, twice = 56.
- Other three models: each method once, US-English = 12.

Those graphs are provisionally reusable within the 896 only if the reviewed
protocol, roster, prompts and model settings stay unchanged. If translation or
engineering review changes them, preserve the calibration evidence separately
and regenerate under a new frozen protocol. Never relabel old runs as new data.

Calibration checks actual model access, prompt handling, token use, parsing,
truncation, graph invariants and recorded replay. Two Luna runs expose initial
variation but do not establish power or reliable interval precision. The other
models are calibrated only in US-English; their multilingual behavior remains
unmeasured until separately tested or collected. These gaps remain explicit.

The offline calibration report requires all 68 receipts and no unresolved
charges before providing a completion forecast. For other models it transfers
Luna's method/condition cost ratios onto their measured US-English costs. This
is an assumption, not measured treatment cost for those other models. The report
retains raw token totals, failures, usage and actual conservative charges.

## Safety and Analysis

- [x] New protocol: `study_protocol_896.json`; target is 896, not 224.
- [x] Eight unique repetitions per condition; full 50-person roster preserved.
- [x] New output namespace keeps older results and 224 planning evidence intact.
- [x] Separate calibration/main receipts bind scope, source, roster and runtime.
- [x] Calibration ceiling is frozen starting ledger total plus $5. Restarting
  must reuse that same ceiling, not grant another $5.
- [x] The original ledger retains prior charges and uncertain reservations.
  Shared legacy pilot commands retain their old $50 protection.
- [x] A matching complete calibration report and explicit code/price/translation/
  cost/design review gates are required before the main study.
- [x] Raw paired differences remain available. Summary tables show observed and
  defined pair counts, mean, sample standard deviation and range, explicitly
  compared with the target of eight. Missing values are not silently zeroed.
- [x] No significance declarations or independent-edge confidence intervals are
  produced. Eight repetitions are a chosen allocation, not proven sufficiency.
- [x] Live credentials were supplied process-only, never saved in the repository.
- [x] Inspect every currently completed graph, actual usage and partial analysis.
- [x] Complete and inspect all 68 calibration cells; 204 offline controls generated.
- [x] Complete independent AI-assisted instruction/retry review; findings and
  limits are in [the translation review](TRANSLATION_REVIEW.md).
- [ ] Obtain human bilingual signoff or explicitly document the accepted review
  standard before approving main-study translation acceptance.
- [ ] Justify required effect sensitivity/precision or retain exploratory claims.
- [ ] Approve a finite main-study ceiling after inspecting the calibration report.

Removing US party names does not make this roster representative. Country
framing studies a prompt intervention, not national behavior. Repetitions do
not replace multiple rosters, empirical benchmarks, prompt/temperature sweeps
or method ablations. Those concerns from `reviews.pdf` remain outside what
this allocation alone can establish. Software checks cannot guarantee acceptance.

## Offline Verification Record

- [x] 88 tests passed, including full-roster response capacity, cache reuse,
  atomic reservation abandonment, global retry completeness, cross-protocol unresolved
  request blocking, localized retry exports, common-neighbor semantics,
  finite budget overrides, calibration/main scope
  isolation, missing calibration receipts, corrected forecasts and zero-count
  contrast coverage. Existing NumPy/pandas deprecation warnings are nonfatal.
- [x] 896 full-roster fixture checks and 12 matched-control fixtures passed.
- [x] Both read-only reviewers reported no unresolved material findings in
  their final changed-path reviews. These are bounded reviews, not guarantees.
- [x] The now-disabled v5 calibration-only receipt binds corrected hashes and retains the
  cumulative ceiling of $7.018819775: original $2.018819775 plus the approved $5,
  not a reset allowance. No main approval exists.
- [x] Offline reports distinguish absent, partial and complete calibration;
  main-study analysis remains partial until all 896 verified graphs exist.
- [x] Notebook JSON and 42 code-cell syntax checks passed. The fresh section
  follows current runner paths and labels historical data separately.
- [x] Offline fix verification made no API calls. The preceding live attempt's
  actual charges and unresolved reservation are reported separately above.

## Files and Commands

`revision224.py` and `revision224_prompts.py` retain their filenames for import
and command compatibility. The former now reads `study_protocol_896.json`.
The roster's `revision224_adults.json` filename is historical; it still contains
50 adults, and its construction seed remains 224. Neither filename sets run count.

New paths:

- `outputs/revision896_retry_v5_preflight/`: manifest, prompts, checks and review receipts.
- `outputs/revision896_retry_v5/`: corrected-protocol adjacency, PNG, events and usage receipts.
- `stats/revision896_retry_v5/`: metrics, group homophily, controls and paired summaries.

Run from the repository root in PowerShell:

Ledger-backed analysis requires the original local billing ledger, which is
intentionally not published. A public clone can inspect saved reports and run
offline fixture tests, but must not create a replacement empty ledger to resume.
The frozen protocol's preparation-time `status` and `cost_review` fields remain
unchanged for hash integrity; the completion report records the observed current
calibration status. Do not alter a frozen protocol merely to update a status label.

```powershell
# Offline: no API calls.
rtk proxy .\.venv\Scripts\python.exe -B -m unittest discover -s tests
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --preflight
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --calibration-report
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --analyze

# Live calibration only, after a hash-matching calibration_review.json and key.
# CURRENTLY BLOCKED: all 68 calibration cells are complete; authorization closed.
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --calibrate

# Offline after calibration; inspect both reports and saved artifacts.
rtk proxy .\.venv\Scripts\python.exe -B inspect_calibration.py

# NOT authorized by calibration: requires separate review.json and main approval.
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --execute --limit 896
```

Do not fill review flags merely to bypass a gate. The calibration report and
analysis do not create an API client. `NOT_COLLECTED` is an honest empty state,
not a request to populate the notebook with fixtures or historical results.
