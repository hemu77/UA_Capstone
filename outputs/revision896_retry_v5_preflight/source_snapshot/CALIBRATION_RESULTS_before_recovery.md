# Calibration Inspection and Cost/Runtime Estimate

Inspection date: 2026-09-30. **Main study not authorized. Calibration incomplete.**

## Decision Summary

- [x] Repaired complete-network retry handling, example-ID contamination,
  initially empty replies, reservation settlement races and response capacity.
- [x] 79 offline tests pass; 896 full-roster fixture cells and 12 control fixtures pass.
- [x] Inspected the current six saved 50-person graphs against their receipts,
  adjacency files, replay events, recomputed metrics, image hashes and ledger.
- [x] Generated 18 matched offline controls for those six graphs, without API calls.
- [x] Preserved every superseded result and all charges. No old graph was relabelled.
- [ ] Complete all 68 current-protocol calibration cells: only **6/68** are complete.
- [ ] Resolve the interrupted request before any further paid call.
- [ ] Accept translation/design evidence and approve a finite main-study ceiling.
- [ ] Generate the 896-network study. It has **not** been started as a main batch.

The latest run stopped on an `APIConnectionError` during GPT-6-Luna's US-English
iterative graph, at request ordinal 194. The provider may have received it, so its
full **$0.000969375** reservation remains `uncertain`; it was not refunded, marked
successful or resent. Request ID:
`659cc446cedb665c4735858c9bd74d243ed79ecf4a076cbb3f5a9574a8c2d9a9`.
The calibration receipt is now disabled. The main `review.json` does not exist.

This is a blocked paid collection, not evidence of a completed calibration or
conference readiness. The rest of this report completes the requested inspection
and provides provisional planning numbers without hiding that limitation.

## Actual Spend

Amounts are conservative usage/reservation accounting, not a provider invoice.
Input is charged at 1.25 times the listed rate; cache discounts are not assumed.

| Item | USD |
| --- | ---: |
| Historical ledger before this calibration authorization | 2.018819775 |
| V1 received usage plus permanently retained interrupted reservation | 0.207082400 |
| V2 failed correction attempts | 0.019651000 |
| V3 usage before the empty-response stop | 0.011369400 |
| V4 usage before the 128-token truncation stop | 1.838349300 |
| V5 current usage plus uncertain reservation | 0.679995675 |
| **Additional calibration spend/reservations** | **2.756447775** |
| **Cumulative ledger** | **4.775267550** |
| Original authorized cumulative ceiling | 7.018819775 |
| Unused calibration allowance | 2.243552225 |

The $5 additional allowance was never reset between versions. The earlier
$0.016331500 reservation is permanently retained as explicitly abandoned,
separate from the new uncertain request above. Neither represents verified usage.

## Provisional 896-Network Estimate

**Baseline-only scenario: about $101 in generation costs; about $121 with a 20%
planning allowance. Serial API time is about 25 hours; allow roughly 37 API hours
with a 50% timing allowance, plus local processing, rendering, analysis and pauses.**

These are not a completed-calibration forecast, a price quote, a spending
authorization or a deadline. They use the 16 verified US-English model/method
baselines collected under the superseded v4 protocol, because v5 does not yet
cover all 16 combinations. Those graphs remain engineering evidence, not v5
research results. No reuse credit is assumed in this 896-network scenario.

| Method | Planned graphs | Baseline-only USD | Serial API hours |
| --- | ---: | ---: | ---: |
| Global | 224 | 1.30 | 0.28 |
| Local | 224 | 13.62 | 3.02 |
| Sequential | 224 | 15.21 | 3.13 |
| Iterative | 224 | 70.89 | 18.45 |
| **Total** | **896** | **101.02** | **24.89** |

Calculation: each of the 16 model/method US-English baselines is multiplied by
seven settings and eight repetitions, or 56. Exact totals are $101.0206008 and
24.8873843 API hours. Applying the planning allowances gives $121.224721 and
37.3310764 API hours. Existing $4.77526755 spending is separate; adding it to the
no-reuse scenarios gives approximately $105.80 or $126.00 cumulative accounting.
Any further failed/restarted calibration not represented here would add cost.

Important assumptions:

- Country/language settings are assumed to cost and run like US-English. This
  is unvalidated; token lengths, graph density and retry rates can change.
- V5 raised per-person output capacity from 128 to 512 tokens. Longer valid
  answers can increase usage, graph density and subsequent iterative input cost.
- Iterative processing dominates cost and time. It initializes a local graph and
  performs three rounds of per-person updates; graph counts alone hide this work.
- Measured timing sums request intervals, excluding idle time between requests.
  Network delays within a request remain included. Local processing, rendering,
  controls and operator recovery add wall-clock time.
- A single US-English observation for most models is not a stable rate estimate.
  The percentage allowances are planning choices, not statistical confidence bounds.
- Prices are the source-bound September 30 rates recorded in the calibration
  receipt. Alias names and returned model IDs do not guarantee immutable weights.

The automated current-protocol cost/runtime forecasts intentionally remain
`null`. Only a complete 68-cell calibration can populate them. The provisional
scenario above must not bypass that gate.

A final independent read-only check reproduced the accounting and scenario
arithmetic and confirmed the disabled calibration gate and absent main approval.
It reported no material findings within that bounded scope. This is not a
whole-project or scientific-acceptance certificate.

## Graph and Analysis Inspection

Current v5 saved graphs:

| Model | Method | Nodes | Edges | Paid requests | Rejected parse attempts |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-4.1 | Global | 50 | 83 | 2 | 1 |
| GPT-5.6-Luna | Global | 50 | 122 | 1 | 0 |
| GPT-6-Luna | Global | 50 | 64 | 1 | 0 |
| GPT-6-Sol | Global | 50 | 85 | 1 | 0 |
| GPT-4.1 | Iterative | 50 | 219 | 343 | 1 |
| GPT-5.6-Luna | Iterative | 50 | 224 | 345 | 0 |

Every current saved graph is US-English, repetition 0. The incomplete Luna
iterative graph is not exported as a completed result. All completed receipt
requests have normal finish reasons; the uncertain request remains separate.
Two rejected parse attempts were repaired and are still charged and recorded.

There are 42 saved graph artifact sets across the debugging versions: four in
v1, two in v3, thirty in v4 and six in v5. **Only the six v5 graphs belong to the
current protocol.** Superseded sets include the known invalid v1 partial-repair
graph and must not be presented as 42 validated current-study networks.

V4 covered global prompts in all seven settings, including Portuguese, but its
30 graphs are excluded after the capacity change. V5 does not yet establish
live multilingual or all-method coverage. Software fixture coverage is broader
than live evidence and is explicitly labelled as such.

Graph checks recompute topology and homophily, replay saved operations to recover
the final edge set, check exact roster IDs and compare file hashes. PNG decoding
checks file integrity, not scientific validity. The plain topology figures use
algorithmic spring positions, not geographic locations or measured social
distance; their one-color nodes do not encode demographics. Dense regions can
overlap, so adjacency/metrics are the analysis authority, not visual counting.
Visual spot checks of the v5 GPT-5.6-Luna global and GPT-4.1 iterative PNGs found
readable files but substantial node overlap in the iterative clusters. They are
inspection artifacts, not publication-ready figures. All six graphs have finite
density, average clustering and largest-component proportion measurements.

Scientific acceptance remains separate: the fictional roster is not a national
sample; country framing is a prompt intervention; English attributes remain
fixed in every instruction language; AI language review is not human bilingual
signoff. Eight repetitions do not establish power or eliminate model/prompt bias.
No significance or population-level RQ conclusions are supported by this partial
calibration. Retain eight as a provisional target, not as a proven optimal count.

## Evidence and Reproduction

- Current receipts and graph/PNG artifacts: `outputs/revision896_retry_v5/`.
- Current actual usage: `outputs/revision896_retry_v5_preflight/calibration_report.json`.
- Current verified analysis and 18 controls: `stats/revision896_retry_v5/`.
- Current timing/integrity inspection: `stats/revision896_retry_v5/calibration_inspection.json`.
- Superseded planning measurements: `stats/revision896_retry_v4/calibration_inspection.json`.
- Protocol/source hashes and approved ceiling: v5 `calibration_review.json`.
- Detailed failure history: [CALIBRATION_FINDINGS.md](CALIBRATION_FINDINGS.md).

Offline verification, from the repository root:

Notebook JSON and all 42 code-cell syntax checks passed. The fresh-study
data-loading cell executed offline and displayed six graphs, partial analysis,
one unresolved request and no automated forecast. Legacy notebook sections were
not rerun as part of this calibration inspection.

```powershell
rtk proxy .\.venv\Scripts\python.exe -B -m unittest discover -s tests
rtk proxy .\.venv\Scripts\python.exe -B inspect_calibration.py
```

Reproduce the provisional scenario without API calls:

```python
import json
from pathlib import Path
r = json.loads(Path('stats/revision896_retry_v4/calibration_inspection.json').read_text())
rows = [x for x in r['networks'] if x['culture'] == 'us'
        and x['language'] == 'english' and x['repetition'] == 0]
assert len(rows) == 16
print(sum(x['conservative_charge_usd'] for x in rows) * 56)
print(sum(x['request_roundtrip_seconds'] for x in rows) * 56 / 3600)
```

**Next decision:** reconcile the uncertain request and review a bounded recovery
plan. Do not authorize or launch 896 on the assumption that calibration passed.
