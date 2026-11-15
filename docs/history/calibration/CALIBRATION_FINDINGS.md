# Calibration Stopped: Partial Global Correction

Date: 2026-09-30. This preserves the initial failure and subsequent repair history,
not a completed main study. The accounting table is the first-stop snapshot.
The latest inspection and spending totals are in
[CALIBRATION_RESULTS.md](CALIBRATION_RESULTS.md): v5 is paused at six graphs after
an uncertain connection failure, not marked complete.

## Observed Failure

Model access succeeded for all four configured models. Four US-English global
graphs were saved. GPT-4.1's first response contained 71 lines representing
67 unique friendships, including four reversed duplicates. The retry returned
only the four corrected pairs, omitting 63 valid friendships. The old parser
accepted it because it checked individual response validity, not completeness
relative to the original answer.

The run was stopped during the following GPT-4.1 iterative graph. That partial
graph has request-cache evidence but no completed-network receipt.

## Evidence and Exclusion

- [x] Original replies and charges remain in `outputs/revision_budget_v1/budget.sqlite`.
- [x] Original four adjacency/PNG/receipt sets remain unchanged in `outputs/revision896/`.
- [x] The four-edge GPT-4.1 graph is excluded as a known partial-correction failure.
- [x] All four initial graphs are excluded from the revised study's pooled results
  because the frozen correction protocol changed. The other three passed their
  structural receipt checks; that does not validate the flawed retry policy.
- [x] Original calibration authorization was revoked, not silently reused.
- [x] Each corrected execution uses a separate versioned namespace. V2 and v3
  evidence is retained; current execution uses `outputs/revision896_retry_v5/`,
  `outputs/revision896_retry_v5_preflight/` and `stats/revision896_retry_v5/`.

Never replace the four-edge artifact with an offline deduplicated graph and call
it a new model result. The full 67-edge reconstruction was used only as an
in-memory regression fixture; no fabricated model output was saved.

## Fix and Verification

- [x] All four fresh languages now explicitly request the entire corrected
  network, preserve valid original ties and forbid new ties. Legacy Spanish
  receives the same protection but is not a fresh study treatment.
- [x] The first rejected global response defines the correction target using
  only complete pair lines with distinct, exact roster IDs. No numbers are
  extracted from prose. Unknown IDs, self-links and malformed lines are excluded.
- [x] Retries must match that canonical undirected edge set exactly before any
  graph mutation. The target cannot shrink after another invalid correction.
- [x] If a nonempty answer has no recoverable valid pairs, stop rather than
  resample. V4 separately handles a wholly empty initial answer as nonresponse.
- [x] Rejections remain inside the paid parse wrapper so failures are recorded.
- [x] An unresolved request anywhere in the shared ledger blocks new execution
  before API-client creation, including execution under a different protocol.
- [x] 72 offline tests passed. The actual 71-line/four-edge failure was replayed
  without API calls: the partial answer was rejected and the complete 67-edge
  fixture passed on attempt three.
- [x] All 896 corrected-protocol full-roster fixture cells and 12 matched-control
  fixtures passed. Independent code and AI language reviewers closed the bounded
  fix review. These checks do not imply successful live v2 calibration.

This deliberately defines retries as **format repair**, not independent network
resampling. Calibration must assess how often the models satisfy that contract.
Passing software tests cannot guarantee future model compliance.

## Spend and Restart Boundary

| Item | USD |
| --- | ---: |
| Historical ledger before this calibration | 2.018819775 |
| New received-request conservative charges (96 requests) | 0.190750900 |
| Interrupted request reservation, billing unresolved | 0.016331500 |
| New usage plus retained reservation | 0.207082400 |
| Cumulative ledger at the first stop | 2.225902175 |
| Previously authorized cumulative calibration ceiling | 7.018819775 |
| Remaining allowance after retained reservation | 4.792917600 |

These are conservative usage/reservation figures, not a verified provider invoice.
The interrupted request was originally reserved under ID
`cb5b221590a54c3bb1938e9c5c9c3f72c9b70d3825bd07d1c11fc8bfd5027e49`.
Its provider billing remains unverified. On the author's instruction to continue
bounded calibration, it was explicitly abandoned while retaining its full
$0.016331500 reservation permanently. It cannot be replayed or subsequently
settled/refunded by the runner. No provider response or usage was invented.
Do not delete that charge, reset the ledger or grant another $5 on restart.

- [x] Resolve the execution block by explicit abandonment with the full reserved
  charge retained; this is not provider-billing reconciliation.
- [x] Approve corrected, hash-bound calibration receipts using the original ceiling.
- [ ] Complete and inspect bounded v5 calibration; see the current calibration report.
- [ ] Inspect actual language/method behavior, costs and variability before choosing six versus eight repetitions.
- [ ] Separately approve any main study. The provisional target remains 896, not a completed result.

## Subsequent Live Findings

| Version | Observed outcome | Additional conservative USD |
| --- | --- | ---: |
| V2 | An illustrative retry pair was copied into the answer; exact-set guard rejected it. Three requests, no completed graph. | 0.019651000 |
| V3 | Two global graphs completed; GPT-6-Luna then returned an empty answer with `stop` finish reason. Collection stopped. | 0.011369400 |
| V4 | Thirty graphs completed and verified. A subsequent iterative initialization returned 128 tokens with `finish_reason=length`; collection correctly stopped. | 1.838349300 |

V2 removed the original missing-tie acceptance defect but revealed example-ID
contamination. V3 removed those illustrative IDs. V4 adds a bounded initial-empty
retry without weakening full-set repair. V1, v2 and v3 evidence stays separate
and excluded from the current calibration, even when an individual graph passed.

The ledger before v4 was **$2.256922575**, including all previous attempts and
the abandoned reservation. The remaining calibration allowance was **$4.7618972**
under the unchanged **$7.018819775** cumulative ceiling. These figures are
usage-based conservative accounting, not an invoice. Current v5 charges are
reported separately after inspection.

V4's blanket 128-token per-person limit could not accommodate every legal list
of up to 49 peers. The failing answer was not accepted or retried silently. V5
uses 512 tokens for per-person responses; a capacity/cache regression failed
before the fix and passes for all four methods afterward. The global cap remains
8192, enough for the compact 1225-pair ASCII schema. Truncation still stops rather
than accepting partial data. V4's 30 graphs and 90 offline controls are retained
as superseded engineering evidence, not pooled into v5.

The v5 starting ledger was **$4.095271875**, leaving **$2.9235479** within the
same original calibration ceiling. No new $5 allowance was granted.

The accounting repair also includes a transactional guard: settlement cannot
race with explicit abandonment and erase a retained reservation. A two-connection
regression reproduces that interleaving. The final offline suite currently has
79 passing tests, including retry, capacity, budget and timing checks; passing tests never
substitute for inspecting live evidence.
