# Pre-Launch Review: Final Collection Contract v4

**Date:** October 19, 2026
**Branch reviewed:** `r1/social-netw` at `967d9f1` (6 commits after the
[October 17 review](PR_REVIEW_OCT17.md)), plus `99cebbc` (Section 9)
**Contracts reviewed:** v4 `final-instruction-study-v4-budget120-restart-safe`
(`be0e2e57…`), and v5 `final-instruction-study-v5-budget120-local-write-retries`
(`17c2750fe36a368150c73acc1fd080696684f273b5aa2c446d65695210e6d9c4`). **Every finding below
applies unchanged to v5**, which only adds local file-write retries; the fixes recommended
here would now go into a v6 contract.
**Responds to:** [Final collection](FINAL_COLLECTION.md),
[verification](FINAL_COLLECTION_VERIFICATION.md), [October 18 launch review](PR_REVIEW_OCT18.md),
[pre-collection review](PRECOLLECTION_REVIEW.md)
**Reviewer:** Claude (AI coding assistant), requested by Raghav. This is an independent,
evidence-based pre-launch review of a frozen design and its implementation. It is **not**
professor approval, human bilingual validation, or authorization to spend.

---

## 0. Verdict

**The engineering is launch-quality. The design has three defects that should be fixed
before the first paid request, because none of them can be fixed afterwards.** All three are
cheap, and no contract has been attempted, so the current one (v5) can still be superseded.

| # | Finding | Severity | Fix before launch |
| --- | --- | --- | --- |
| **B1** | **A stopped run destroys almost all primary inference.** All 896 cells run in one global random order across all eight seeds, and any missing cell suppresses its whole contrast. Stopping at 90% for any reason (cap, price expiry, a 31st outage, a model retirement) leaves a given contrast complete with probability ≈ 0.9⁶⁴ ≈ 0.001. | **Blocker** | Run in **block-major order**: finish seed 44000's 112 cells (in random order), then 44001, and so on. Same protection against drift, and a 90% stop still leaves 7 complete blocks. |
| **B2** | **Even a complete run will probably lose global primaries.** At the best available estimate of global replies that stay invalid under the new parser (1/106 ≈ 0.9%), there is only about a **12% chance all six global primaries survive** the complete-block rule. There is a **26% chance that one failed US-English global cell suppresses all six at once.** | **Blocker** | Predeclare a block-level fallback: drop an incomplete schedule block from that contrast (the four-model panel stays intact), and infer if at least 6 of 8 blocks remain. Keep the current bounds as sensitivity analysis. |
| **B3** | **The launch deadline in the docs is wrong.** "Launch by October 25" cannot work: the price window is checked before **every** request, ends October 25, and the run needs 26–40 serial API hours. A run still going on October 26 stops permanently, because new prices mean a new contract hash and a new output folder. | **Blocker** | Launch no later than the morning of **October 23**, on a machine that will not sleep. Or check prices once at authorization rather than per request, since the ledger enforces the $120 cap either way. |
| **B4** | **The exposed API key is still not confirmed rotated.** This is the third review to raise it. | **Blocker (security)** | Rotate it in the provider dashboard and record the date (not the key) in the verification doc. |
| R1 | **Expected results are mostly null, and the docs should say so before data exist.** Projected from calibration, the design has high power for at most 1–2 of the 24 primary hypotheses. Global density is essentially unpowered (simultaneous half-width ≈ 0.06). | High | Write the expected precision into the plan, so that nulls are reported as "not resolvable at this precision", not as "no effect". |
| R2 | **Global precision is nearly free.** One global schedule block costs **$0.09**; one iterative block costs **$9.08**. Raising global alone from 8 to 32 blocks costs **+$2.07** and narrows its intervals about 3×. | High | Owner's decision. The trade-off is quantified in Section 4. |
| R3 | The frozen contract contradicts itself on transport recovery. | Medium | One-line edit in the v6 contract (Section 6). |
| N1 | **New in `99cebbc`:** local write retries are correct and safe, but a lock that outlasts them (about 1.9 s) still ends in a `Journal/ledger mismatch` stop with **no supported repair command**. V6 and V8 each needed a one-off script for exactly this state. | Medium | Add a narrow, reviewed journal reconciliation and lengthen the backoff (Section 9). |
| R4 | No professor sign-off on the analysis plan is recorded; the handoff goes straight to "the owner's launch instruction". | Medium | Get it recorded in `TEAM_APPROVAL.md` before launch, or state in the paper that the plan was not externally approved. |

Minor items are in Section 7. What the team did well is in Section 8; it is a long list.

---

## 1. What I Verified Independently

All checks ran on this machine with no API key in the environment and no network access to
the provider. Scripts are in Appendix A.

| Check | Result |
| --- | --- |
| **Fresh clone** of `r1/social-netw` at `967d9f1`, full offline suite (Python 3.11.9, `requirements.lock.txt`) | **244/244 passed** in 331.8 s |
| `python -m final_study preflight` in the fresh clone | `OFFLINE_PREFLIGHT_PASSED`, **112 cases**, contract `be0e2e57…` (matches the docs), 0 paid requests |
| `python -m final_study verify --stage hindi_check` in the fresh clone | `COMPLETE`, 8/8 valid, 1 empty, $0.0274 (matches the docs) |
| The five files with 156–805 changed lines each (`bias.py`, `generate_personas.py`, `network_datasets.py`, `plotting.py`, `repair_outputs.py`) | **Line endings only**: zero changes under `--ignore-cr-at-eol`. Matches the team's explanation. |
| Analysis math (`precollection_analysis.primary_contrasts`) | Correct: paired block differences, Holm over all 24 with suppressed tests kept at p = 1, Bonferroni simultaneous intervals with t(n−1), correct missingness bounds (min target − max baseline). |
| Execution order (`protocol.cells`) | One `random.Random(44032).shuffle` over all 896 cells, across seeds (B1). |
| Price check (`Caller.__call__` → `check_prices`) | Runs before every paid request; window 2026-10-18 to 2026-10-25 inclusive, local date (B3). |
| Global parser (`precollection_candidate.canonical_global`) | Sound: validates every line first; rejects bad IDs, self-links and malformed lines for the whole reply; collapses only exact undirected duplicates; keeps raw text and flags. |
| Hindi global wording | Fixed. Reads "a friendship configuration that is realistic and believable" (Section 7). |
| Seed quota totals, 44000–44007 | 288, 205, 271, 294, 226, 204, 305, 232. Matches the October 18 review's 204–305. |

**Not verified:** private-ledger accounting and `status` with a credential (both need the
owner's private files, by design); provider billing; human translation quality.

---

## 2. B1: Execution Order × Complete-Block Rule

### 2.1 The mechanism

`protocol.cells()` shuffles all 896 planned cells together:

```python
random.Random(spec['execution_order_seed']).shuffle(result)   # protocol.py line 146
```

The analysis requires all 64 cells of a contrast (4 models × 2 arms × 8 seeds) before it
computes any inference for that hypothesis:

```python
if count == n:          # precollection_analysis.py: every one of the 8 blocks complete
```

Under a global shuffle, the cells of any one contrast are spread uniformly through the run. If
the run stops after a fraction *f* of cells, the chance that a given contrast is complete is
about *f*⁶⁴.

*(computed)*

| Run stops at | P(a given contrast complete), global shuffle | Complete blocks, block-major order |
| ---: | ---: | ---: |
| 50% | 5 × 10⁻²⁰ | 4 / 8 |
| 80% | 6 × 10⁻⁷ | 6 / 8 |
| 90% | 1 × 10⁻³ | 7 / 8 |
| 95% | 4 × 10⁻² | 7 / 8 |

### 2.2 Why a stop is realistic

The docs correctly say that "full collection is the target, but future provider behavior can
exhaust the cap." A run can stop at the $120 cap, at price expiry (B3), on a 31st transport
recovery, on a non-retryable provider error, or on a received-but-unreadable reply that
needs human review. Each of these is designed to stop rather than guess, which is right. But the
current ordering turns any of them into the loss of nearly the whole confirmatory analysis,
even after spending most of the $120.

### 2.3 The fix

Run the eight schedule blocks one after another, with a random order **within** each block:

```python
# sketch: same seed, same cells, different order
rng = random.Random(spec['execution_order_seed'])
ordered = []
for seed in spec['analysis']['schedule_seeds']:
    block = [c for c in result if c['seed'] == seed]
    rng.shuffle(block)
    ordered += block
```

Why this is at least as good as the current order:

- **Drift protection is preserved.** Provider-side change over the ~30-hour run now lines up
  with blocks, and the paired, schedule-blocked analysis removes block effects. Time drift
  becomes part of the block effect it already removes.
- **A stop leaves complete blocks**, which a k-of-8 rule (B2) can analyse.
- **Nothing else changes**: same 896 cells, same seeds, prompts, parser and cap.

It needs a new contract (v6, since v5 now exists). The `freeze` path already allows superseding
a contract with no main attempts; `99cebbc` shows the pattern (add the v5 hash to the unexecuted set).

---

## 3. B2: The Complete-Block Rule Is Too Fragile for Global

### 3.1 How likely global suppression is

Global replies are terminal on the first invalid answer (no correction request), unlike
local decisions, which get three corrections. Under the adopted parser, the historical
evidence for global replies is:

| Source | Global replies | Still invalid under the new parser |
| --- | ---: | ---: |
| V7 clause screen | 64 | 0 |
| V7 task extension | 8 | 0 (1 duplicate, now collapsed) |
| V8 calibration | 26 | 1 (ID `50` in a 0–49 roster; 3 duplicates now collapsed) |
| Hindi live check | 8 | 0 |
| **Total** | **106** | **1 (0.9%; 95% CI 0.02%–5.1%)** |

*(computed)* The probability that a given set of global cells is all valid:

| Invalid rate p | One contrast (64 cells) | All 6 global primaries (224 cells) | US-English baseline (32 cells) |
| ---: | ---: | ---: | ---: |
| 0.2% | 0.88 | 0.64 | 0.94 |
| 0.5% | 0.73 | 0.33 | 0.85 |
| **0.9% (best estimate)** | **0.55** | **0.12** | **0.74** |
| 2.0% | 0.27 | 0.01 | 0.52 |
| 5.0% | 0.04 | 0.00 | 0.19 |

Every global contrast shares the US-English baseline, so a single invalid US-English global
reply removes **all six** global primary hypotheses at once. At the best estimate, that happens
about one time in four. The confidence interval is wide, which is exactly why the outcome should
not be left to chance.

For the quota methods the risk is lower: there have been no terminal failures under the
reference wording in 188 historical cells. But 0/188 still allows a true rate of up to about 2%.

### 3.2 Why the current fallback does not help

The missingness bounds replace each missing endpoint with 0 and 1. For global density, whose
typical values are about 0.03–0.10, one missing model-cell widens that block's bound by 0.25.
That is wider than any plausible effect, so the bounds report "unknown" rather than an estimate.

### 3.3 The fix (predeclare one, before launch)

- **Option 1, block-level complete case (recommended).** If a schedule block is incomplete for a
  contrast, drop that **whole block** (all four models, both arms) from that contrast only. Run the
  same paired t and Holm procedure on the remaining blocks if at least 6 of 8 remain; otherwise
  suppress, as now. This keeps the fixed four-model panel, which was the reason for the
  no-three-model-average rule, and it keeps every failure visible in the denominators. Report the
  current bounds alongside as the sensitivity analysis, and report failure counts by condition so
  readers can judge whether failures were informative.
- **Option 2, one global correction request** for replies that fail validation (V6 had preserving
  global corrections; V8 removed them). This lowers the failure rate but does not remove the
  structural risk, so it complements Option 1 rather than replacing it.

Either option is a prospective rule, written before any main data exist, so it is not outcome
selection.

---

## 4. R1/R2: What Eight Blocks Can Detect, and What More Would Cost

### 4.1 Expected precision

The adopted analysis averages the four models within each schedule and compares conditions within
schedule. With 8 blocks and Bonferroni over 24, the simultaneous 95% half-width is
**1.679 × SD(block difference)**. This matches the October 18 review's "about 1.7×".

I estimated SD(block difference) two ways *(computed)*:

- **Directly:** seeds 21000 (V6) and 33000 (V8) have all four models in the four US-language
  conditions, so model-averaged RQ4 block differences can be formed (3 degrees of freedom).
- **Bottom-up:** per-graph noise from GPT-6-Luna's repeated cells, assuming the four models are
  independent and alike.

| Method | Primary endpoint | SD(block diff), direct | SD(block diff), bottom-up | Simultaneous half-width at 8 blocks | Calibration contrasts (absolute values) |
| --- | --- | ---: | ---: | ---: | --- |
| global | density | n/a (1 difference) | 0.036 | **0.061** | Portuguese 0.016 |
| local | clustering | 0.009 | 0.028 | 0.015–0.048 | Hindi 0.008, Japanese 0.034, Portuguese 0.008 |
| sequential | clustering | 0.024 | 0.025 | 0.041–0.042 | Hindi 0.017, Japanese 0.009, Portuguese 0.045 |
| iterative | clustering | 0.038 | 0.033 | 0.056–0.063 | Hindi 0.010, Japanese 0.047, Portuguese 0.018 |

*(computed)* Power at the first Holm step (α/24, 8 blocks) to detect an effect equal to each
calibration contrast:

| Contrast | Power, conservative SD | Power, optimistic SD |
| --- | ---: | ---: |
| local Japanese (0.034) | 0.23 | **1.00** |
| sequential Portuguese (0.045) | 0.63 | 0.67 |
| iterative Japanese (0.047) | 0.26 | 0.37 |
| all other contrasts (0.008–0.018) | 0.01–0.04 | 0.01–0.10 |
| global Portuguese (0.016) | 0.01 | 0.01 |

**Reading:** if the true effects resemble calibration, expect roughly one to three significant
primaries out of 24, and none for global. This is not a reason to cancel the study. It is a
reason to say so in the plan **now**, so that the paper reports the expected nulls as limited
precision, which they are, rather than as evidence of no effect. The repository already has the
right language ("a nonsignificant result is not proof of equivalence"); it needs the numbers.

**Caveats:** the direct estimate rests on 3 degrees of freedom and two batches that differ in
time and schedule; the bottom-up estimate assumes independent models. The truth is probably
between them. The calibration contrasts are noisy too, so they are scenarios, not true effects.

### 4.2 Cost per extra block by method

*(computed; per-cell means from V8, scaled so that 8/8/8/8 equals the team's $102.44 forecast)*

| Method | Cost per schedule block (4 models × 7 conditions) |
| --- | ---: |
| global | **$0.086** |
| local | $1.712 |
| sequential | $1.925 |
| iterative | **$9.082** |

Iterative is about 71% of the budget. Global is under 1%.

### 4.3 Allocations that fit under $120

| Allocation (global / local / sequential / iterative) | Forecast | Headroom | Half-width multiplier (× block SD) |
| --- | ---: | ---: | --- |
| **A, current:** 8 / 8 / 8 / 8 | $102.44 | $17.56 | 1.68 for all |
| **F:** 32 / 8 / 8 / 8 | $104.51 | $15.49 | global **0.59**; others 1.68 |
| **E:** 32 / 16 / 12 / 6 | $107.75 | $12.25 | global 0.59, local 0.93, sequential 1.16, iterative 2.38 |
| **G:** 32 / 12 / 12 / 7 | $109.98 | $10.02 | global 0.59, local 1.16, sequential 1.16, iterative 1.95 |

The owner chose equal repetitions deliberately, and that is a legitimate choice: it keeps the
four methods symmetric in the paper and is simple to explain. The point here is narrower.
**Option F costs $2.07, keeps three of four methods exactly as designed, and turns global from
unpowered into usable** (half-width ≈ 0.021 instead of 0.061). It is the same "global 32" scenario
the team already costed at $104.49 in V8. If equal allocation is kept, the paper should state that
global is reported descriptively because 8 blocks cannot resolve its plausible effects.

Unequal allocation is statistically valid here: each method's contrasts use only that method's
blocks, and the 24-test Holm family does not require equal n.

---

## 5. B3: Price Window and Run Duration

`Caller.__call__` calls `p.check_prices(self.spec)` before every paid request, and `check_prices`
refuses any date after `prices_valid_through = 2026-10-25`. The prices, and the window itself, are
part of the contract hash, and the contract hash is part of every `run_id`.

So:

- A run still going on October 26 stops at its next request.
- Refreshing the price evidence produces a new contract, a new run-id prefix and a new output
  folder, so the stopped run cannot be resumed. Its cells stay under the old contract, which (with
  B1) means almost no primary inference.
- The forecast is **26.3 serial API hours (39.5 with a 50% allowance)**, plus 30–120 s per transport
  recovery and any waits for the model-availability check.

"Launch by October 25" in [FINAL_COLLECTION.md](FINAL_COLLECTION.md) and
[PR_REVIEW_OCT18.md](PR_REVIEW_OCT18.md) should read **"complete by October 25"**. In practice:

1. **Launch no later than the morning of October 23** to keep the 50% allowance, and earlier is better.
2. **Disable sleep and hibernation** on the execution machine for the whole run. A sleeping laptop
   drops connections, and each drop consumes a transport recovery.
3. **Or change the semantics** in v6: verify prices once at authorization and record them. The ledger
   enforces the $120 cap from the recorded conservative rates regardless, and a price change mid-run
   affects the invoice, not the scientific validity of received replies. This is a policy choice for
   the owner; the current behavior is conservative but brittle.

---

## 6. R3: Contradictory Transport Text Inside the Frozen Contract

The contract embeds the October 17 analysis spec unchanged. Its `analysis.transport` field says:

> "SDK retries zero; automatic replacement retries zero … A general automatic-recovery runner is
> not approved by this document."

The same contract's top-level `transport_policy` (v4) permits `recover` and `launch --auto-recover`,
with up to 30 replacements. Both strings sit in one hash-pinned document. A reader of the
preregistration will not know which governs. In v6, replace the embedded text with a pointer to
`transport_policy`, or add one sentence: "Superseded by `transport_policy` in contract v4 and later."

---

## 7. Minor Findings

| # | Finding | Suggested action |
| --- | --- | --- |
| M1 | On a public clone, `python -m final_study status` crashes with a raw `FileNotFoundError` for the private `authorization.json`. It correctly writes `VALIDATION_PENDING` first, so no stale success survives. | Catch it and print "status requires the owner's private Hindi ledger; use `verify --stage hindi_check` on a public clone." |
| M2 | The longest tracked path is 143 characters. On Windows, cloning under a parent folder longer than about 115 characters fails with "Filename too long". | Add one line to the README: `git config --global core.longpaths true` before cloning. |
| M3 | `README.md` line 218 still shows the cumulative ledger as $14.3367342. This is the third review to flag it; the true figure is about $28.65 (V8 anchor $20.43 + calibration $7.89 + Hindi probe $0.31 + Hindi check $0.03). | Update it or remove the row. |
| M4 | The Hindi fix reads "realistic **and** believable" (यथार्थसंगत और विश्वसनीय), which is slightly stronger than English "plausible". | Acceptable. Mention it in the paper's prompt appendix with the double-space note. |
| M5 | Paired t with 7 degrees of freedom inflates family-wise error for light-tailed differences (the repository's own simulation showed 12.5% for uniform data). Model averaging makes block differences closer to normal, so the real risk is smaller. | Keep the planned bounded-mean or permutation sensitivity analysis as a secondary report. |
| M6 | Primary endpoints are clustering, not homophily. The team's reason (not choosing the strongest calibration signal) is sound. | No change. Make sure the abstract does not promise homophily findings from the primary analysis. |
| M7 | Running `status` or `preflight` rewrites tracked files under `outputs/final_study/`. | Harmless, but teammates will see a dirty tree after verifying. Consider untracking generated reports or documenting `git checkout -- outputs/final_study`. |

---

## 8. What the Team Did Well

This deserves saying plainly, because it is unusual in LLM-simulation research:

- **Every October 17 recommendation was addressed or explicitly declined with a reason**:
  schedule-blocked pairing, a fixed four-model panel, a duplicate-only parser, the Hindi fix with
  a paid live check, a general transport policy, and a frozen 24-hypothesis family.
- **The transport-recovery design is excellent.** It retires only no-reply transport failures,
  keeps the full reservation, allows one replacement per request, caps recoveries at 30, uses
  self-verifying replacement IDs, and makes every crash boundary in the recovery writes restartable.
  The test suite covers each of those boundaries.
- **Fail-closed everywhere.** Stale reports are invalidated before checks run, ledgers cannot be
  re-anchored, caps cannot be raised after authorization, and received replies are never bought twice.
- **Honest scoping.** "Seven named instruction contracts" is the correct claim for one wording per
  language, and the docs refuse to call nulls equivalence.
- **The byte-preservation fix was right.** Restoring the pinned line endings, rather than re-pinning
  normalized files, keeps the historical evidence chain intact.
- **Reproducibility.** A fresh clone passes all 244 tests and replays the paid Hindi evidence without
  a key or a private ledger.

---

## 9. Addendum: Commit `99cebbc` (Contract v5, Local File-Write Retries)

`99cebbc` ("fix: retry locked local receipt writes without resending requests") arrived while this
review was being written. It changes `final_study/runner.py`, `protocol.py`, the contract
(version, two source pins and a new `local_write_policy` field), the docs, the regenerated
manifest and preflight report, and adds four tests.

### 9.1 What it does

`runner.write_json` now wraps the existing atomic writer (`precollection_audit.write_json`:
temp file in the same folder, flush, `fsync`, `os.replace`). It retries **only `PermissionError`**,
up to five more times, waiting 0.05, 0.1, 0.25, 0.5 and 1 s. During `launch`, it is patched over
both inherited writer aliases (`p.engine.write_json` and `p.legacy.write_json`), so intents, raw
transport receipts and the post-reply and post-parse checkpoints are all covered. The aliases are
restored when launch exits.

### 9.2 Assessment

**Correct, safe, and a real improvement.** This addresses the exact Windows failure that
interrupted both V6 and the V8 wording screen.

- **No API resend is possible.** The retry sits below the transport layer and rewrites identical
  bytes. The new test `test_transient_file_locks_never_resend_received_reply` asserts one POST,
  one ledger row and identical failed bytes at each of the four boundaries, then relaunches and
  asserts byte-identical evidence.
- **Atomicity is preserved.** A failed attempt leaves the previous complete file in place and
  cleans up its temp file (`test_local_write_retries_are_bounded_and_atomic`).
- **Only locks are retried.** Disk-full and other errors fail immediately (`test_non_lock_errors_are_not_retried`).
- **Patch hygiene is right.** The inner `completed_marker` patch in `launch` restores to the
  retrying writer, not the original, so retries also cover the completion marker.
- **The migration guard was extended correctly:** v4 is added to the set of unexecuted contracts
  that `freeze` may supersede, with a test.

### 9.3 Remaining gaps

| # | Finding | Severity | Suggested action |
| --- | --- | --- | --- |
| N1 | **A persistent lock has no supported repair path.** If a post-reply checkpoint stays locked for all six attempts, the run stops with a received ledger row and a stale journal. Relaunch then fails with `Journal/ledger mismatch`, and `recover` refuses (`test_exhausted_checkpoint_lock_stops_without_rebuy` asserts both). That is the safe outcome, but V6 and V8 each needed a one-off, separately reviewed recovery script for exactly this state. In the main study that means an unplanned stop of unknown length, with the October 25 window closing (B3) and the global-shuffle order (B1) amplifying any stop into lost inference. | Medium | Add a reviewed reconciliation for this one state, in the same style as `reconcile_recoveries`: if the ledger row is `received`, its transport receipt exists and matches, and the journal equals the row's previous digest (`INTENT_BEFORE_TRANSPORT` or the pre-parse digest), rewrite the journal from the ledger. Refuse every other mismatch. |
| N2 | **The retry window is short.** Six attempts span about **1.9 seconds** in total. Antivirus scans and sync clients can hold a file for longer; the V8 incident's lock holder was never identified. Retrying identical local bytes costs nothing. | Low | Extend the backoff to roughly 30–60 s in total (for example 0.05 → 1 → 2 → 5 → 10 → 20 s). |
| N3 | The docs advise "a nonsynced working folder" but nothing checks it. | Low | A one-line warning at `launch` if the path is under OneDrive, Dropbox or Google Drive. |
| N4 | The regenerated `main_manifest.csv` (1,792 changed lines) changes only because run IDs embed the new contract hash. The order is the same `Random(44032)` global shuffle, still mixing seeds; under v5 the first cells run are seeds 44005, 44003, 44006, 44005, … | Info | Confirms B1 still applies. |

**None of findings B1–B4 or R1–R4 is affected by `99cebbc`.** The analysis, ordering, price check
and allocation are unchanged.

### 9.4 Verification of `99cebbc`

Run on this machine at `99cebbc` plus this review document, with no API key:

| Check | Result |
| --- | --- |
| Full offline suite (Python 3.11.9, lock file) | **248/248 passed** in 253.9 s: the previous 244 plus the 4 new tests |
| `python -m final_study preflight` | `OFFLINE_PREFLIGHT_PASSED`, **112 cases**, contract `17c2750f…` (matches the verification doc), 0 paid requests |
| `python -m final_study verify --stage hindi_check` | `COMPLETE`, 8/8 valid, original Hindi contract `5886a306…` |
| Execution order under v5 | Unchanged `Random(44032)` shuffle across all seeds (N4) |

---

## 10. Recommended Pre-Launch Sequence

1. **Rotate the API key** (B4).
2. **Contract v6**, offline only:
   - block-major execution order (B1);
   - block-level complete-case rule with a 6-of-8 threshold, current bounds kept as sensitivity (B2);
   - optionally one global correction request (B2, Option 2);
   - the transport-text fix (R3);
   - the expected-precision table from Section 4 written into the analysis spec (R1);
   - the owner's allocation decision: A or F (R2);
   - a narrow journal reconciliation for a persistent post-reply lock, plus a longer local-write
     backoff (N1, N2).
3. **Rerun the offline suite, `preflight` and `verify --stage hindi_check`** on a fresh clone. The
   Hindi bridge depends on the prompt catalog, not on execution order, so it should still replay.
4. **Record professor sign-off**, or the explicit decision to proceed without it, in `TEAM_APPROVAL.md` (R4).
5. **Launch by the morning of October 23**, with sleep disabled, using `--auto-recover` (B3).

None of these steps requires a paid call. Together they are about a day of work. Skipping them risks
spending most of $120 on a run whose primary analysis is mostly suppressed.

---

## Appendix A: Reproduction Scripts

Run from the repository root with the project `.venv` (Python 3.11.9). All scripts are read-only.

### A.1 Fresh-clone verification

```bash
git -c core.longpaths=true clone --branch r1/social-netw https://github.com/hemu77/UA_Capstone.git fresh
cd fresh
python -X utf8 -B -m unittest discover -s tests          # 244 tests
python -X utf8 -B -m final_study preflight                 # 112 cases
python -X utf8 -B -m final_study verify --stage hindi_check
```

### A.2 Suppression probabilities (Sections 2 and 3)

```python
from scipy import stats
lo, hi = stats.binomtest(1, 106).proportion_ci(method='exact')
print(1/106, lo, hi)
for p in [0.002, 0.005, 1/106, 0.02, 0.05]:
    print(p, (1-p)**64, (1-p)**224, (1-p)**32)
for f in [0.5, 0.8, 0.9, 0.95]:
    print(f, f**64, int(f*8))
```

### A.3 Precision and power (Section 4.1)

```python
import numpy as np
from scipy import stats
mult = stats.t.isf(.05/48, 7) / np.sqrt(8)                    # 1.679
def power(delta, sd, n=8, a=.05/24):
    nc, c = delta / (sd / np.sqrt(n)), stats.t.isf(a/2, n-1)
    return stats.nct.sf(c, n-1, nc) + stats.nct.cdf(-c, n-1, nc)
print(mult, power(0.045, 0.025), power(0.016, 0.036))
```

### A.4 Cost per block (Section 4.2)

```python
import pandas as pd
v8 = pd.read_csv('outputs/repair_v7/calibration/176533db7041/inspection_networks.csv')
blk = v8.groupby(['method', 'model'])['conservative_usd'].mean().unstack().sum(axis=1) * 7
scale = 102.44275142 / (blk.sum() * 8)
print((blk * scale).round(3))
```

### A.5 Seed quota totals (Section 1)

```python
import revision_next as v6
for s in range(44000, 44008):
    print(s, sum(v6.schedule(s)['nomination_quotas'].values()))
```
