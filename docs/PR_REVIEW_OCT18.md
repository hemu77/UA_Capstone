# Launch-Readiness Review: `8de0765` (eight repetitions, $120 cap)

**Date:** October 18, 2026 · **Branch reviewed:** `r1/social-netw` at `8de0765`
**Checks against:** [Raghav's October 17 review](PR_REVIEW_OCT17.md)
**Reviewer:** Claude (AI assistant), requested by the repository owner. Engineering
and evidence review only; not professor approval and not authorization to spend.

## Verdict

**Follow-up:** the restart-safety repair and its current validation are recorded
in [Final Collection Verification](FINAL_COLLECTION_VERIFICATION.md). This is
the original v3 review, not clearance for later source changes. All five missing
byte matches below were line-ending differences, not missing algorithm changes;
the owner copies preserve the bytes used by the historical evidence.

**Go for the main launch after three owner actions, not before.**

1. **Commit five local files** (blocker, owner machine only). The contract pins
   `bias.py`, `generate_personas.py`, `network_datasets.py` and `plotting.py` at
   bytes that are **not on GitHub**; `repair_outputs.py` differs only by CRLF.
   On a fresh clone, every Hindi-evidence check, `status` and `launch` fail with
   `Historical Hindi source evidence changed: bias.py` (5 test errors). Run
   `git status` / `git diff` in `UA_Captsone_SocN`, confirm the local edits are
   intended, and commit them. Do not edit them first: the paid Hindi evidence is
   pinned to those exact bytes.
2. **Use contract v3 from this review** (blocker fixed here). In `8de0765`, one
   lost provider request permanently stops the 896-cell run: the ledger row stays
   `uncertain`, `launch` refuses to resume, and there is no command to reconcile
   it. Raghav estimated 5–20 such stops for ~100,000 requests. Contract v3 adds
   `recover` and `launch --auto-recover` (details below).
3. **Re-run, on the owner machine, before launch:** the offline suite,
   `preflight`, `verify --stage hindi_check` and `status`. `status` must report
   `READY_FOR_MAIN_EXECUTION` (it needs the private Hindi ledger and the key).
   Launch by **October 25**: price freshness then blocks new calls, and changing
   prices changes the contract, so a stopped run cannot resume under it.

## Raghav's October 17 Findings: Current Status

| # | Finding | Status in `8de0765` (+ v3) |
| --- | --- | --- |
| 1 | Wording variance ≈ language variance | **Scoped, not solved.** Claim narrowed to "seven named instruction contracts"; one wording per language. Acceptable if the paper never states a language effect. |
| 2 | Schedules dominate variance | **Addressed in analysis.** Eight shared seeds, schedule-blocked paired contrasts, models not counted as repetitions. Quota totals still range 204–305 across seeds 44000–44007; blocking removes their main effect, not schedule × condition interaction. |
| 3 | Global failures (duplicates) | **Addressed.** Option A: exact undirected duplicates are collapsed and flagged; raw reply kept; invalid IDs stay failures. |
| 4 | Interruptions need a general policy | **Missing in `8de0765`; added in v3.** |
| 5 | Statistical plan undecided | **Frozen.** One primary endpoint per method, 24-hypothesis Holm family, complete-block rule. `meaningful_effect` and the precision target are still `null`. |
| 6 | Hindi "संभावित" | **Fixed** (`यथार्थसंगत और विश्वसनीय`) and paid-checked: 8/8 parsed. The cosmetic double space remains in three languages; fixing it would invalidate the pinned Hindi check, so leave it and note it in the appendix. |
| 7 | Exposed API key | **Not confirmed.** Docs still state no rotation is claimed. Rotate before launch. |
| 8 | Stale README ledger ($14.34) | **Still stale** (`README.md` cumulative ledger line). Low priority. |

## What Was Verified

- Python 3.11.9 with `requirements.lock.txt` (Linux; only Windows-only packages skipped).
- Fresh-clone suite on `8de0765`: 232 tests, **227 pass, 5 errors**, all from item 1.
- With those five files reporting their pinned hashes (simulating the owner
  checkout), the full suite passes **239/239** (final-study 29/29, including 7 new recovery tests);
  `preflight` passes all 112 combinations; the new manifest has 896 unique rows,
  8 per model/method/condition; public Hindi replay is `COMPLETE`, 8/8 valid.
- Cost: recomputing from V8 receipts gives ~$104 conservative ledger usage for 896
  cells; per-graph cost varies about 4%, so the $120 cap has comfortable headroom
  at unchanged prices. Request latency maximum was 31 s against the 60 s timeout.
- Precision: with 8 blocks and Bonferroni over 24, the simultaneous half-width is
  about **1.7 × the block-difference SD** (0.034 for SD 0.02). Effects smaller than
  that will not be detected. Report this honestly; it is the cost of eight repetitions.
- A full-suite run on Linux rewrites `text-files/revision224_adults.json` with LF
  endings. Harmless on Windows; restore it with `git checkout` after Linux runs.

## Contract v3: Transport Recovery

`final-instruction-study-v3-budget120-recovery` changes only
`final_study/runner.py`, `__main__.py`, `protocol.py`, the contract's version,
policy text and these three source pins. Every other pin is the owner's original.
No main attempt existed under v2, so `freeze` may supersede it.

A stop is recoverable only if **all** hold:

- exactly one unresolved attempt, a planned main-study request, not itself a replacement;
- no saved provider reply (`transport/<id>.json` absent);
- `uncertain` with error stage `transport` and a connection, timeout, 408/409/429
  or 5xx error, or `reserved` after a crash;
- fewer than 30 retirements so far.

Then the attempt becomes `abandoned` with its **full reservation retained** against
the same $120 cap, and exactly one replacement of the identical request is allowed.
Replacement ids are derived from the original id, so public replay verifies them
without the private ledger. Everything else exits with `RecoveryRefused`.

```powershell
python -X utf8 -B -m final_study launch --stage main --approve-usd 120 --approval-reference owner-eight-repetitions-120 --auto-recover
# or, after a stop:
python -X utf8 -B -m final_study recover --stage main --approval-reference owner-eight-repetitions-120
```

`precollection_spec.json` still says a general automatic-recovery runner is "not
approved by this document". `--auto-recover` is opt-in for that reason. Plain
`recover` is the explicit one-request recovery the contract already named.
