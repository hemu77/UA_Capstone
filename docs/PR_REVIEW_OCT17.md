# Follow-Up Design Review: V7 Diagnostics and V8 Calibration

**Date:** October 17, 2026
**Branch reviewed:** `r1/social-netw` at `8fa654b` (8 commits after the
[October 7 review](PR_REVIEW_r1_social-netw.md) at `3dc14e3`)
**Responds to:** [PR review response, October 12](PR_REVIEW_OCT12.md),
[V7 execution record](V7_EXECUTION_NOTES.md), [V8 execution record](V8_EXECUTION_NOTES.md),
[V8 results](V8_COMPLETION_RESULTS.md), [V8 handoff](V8_TEAM_HANDOFF.md)
**Reviewer:** Claude (AI coding assistant), requested by the repository owner. This is
an independent, evidence-based design review. It is **not** human bilingual validation,
professor approval, or authorization to spend.

---

## 0. Summary

The team responded to the October 7 review quickly and well. The global ablation was
run, a three-wording screen was run, the Portuguese grammar was fixed, the documentation
contradictions were resolved, and a fresh 104-cell calibration (V8) completed at
$7.89 of its $15 cap. The engineering discipline is still exemplary: every failure is
retained, nothing was silently repaired, and every paid replacement was separately
approved.

The new data answer some of the October 7 questions and raise three new ones. In order
of importance:

| # | Finding | Severity | What it means for the main study |
| --- | --- | --- | --- |
| 1 | **Wording variance is as large as language variance.** In the 96-graph wording screen, language differences look significant against run-to-run noise (religion homophily p = 0.0002), but **none** is significant when tested against variation between wordings of the same language (all p ≥ 0.06). For political-orientation homophily, the spread between wordings (≈0.07) is larger than the spread between languages (0.04). | **High** | RQ4 cannot be answered with one wording per language. The main study needs several wordings per language, with wording treated as a random effect. |
| 2 | **Randomization schedules (seeds) are a dominant variance component.** V8's seed 33001 drew 183 total nominations versus about 240 for the other seeds. That one draw lowered density in every quota-method graph and raised religion homophily by about 0.1. Every condition in a batch shares the same two seeds. | **High** | Block every contrast by schedule. Use more distinct schedules. Report or control homophily for quota totals. The effective sample size for generalizing across schedules is the number of seeds, not the number of graphs. |
| 3 | **The global task now works, but its failures are informative.** V8 produced 18 nonempty, non-matching global graphs out of 22 valid, which fixes the V6 problem. However, 4/26 (15%) were terminal failures with no correction allowed. 2 of the 3 duplicate failures repeat a single pair once in reversed order, and the fourth is an off-by-one ID. | Medium | 15% missingness concentrated in one method, for trivially fixable reasons. Predeclare either a V6-style preserving correction or canonicalization of exact duplicate pairs before main collection. |
| 4 | **The main study will interrupt often.** V8 hit two provider 5xx errors in 11,634 requests, and each needed owner approval plus a new one-off recovery script. A 896-graph main study is about 100,000 requests. | Medium | Expect on the order of 5–20 interruptions. A general, pre-reviewed transport-retry policy is needed before main collection, not a new bespoke adapter each time. |
| 5 | **The statistical plan is still undecided, and new artifacts disagree with old ones.** `detectable_effect_scenarios.csv` uses families of 1, 3 and 12 hypotheses; `plan.md` still declares 24 and 84. No primary outcome, family, precision target or failure rule has been chosen. | Medium | This is now the critical-path blocker, not data collection. |
| 6 | **The Hindi global instruction is weaker than the other three.** It renders "plausible" as संभावित ("possible / potential"). | Low | Fix in the next contract; it is exactly the kind of drift a fluent reader catches in minutes. |
| 7 | **An exposed API key has not been confirmed rotated.** [V7_EXECUTION_NOTES.md](V7_EXECUTION_NOTES.md) records that the credential "remains exposed in prior chat and should be rotated." | **High (security)** | Rotate it now, before any further paid run. No key is committed to the repository; this concerns the chat exposure only. |
| 8 | Stale figures remain in `README.md` (October 2 status block, $14.34 cumulative ledger). | Low | Doc cleanup (Section 7). |

**Recommendation:** keep main collection blocked. **No further paid diagnostics are
needed before the design decisions.** The data already in hand are enough to make
them. The next step is a one-page frozen analysis plan (Section 8) that the professor
can approve. After that come a small multi-wording pilot inside the main design and
the main collection.

---

## 1. What Changed Since October 7

| October 7 recommendation | What the team did | Status |
| --- | --- | --- |
| Global ablation, about 64 graphs, under $1 | V7: 4 clause arms × 4 models × 4 reps = 64 graphs, $0.118. Then 8 graphs with explicit "construct a hypothetical network" framing, $0.024. V8 adopted that framing. | Done. Cost estimate held ($0.142 total). |
| Fix Portuguese agreement error | `revision_v7_prompts.py`: "O contexto social é o dos Estados Unidos." Applied to all countries and operations, with composition tests. | Done, correctly. |
| Translation-variant arm | V8 wording screen: 4 models × 4 languages × 3 local wordings × 2 reps = 96 graphs, $5.95. | Done. Analysis of wording vs language not yet reported (Section 3.1 does it). |
| Fluent-reader check | Not done; waiver retained. A GPT-6.1-Sol reviewer checked semantics. | Open. |
| Fix doc contradictions | Simulator status reconciled; current-status blocks added. | Mostly done (Section 7). |
| Model-averaged primary estimand, replace Hoeffding, failure-replacement rule | Discussed in [PR_REVIEW_OCT12.md](PR_REVIEW_OCT12.md); not decided. | Open. |

Spending since October 7 (conservative local accounting): V7 diagnostics $0.142,
wording screen $5.946, V8 calibration $7.891, Hindi probe $0.306, **total ≈ $14.29**.
The cumulative ledger is therefore about **$28.62** (V8 calibration anchor $20.425088
+ $7.8912884 + $0.30618).

---

## 2. Response to the Team's Pushback

The [October 12 response](PR_REVIEW_OCT12.md) raised fair objections. Taking them in turn:

| Team point | Assessment |
| --- | --- |
| "Pooling four configurations does not automatically make 32 IID pairs." | **Correct, and the V8 data make it stronger.** All four models share the same seeds, and seeds turn out to be a large variance component (Section 3.2). Schedule main effects cancel in within-schedule contrasts, but schedule × condition variation is estimated from the number of seeds, not 4 × seeds. The October 7 "n = 32" figure was too optimistic. Section 4 restates the precision arithmetic with the schedule as the block. |
| "A valid paired design does not require positive observed correlation. Do not choose paired versus unpaired after seeing results." | **Correct. I withdraw the October 7 suggestion to consider unpaired analysis.** The V8 seed effects show that blocking by schedule is clearly beneficial here. Declare the blocked analysis now. |
| "Replacement is not automatically unbiased: failure can depend on output length or content." | **Correct.** The four V8 failures are too few to show a length dependence (Section 3.3), but the mechanism is plausible. That argues for preserving correction or canonicalization rather than replacement. It also argues against strict terminal failure, which carries the same risk. |
| "Do not accept an unverified 'under $1' estimate." | Fair as process. The outcome was $0.142 for 72 graphs. |
| "The review's 11.4% … were not independently reproduced." | The script is in Appendix A.1 of the October 7 review and runs in a few seconds. Either way, the repository's own 12.5% supports the same conclusion. |
| "The third-party `review_experiments/` folder is absent from this checkout." | True for the team's checkout. It exists untracked in the repository owner's working copy. The owner should either commit it (it is small, offline-only and read-only) or share it, so that the team can cite it. |
| Clause screen showed "the earlier assumption that a clause change would be sufficient was not supported." | Agreed, with one caveat (Section 3.4): every arm still offered `NONE` in its response-syntax line, so the one feature that most distinguishes V5 from V6 was never removed. The construction framing made this moot for V8. |

---

## 3. New Findings From the V7/V8 Data

All numbers marked *(computed)* were produced from committed CSVs and receipts using the
scripts in Appendix A. No API calls were made.

### 3.1 Wording variance is as large as language variance

The wording screen is the right experiment: for each of four languages it has three local
wordings (`reference`, `alternative_1`, `alternative_2`) × 4 models × 2 reps, with matched
schedules. The repository reports only edge Jaccard and density differences per pair. The
question that matters for RQ4 is whether **language** effects exceed **wording** effects.

*(computed)* Nested ANOVA on the 94 valid graphs:
`metric ~ model + language + language:wording + model:language`.

| Metric | F, language vs residual | p | F, wording-within-language vs residual | p | **F, language vs wording** (df 3, 8) | **p** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clustering | 3.22 | 0.028 | 0.87 | 0.54 | 3.68 | **0.062** |
| modularity | 2.95 | 0.039 | 1.47 | 0.18 | 2.01 | 0.19 |
| Coleman gender | 1.02 | 0.39 | 0.44 | 0.90 | 2.33 | 0.15 |
| Coleman religion | 7.31 | **0.0002** | 2.10 | **0.047** | 3.48 | **0.070** |
| Coleman political orientation | 1.51 | 0.22 | 2.15 | **0.042** | 0.70 | 0.58 |
| age assortativity | 1.87 | 0.14 | 1.51 | 0.17 | 1.23 | 0.36 |
| density | 0.43 | 0.74 | 0.39 | 0.92 | 1.09 | 0.41 |

Reading the table:

- The "language vs residual" column is what a one-wording design implicitly tests. It
  would report language effects on clustering, modularity and religion homophily.
- The "language vs wording" column is the correct test **if the goal is to generalize to
  the language rather than to one sentence** (wording as a random effect). Nothing reaches
  p < 0.05 there.
- For political orientation, wording matters significantly (p = 0.042) while language does
  not (F < 1).

*(computed)* Cell means by language and wording:

| Coleman religion | alternative_1 | alternative_2 | reference | language mean |
| --- | ---: | ---: | ---: | ---: |
| English | 0.222 | 0.264 | 0.248 | 0.245 |
| Hindi | 0.194 | 0.237 | 0.199 | **0.210** |
| Japanese | 0.226 | 0.247 | 0.222 | 0.232 |
| Portuguese | 0.242 | 0.289 | 0.264 | **0.265** |

| Coleman political orientation | alternative_1 | alternative_2 | reference | language mean |
| --- | ---: | ---: | ---: | ---: |
| English | 0.519 | 0.579 | 0.590 | 0.563 |
| Hindi | 0.540 | 0.518 | 0.551 | 0.536 |
| Japanese | 0.524 | 0.587 | 0.546 | 0.552 |
| Portuguese | 0.605 | 0.543 | 0.579 | 0.576 |

There is one encouraging pattern. For religion homophily, Hindi is the lowest language
**under all three wordings** and Portuguese the highest under two of three, while
`alternative_2` raises religion homophily in every language. That is what a real language
effect plus a wording main effect would look like. With two reps and 8 degrees of freedom
for wording, the evidence is suggestive (p = 0.07), not conclusive. For political
orientation, by contrast, the language ordering reshuffles with each wording, so any
"language effect" seen with one wording would mostly be a property of that sentence.

**Implications:**

1. Report RQ4 as a **wording-robust** language contrast: estimate it with several wordings
   per language and wording as a random effect. Results from the reference wording alone
   are wording-specific.
2. **Carry the multi-wording design into the main study** for the RQ4 cells (US × 4
   languages), at least for local and sequential. Three wordings with the same total budget
   is better than one wording with three times the repetitions: the wording variance can
   only be averaged down by adding wordings, not repetitions.
3. English needs the same treatment. `alternative_1` and `alternative_2` are English
   paraphrases, so the English baseline is already measured with wording noise. Keep that.
4. The alternatives are AI-authored by the implementation agent (as the code comment in
   `revision_v7_prompts.py` says). Independent authorship (a different person or system)
   would make the wording variance more representative of real translation choices.

### 3.2 Schedules (seeds) drive large shared shifts

V6 and V8 use the same local, sequential and iterative prompts outside Portuguese, yet the
batches differ systematically. *(computed)* V8 minus V6, non-Portuguese cells, all models:

| Method | Coleman religion | Welch p | Density | Welch p |
| --- | ---: | ---: | ---: | ---: |
| local | +0.038 | 0.064 | −0.010 | 0.010 |
| sequential | +0.033 | 0.23 | −0.012 | 0.013 |
| iterative | +0.056 | **0.002** | −0.013 | **0.002** |

The cause is the schedule, not provider drift. *(computed)* Total nomination quota per seed:

| Seed | Batch | Total quota (sum over 50 actors) |
| ---: | --- | ---: |
| 21000 | V6 rep 0 | 241 |
| 21001 | V6 rep 1 | 243 |
| 33000 | V8 rep 0 | 237 |
| **33001** | **V8 rep 1** | **183** |

*(computed)* GPT-6-Luna means by seed:

| Method | Seed | Density | Coleman religion | Clustering | Age assortativity |
| --- | ---: | ---: | ---: | ---: | ---: |
| local | 21000 | 0.154 | 0.330 | 0.473 | 0.531 |
| local | 21001 | 0.165 | 0.286 | 0.453 | 0.566 |
| local | 33000 | 0.160 | 0.331 | 0.458 | 0.527 |
| local | **33001** | **0.124** | **0.408** | **0.416** | **0.630** |
| sequential | 21000 | 0.158 | 0.304 | 0.478 | 0.358 |
| sequential | **33001** | **0.124** | **0.393** | **0.417** | 0.442 |
| iterative | 21000 | 0.162 | 0.178 | 0.587 | 0.414 |
| iterative | 33000 | 0.161 | 0.256 | 0.585 | 0.524 |

A single low-quota draw lowered density, lowered clustering and raised religion homophily
by about 0.08–0.12. That is **larger than the language spread in Section 3.1 (0.055)**.
The mechanism is plausible: when each actor makes fewer nominations, a larger share of them
are the actor's top picks, which tend to be similar people. Iterative 33000 vs 21000 shows a
religion shift (+0.08) with similar quota totals, so quota totals are not the only
schedule-level factor.

**Implications:**

1. **Block every contrast by schedule.** Every condition in a contrast must share the same
   schedule list, and the analysis must include the schedule as a block (paired differences
   or a random/fixed seed effect). The repository's design already shares schedules; the
   analysis plan must use them.
2. **Use more distinct schedules.** Schedule main effects cancel in within-schedule
   contrasts, but schedule × condition interactions do not. Their precision depends on the
   number of seeds. Prefer more seeds with fewer models per seed over fewer seeds reused
   everywhere, if the budget forces a choice.
3. **Homophily outcomes are partly quota-driven.** Either stratify schedules on total quota
   (draw seeds until their quota totals fall in a narrow band, decided before data), fix the
   quota sequence, or analyse homophily relative to the degree-preserving controls that
   `make_matched_baselines.py` already generates. Otherwise a contrast that happens to land
   on different quota totals can produce a spurious homophily difference.
4. **Never compare across batches.** V6 vs V8 differences are mostly schedule effects. This
   reinforces the repository's existing "no calibration reuse" rule.

### 3.3 The global task works now, but failures are informative

*(computed)* V8 global by model:

| Model | Attempts | Valid | Empty | Median edges (valid) | Max edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-4.1 | 4 | 3 | 0 | 53 | 55 |
| GPT-5.6-Luna | 4 | 4 | 0 | 130 | 180 |
| GPT-6-Luna | 14 | 12 | 4 | 62.5 | 121 |
| GPT-6-Sol | 4 | 3 | 0 | 78 | 96 |

Across the 22 valid graphs, mean degree-one share is 0.03 and median LCC share is 0.97, so
the V5 near-perfect-matching failure mode has not returned. **The construction framing is a
genuine improvement.** It is post-diagnostic, which the docs disclose correctly.

*(computed)* The four terminal failures:

| Cell | Model / setting | Lines in reply | Distinct pairs | Problem |
| --- | --- | ---: | ---: | --- |
| 023 | Luna, India-English | 56 | 55 | One pair repeated, reversed order (`44, 49` and `49, 44`) |
| 072 | Sol, US-Japanese | 98 | 97 | One pair repeated, reversed order (`32, 49` and `49, 32`) |
| 088 | Luna, US-Japanese | 118 | 118 | One out-of-roster ID (`50`; roster is 0–49) |
| 089 | GPT-4.1, US-Hindi | 47 | 41 | Six pairs repeated, reversed order |

*(computed)* The failed replies are not systematically longer than the valid graphs: two are
longer than their model's median nonempty graph (98 vs 78 for Sol, 118 vs 76 for Luna) and two
are shorter (56 vs 76 for Luna, 47 vs 53 for GPT-4.1). With four failures there is no evidence
either way that failure depends on graph size. The risk remains structural, though: a longer
list has more chances to repeat a pair, so at main-study scale strict terminal failure could
remove dense graphs preferentially. That is the team's own point about informative failure.
It should be checked by condition in the main study rather than assumed away.

The more immediate problem is the rate. 15% missingness in one method, for reasons as trivial
as a reversed duplicate or an off-by-one ID, costs precision and complicates every global
contrast.

V8 treats a malformed global reply as terminal with no correction (`repair_v8.py`, line 186).
This differs from V6, where global had a preserving correction ("return exactly the original
valid friendship set"), and from local in V8, which gets three corrections. The asymmetry
appears to come from the diagnostic code path (`treatment_retry` rejects global corrections
for treatment arms), not from a separate design decision.

**Recommendation (predeclare one before main collection):**

- **Option A, canonicalize exact duplicates.** Collapse an exact repeat of an undirected pair
  (including reversed order) into one friendship and record it as a compliance deviation.
  This invents no choice: the model named the same friendship twice. Out-of-roster IDs remain
  failures or trigger a correction. Report compliance rates by condition either way.
- **Option B, restore the V6 preserving correction** for global: one or two correction
  requests that must return the original valid set without additions or omissions.

Either option, declared in advance, is less biased than terminal failure. The current V8
choice (no repair, no correction) should be re-justified if kept.

### 3.4 Note on the V7 clause screen

*(computed)* V7 global probe, empty graphs by model across all four arms:

| Model | Empty / 16 |
| --- | ---: |
| GPT-4.1 | 16 |
| GPT-5.6-Luna | 16 |
| GPT-6-Luna | 16 |
| GPT-6-Sol | 3 |

Every arm appended `Response syntax: NONE, or one ID1, ID2 pair per line.` So "permission
to answer NONE" was present in all four arms. The arms varied the "no required number"
clause and an extra empty-network reminder, not the availability of `NONE`. V5, which never
mentioned `NONE`, had 0/17 empty graphs. The screen therefore shows that removing those two
clauses is not sufficient. It does not show that `NONE` availability is irrelevant. The
construction framing fixed the problem anyway, so this matters only for how the V7 result is
described in the paper: say "the clause variants tested did not reduce empty outputs," not
"the clauses were not the cause."

### 3.5 Translation of the V8 global instruction

The V8 global prompts were reviewed by an AI reviewer only ([V8 notes, "Language Review"](V8_EXECUTION_NOTES.md)).
This reviewer's own reading (also AI, not native-speaker review):

| Language | Construction sentence | Observation |
| --- | --- | --- |
| English | "Generate a plausible realization of friendships …" | Reference. |
| Portuguese | "Gere uma configuração plausível de amizades …" | Faithful. The context fix ("O contexto social é o dos Estados Unidos") reads naturally for all four countries. |
| Japanese | "あり得そうな友人関係の一例を生成してください" | Faithful ("an example of likely-possible friendships"). |
| Hindi | "मित्रताओं का एक संभावित विन्यास उत्पन्न करें" | **Weaker.** संभावित means "possible / potential", not "plausible". A closer rendering is "यथार्थसंगत" or "विश्वसनीय" (realistic, believable). The instruction drops the realism constraint the other three languages carry. |
| All | `… with ID1 < ID2.  No self-links …` | Double space where the removed `NONE` sentence used to be. Cosmetic; harmless to the model, but visible in a paper appendix. |

*(computed)* Valid V8 global graphs by language: English mean 56.9 edges (n = 10), Hindi
94.5 (n = 4), Japanese 98.3 (n = 3), Portuguese 65.4 (n = 5). These counts are too small and
too mixed across models to attribute to wording. Still, Hindi being the densest under the
weakest realism wording is exactly what Section 3.1 warns about: a language contrast can be
a sentence contrast.

**The fluent-reader check from October 7 is still the cheapest high-value step.** The full
V8 instruction set is about 10 sentences per language. One fluent reader per language,
about 20 minutes each, would catch this kind of drift before 100,000 requests are spent on it.

---

## 4. Repetition Count and Precision, Restated

### 4.1 What the October 7 arithmetic got wrong

The October 7 review treated the model-averaged contrast as n = 32 (8 reps × 4 models). Since
all models share seeds, the honest accounting is:

- **Schedule main effects** cancel within each schedule block. Good.
- **Schedule × condition variation** is estimated from the number of distinct schedules. With
  8 seeds that is 8, not 32.
- **Model × schedule × condition noise** averages over models and seeds (4 × 8 = 32).

So the effective n lies between 8 and 32, depending on how large the schedule × condition
variance is. The current data cannot separate these components (two seeds per batch).

### 4.2 What to do about it

1. **Estimate the components before freezing the allocation.** The V6 and V8 batches together
   give GPT-6-Luna four seeds per cell for local, sequential and iterative under identical
   non-Portuguese prompts. They cannot be pooled as confirmation data, but they **can** be
   pooled for variance-component planning, with batch as a fixed effect. This is free.
2. **Prefer more seeds over more models per seed** when the budget binds, for the reasons in
   Section 3.2.
3. **Reconcile the hypothesis families.** `outputs/repair_v7/calibration/176533db7041/detectable_effect_scenarios.csv`
   uses families of 1, 3 and 12. `plan.md` still declares 24 (RQ1, RQ4) and 84 (RQ3) per
   method. Pick one, and write down where the number comes from.
4. **Precision target first.** The team still has no raw-unit meaningful effect for any
   outcome. Without one, no repetition count can be justified. A pragmatic default for
   bounded indices is to target a 95% interval half-width of 0.02–0.03 on the
   schedule-blocked, model-averaged contrast. The professor should confirm or replace it.

---

## 5. Analysis Plan: Decisions Still Needed

Nothing in this section requires new data. Each item is a decision the team or professor
can make now.

| Decision | Recommendation | Why |
| --- | --- | --- |
| Confirmatory RQs | RQ1 and RQ4 confirmatory; RQ2 and RQ3 descriptive. | Agreed in principle in the October 12 response; not yet recorded. |
| Primary outcome per method | One homophily outcome relative to degree-preserving controls (for example religion Coleman, which shows the most consistent pattern in Section 3.1), plus clustering as secondary. | Homophily is what RQ1/RQ2 promise, and controls absorb the quota effect in Section 3.2. |
| Estimand | Model-averaged contrast, blocked by schedule; per-model estimates and a heterogeneity summary reported as secondary. | Sections 3.2 and 4.1. |
| RQ4 wording | Three wordings per language, wording as a random effect. | Section 3.1. |
| Inference | Mixed model with a fixed condition effect and random intercepts for seed, model and model × condition, plus language × wording for RQ4 (formula below the table); permutation or cluster-bootstrap sensitivity analysis. Drop the Hoeffding fallback as a decision rule. | Hoeffding has no practical power at these n (October 7, Section 4.2; confirmed in the October 12 response). |
| Global failures | Option A or B in Section 3.3, declared in advance. | Section 3.3. |
| Run order | Randomize or interleave conditions across the whole collection window, so that provider-side changes over a multi-day run do not align with conditions. | The main study is about 26–40 serial API hours. |
| Multiplicity | With the estimand above, RQ1 and RQ4 have 3 contrasts × 1 primary outcome per method. Correct within each RQ across methods, or study-wide, whichever the professor prefers. | Far smaller families make study-wide control realistic. |

Proposed mixed-model formula (lme4 notation):

```text
outcome ~ condition + (1 | seed) + (1 | model) + (1 | model:condition)
# RQ4 only, add:  + (1 | language:wording)
```

---

## 6. Operational Readiness for Main Collection

### 6.1 Interruptions will recur

*(computed)* V8 calibration: 11,634 requests, 2 provider 5xx interruptions, plus 1 local
file-lock interruption in the wording screen. V6: 11,548 requests, 0 provider interruptions,
1 file-lock interruption. Main collection at 896 graphs is about **100,000 requests**
(≈112 requests per graph). At the observed rate (2 per ~23,000), expect **on the order of
5–20 interruptions**.

Each V8 interruption required an owner approval, a new fingerprinted recovery script
(`repair_v8_transport.py`, then `repair_v8_transport_second.py`), new tests and a reviewer
pass. The repository now has about nine checkpoint- or incident-specific recovery and
adapter scripts, for example `resume_wording_probe.py`, `recover_calibration_v6_checkpoint.py`,
`calibration_v6_io.py`, `repair_v7_checked.py`, `finish_global_task_probe.py`, `repair_v8_io.py`,
`repair_v8_transport.py`, `repair_v8_transport_second.py` and `inspect_repair_v8_checkpoint.py`.
That pattern does not scale to the main study.

**Recommendation:** before main collection, write and review **one** general transport policy
and implement it once in the main runner. For example:

- An HTTP 5xx or connection error **with no received response** may be retried automatically
  under a fresh deterministic request identity. The original reservation is retained, as now.
- Automatic retries per request are capped (for example 2), and so is the total per run (for
  example 30). Exceeding a cap stops for human review.
- Every retry is logged with its reservation, and the reconciliation report lists them.
- Received but malformed responses are **not** transport errors and follow the parse-failure
  rule instead.

This keeps every safeguard the team built (no silent repurchase of a received reply, no
budget reset, full audit) while removing the need for a new script and approval per incident.

### 6.2 Credential rotation

[V7_EXECUTION_NOTES.md](V7_EXECUTION_NOTES.md) (lines 16–18) states that the API key used for
V7 "remains exposed in prior chat and should be rotated; this run does not assert that a fresh
key was configured." V8 then ran further paid stages. **Rotate the key in the provider dashboard
now** and record in the execution notes that it was rotated, without recording the key itself.
I scanned the repository and the eight new commits: no live key is committed (only `YOUR_KEY`
placeholders in archived READMEs).

---

## 7. Documentation Consistency

| Location | Problem | Suggested fix |
| --- | --- | --- |
| `README.md` lines 35–60 | The superseded "Status, October 2" block (V6, "22/26 global graphs are empty") still follows the October 13 status. | Move it under a "History" heading or delete it; the V6 results doc retains it. |
| `README.md` line 160 | "Cumulative historical + probe + calibration ledger: $14.3367342." | About $28.62 after V8 and the Hindi probe (Section 1). |
| `plan.md` §5C vs `detectable_effect_scenarios.csv` | Families of 24/84 vs 1/3/12. | Reconcile (Section 4.2). |
| `docs/PR_REVIEW_OCT12.md` | Says `review_experiments/` is absent. | True for the team; resolved if the owner commits it (Section 2). |
| `docs/V7_EXECUTION_NOTES.md` | Describes the clause screen as showing the clauses were not sufficient, without noting that `NONE` stayed available in all arms. | Add one sentence (Section 3.4). |
| Simulator status | Now consistent: README, V8 notes and handoff all say the simulator shows V6. | No action. |

---

## 8. Recommended Next Steps

In order. Steps 1–5 are free.

1. **Rotate the exposed API key** (Section 6.2).
2. **Write a one-page frozen analysis plan** covering every row of the Section 5 table, and get
   professor sign-off in `TEAM_APPROVAL.md`. This is the critical-path item.
3. **Estimate variance components** from the pooled V6 + V8 GPT-6-Luna data (four seeds per
   cell, batch as a fixed effect) and the wording screen. Use them to set seeds, models per seed
   and wordings per language for the main study (Sections 3.1, 3.2 and 4.2).
4. **Fix the Hindi construction sentence and the double space**, get the fluent-reader check
   done, and freeze the next prompt contract (Section 3.5).
5. **Implement the global-failure rule and the general transport policy** in the main runner,
   with tests (Sections 3.3 and 6.1).
6. **Commit or share `review_experiments/`** so that its evidence can be cited (Section 2).
7. Only then: reforecast cost for the frozen design (it will change, because of multiple RQ4
   wordings and possibly more seeds) and bring a finite budget for main-collection approval.

No further paid diagnostic is needed before steps 2–5. The existing V6, V7 and V8 data already
contain the information those decisions require.

---

## Not Verified

- **Update, October 17:** the 191-test offline suite **was** rerun independently, after
  installing Python 3.11.9 and the pinned dependencies: **191 tests passed in 243.1 seconds**,
  with no API key in the environment. Only the known pandas/NumPy deprecation warnings appeared.
  The environment is now reproducible via `scripts/setup_env.ps1` and `requirements.lock.txt`.
  Passing tests confirm the software contracts, not the scientific findings above.
- Private-ledger reconciliation was not rechecked; it needs the private ledger.
- The ANOVA in Section 3.1 has two reps per cell and 8 degrees of freedom for wording. Its
  p-values are planning evidence, not confirmatory results. The schedule analysis in Section 3.2
  rests on four seeds.
- Translation observations are an AI reading, not native-speaker review.

---

## Appendix A: Reproduction Scripts

Run from the repository root with any Python that has NumPy, SciPy, pandas and statsmodels.
All scripts are read-only.

### A.1 Language vs wording (Section 3.1)

```python
import pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
d = pd.read_csv('outputs/repair_v7/wording_probe/026a883076d5/inspection_networks.csv')
d = d[d.valid_graph == True].rename(columns={'coleman_political orientation': 'coleman_pol'})
for k in ['avg_clustering_coef', 'modularity', 'coleman_gender', 'coleman_religion',
          'coleman_pol', 'age_assortativity', 'density']:
    fit = smf.ols(f'{k} ~ C(model) + C(language) + C(language):C(wording) + C(model):C(language)', data=d).fit()
    a = anova_lm(fit, typ=1); ms = a.sum_sq / a.df
    f = ms['C(language)'] / ms['C(language):C(wording)']
    print(k, 'p_lang_vs_resid', round(a.loc['C(language)', 'PR(>F)'], 4),
          'p_wording', round(a.loc['C(language):C(wording)', 'PR(>F)'], 4),
          'F_lang_vs_wording', round(f, 2), 'p', round(stats.f.sf(f, 3, 8), 3))
```

### A.2 Schedule effects (Section 3.2)

```python
import json, glob, pandas as pd
for pattern in ['outputs/calibration_v6/5fb3715550db/runs/*local*.json',
                'outputs/repair_v7/calibration/176533db7041/runs/*.json']:
    seen = {}
    for p in glob.glob(pattern):
        s = json.load(open(p, encoding='utf-8')).get('schedule') or {}
        if 'nomination_quotas' in s and s['seed'] not in seen:
            seen[s['seed']] = sum(s['nomination_quotas'].values())
    print(seen)
v8 = pd.read_csv('outputs/repair_v7/calibration/176533db7041/inspection_networks.csv')
v8 = v8[v8.valid_graph == True].assign(seed=lambda x: x.repetition.map({0: 33000, 1: 33001}))
v6 = pd.read_csv('outputs/calibration_v6/5fb3715550db/network_metrics.csv')
v6 = v6.assign(seed=v6.repetition.map({0: 21000, 1: 21001}))
x = pd.concat([v6, v8]).query("method != 'global' and model == 'gpt-6-luna'")
print(x.groupby(['method', 'seed'])[['density', 'coleman_religion', 'avg_clustering_coef']].mean().round(3))
```

### A.3 Global failures (Section 3.3)

```python
import json, glob, re, collections
for p in sorted(glob.glob('outputs/repair_v7/calibration/176533db7041/failures/*.json')):
    r = json.load(open(p, encoding='utf-8'))['requests'][0]['response']
    pairs = [tuple(sorted(map(int, re.findall(r'\d+', l)))) for l in r['text'].splitlines() if l.strip()]
    c = collections.Counter(pairs)
    print(p[-8:-5], r['parse']['error'], 'lines', len(pairs), 'distinct', len(c),
          'repeated', {k: v for k, v in c.items() if v > 1})
```

### A.4 V7 clause screen (Section 3.4)

```python
import pandas as pd
c = pd.read_csv('outputs/repair_v7/global_probe/bbd4c3c82c66/condition_summary.csv')
print(c.groupby('model')[['graphs', 'empty_graphs']].sum())
```
