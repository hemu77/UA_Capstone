# V8 Handoff: What Changed, What Was Checked, What Comes Next

October 13, 2026. This is the current plain-language companion to the technical
[execution record](V8_EXECUTION_NOTES.md). Earlier V5/V6/V7 results remain history,
not replacement evidence for this version. Main collection is not authorized.

## One-Minute Explanation

We ask models to build friendships among the same 50 fictional adults. We test
whether their choices change with country framing, instruction language, model,
and generation method. We are studying generated behavior, not measuring real
national populations. A graph passing a software check does not prove that its
friendships are realistic.

The recent work has two jobs: make the intended task clearer, and make every
saved result traceable to the exact prompt and paid response that produced it.
We retain inconvenient outcomes instead of redrawing them until they look good.

## Earlier Versus Now

| Earlier concern | What changed | What the change does not prove |
| --- | --- | --- |
| Original US political-party labels could imply a general population | The revised roster uses fictional adults and generic political orientation, without US party names or race labels | Its deliberately chosen proportions are not census estimates or culturally equivalent measurements |
| Fixed candidate order could confound position with identity | Candidate order and actor order have separate recorded randomization streams | Position sensitivity is not eliminated merely by randomization; matched schedules and analysis remain necessary |
| Some outcomes were largely imposed by the task | Nomination quotas are explicit; density in quota-based methods is treated as a constraint/compliance descriptor | We cannot interpret constrained density as unrestricted sociability |
| Global instructions could be read as recovering known friendships from profiles | New English, Hindi, Japanese and Brazilian Portuguese prompts explicitly request a plausible hypothetical simulated network | Different output after a post-diagnostic change is not an untouched preregistered finding |
| Empty networks looked like a software error | Valid `NONE` outputs remain genuine empty graphs; invalid responses remain failures | A nonempty network is not inherently a better or more realistic result |
| Portuguese singular phrasing and Hindi answer formatting caused concern | Separate wording screens and an isolated Hindi clarification probe test operational compliance | AI review and successful parsing do not establish human bilingual equivalence |
| A local file-write interruption could tempt a paid retry | Bounded retries write only the identical local checkpoint; received responses replay from the cache | This is not permission to resend an uncertain provider request |
| Provider HTTP 500 and 520 left billing uncertain | Each separately authorized exact replacement retains its original reservation and has its own deterministic identity | A local conservative reservation is not the provider invoice |
| Earlier completion wording could hide missing graphs | Reports distinguish planned observations, valid graphs, terminal failures, and incomplete runs | Completing the planned attempts does not make every graph valid |

## Study Scope

- Four models: GPT-4.1, GPT-5.6-Luna, GPT-6-Luna and GPT-6-Sol.
- Four procedures: global, local, sequential and iterative.
- Four country framings: US, India, Japan and Brazil.
- Four instruction languages: English, Hindi, Japanese and Brazilian Portuguese.
- Seven settings: four countries in English, plus US in the other three languages.
- Each graph contains the same 50 fictional adults. Persona fields remain English.

This is not a full country-by-language factorial. Country-language interactions
cannot be identified. The model panel is a budget-aware within-provider comparison:
GPT-4.1 is a legacy reference, the Luna configurations broaden the lower-cost
comparison, and Sol adds another deployed configuration. Four model names do not
establish cross-vendor robustness or isolate architecture from decoding settings.
Provider aliases, resolved model names, supported settings and response metadata
must remain part of the evidence.

The four research questions remain: country-framing sensitivity with English
fixed; demographic mixing; consistency across models; and instruction-language
sensitivity with US framing fixed. All four methods contribute to those questions.
The simulator is an internal exploration tool, not an independent validation.

Calibration is deliberately smaller than the proposed main study: 26 cells per
method (104 total). GPT-6-Luna has 56 cells covering seven settings twice; each
other model has 16 cells covering the four US-language settings once. Therefore,
non-US costs for the other three models require a proxy, and one observation in
many cells cannot establish their variance or statistical power.

## Debugging And Evidence

The 96-cell wording screen completed with 94 valid graphs and two retained
GPT-4.1 Hindi alternative-2 failures. All 32 prospectively selected reference
graphs completed. The failed answers were not repaired by flattening lines or
deduplicating IDs, because that could invent choices the model did not make.

The calibration then stopped on a provider HTTP 500 after 63 completed
observations. The unfinished cell already had 236 received replies. Recovery
reuses those replies and permits only one separately identified replacement for
the failed request, retaining its $0.001172 reservation under the original $15
ceiling. This does not reset the stage budget or silently retry other failures.

Offline recovery checks cover successful replacement and cached resume, a second
lost response, insufficient budget, orphan evidence, interruption immediately
after retirement is committed, and unchanged earlier response records. A bounded
GPT-6.1-Sol reviewer also identified a publication-order issue: verification and
retained-cost evidence now publish together, rather than briefly exposing an
incomplete success report.

## Completed Calibration

All **104 planned observations completed: 100 valid graphs and four terminal
global failures**. Each method had 26 planned observations. Local, sequential
and iterative each produced 26 valid graphs; global produced 22 valid graphs.
Four of those 22 global graphs are genuinely empty, so only 18 global graphs
are nonempty. Empty is an allowed answer, not a missing observation.

The four invalid global responses contain three duplicate-friendship lists and
one out-of-roster ID. We neither repaired their edges nor purchased replacements
to turn them into successes. Global missingness is 4/26 (15.38%); interpreting
only its valid outputs can introduce selection bias. A larger study must report
missingness by condition and specify its handling before collection.

All four valid empty graphs came from GPT-6-Luna: US/English repetition 0,
Brazil/English repetition 1, US/Portuguese repetition 1, and Japan/English
repetition 1. The prior V6 batch had 22/26 empty global graphs. This is a
descriptive difference between versions, not proof that revised wording alone
caused improvement or that generated ties became more realistic.

The separately authorized HTTP-520 replacement succeeded. It reused all 139
received prefix replies, just as the earlier recovery reused 236. Both original
reservations remain in spending. The final three cells added $0.086225 beyond
the $7.8050634 checkpoint. Frozen prompts, earlier responses and original
authorization anchors stayed unchanged.

| Item | Measured status |
| --- | --- |
| Wording screen | 96 observations; 94 valid graphs, two terminal failures; $5.9461527 |
| Calibration received responses | 11,634 responses; $7.889144775 |
| Retained HTTP-500 reservation | $0.001172; exact replacement succeeded |
| Retained HTTP-520 reservation | $0.000971625; exact replacement succeeded |
| Calibration total usage plus reservations | **$7.8912884 of the original $15 allowance** |
| Calibration recorded API duration | 12,096.272 seconds, or 3.360 hours |
| Separate Hindi probe | Four observations, one valid graph/three failures, $0.30618 of $1; candidate not adopted |
| Regression checks | 191 offline tests in 455.082 seconds; final recovery tests 2/2; clean publishing checks 7/7 in 20.057 seconds |

Costs are conservative recorded-token estimates and retained reservations, not
reconciled invoices. API duration is summed received-request time; it excludes
offline replay, interruption/review pauses and other operational overhead.
The old `checkpoint_inspection.json` remains a historical partial audit, not the
final clearance. Use the current `inspection.json` bound to the completed
summary. Neither report authorizes main collection.

Both stages also replayed successfully in the clean publishing checkout: 108
observations checked with network blocked, no key and no private ledger. The
read-only reviewer found no remaining material claim errors in the final report.
Only whitelisted public artifacts are published; private logs and snapshots stay local.

## Updated Cost And Runtime Scenarios

These are fresh **attempted graph cells**, not promised valid graphs. Calibration
is not silently reused as confirmation data. All use 50 personas, four models,
four methods and the same seven settings described above.

| Fresh allocation | Attempts | Estimated usage | With 25% cost contingency | Serial API hours | With 50% time allowance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight repetitions in every cell | 896 | $102.44 | $128.05 | 26.30 | 39.46 |
| Global 32 repetitions; other methods eight | 1,568 | $104.49 | $130.61 | 27.03 | 40.54 |

The larger-global scenario adds approximately $2.04 and 0.73 API hours because
global makes one request per graph, while iterative makes hundreds. This is a
cost tradeoff, not automatic statistical approval. Four countries times four
languages is **not** the design: the US/English overlap is counted once.

For 288 of the 896 attempts (504 of 1,568), the other models' unmeasured non-US
costs/time are estimated using GPT-6-Luna country/US ratios. Those ratios have
only two repetitions. The contingencies are planning buffers, not confidence
intervals or guaranteed spending ceilings. Prices, provider latency, corrections,
failures and transfer assumptions can change the outcome. Wall-clock delivery
also needs replay/review time; no concurrency speedup is assumed.

## Completion Checklist

- [x] Obtain separate exact HTTP-500 and HTTP-520 replacement approvals; retain
  both reservations and preserve the 236- and 139-response prefixes.
- [x] Complete all 104 observations within the original calibration allowance;
  retain four invalid and four valid-empty global outcomes.
- [x] Finish final private accounting and public replay of all 104 observations;
  [full results and reproducible evidence](V8_COMPLETION_RESULTS.md).
- [x] Complete and inspect the separate Hindi comparison: original 0/2 valid,
  clarified 1/2 valid. The extra instruction still produced multiline answers.
  Do not adopt it or claim the Hindi alternative is fixed. The already selected
  calibration reference is a different wording and remains unchanged.
- [ ] Resolve the research decisions below and authorize a finite main-study
  allocation. No main-study request has been sent.

## What The Professor And Team Should Review

1. Are country-framing and instruction-language effects on this fictional roster
   an appropriate claim scope, rather than claims about actual national cultures?
2. Are homophily and conditional topology the right primary outcomes for the
   quota-based methods, with constrained density treated descriptively?
3. Is unrestricted global generation, including valid empty graphs, the intended
   task? Invalid graphs must remain missing observations with failure rates shown.
4. Which effect size or precision matters scientifically, and which comparisons
   form the multiple-testing family? Eight or 32 repetitions is not a universal
   conference-validity threshold. Nodes and edges are not independent repetitions.
5. Is the absent human bilingual review an acceptable disclosed limitation?
   Neither the implementation agent nor its AI reviewer can certify it.
6. Which fresh main-study allocation and finite budget should be authorized?
   Tuned calibration samples must not silently become confirmatory evidence.

## Release Boundary

The main study, new population rosters, simulator redeployment, and professor
email delivery are outside this calibration-completion change. Review the code,
evidence and proposed design first. No software or AI review guarantees research
validity, absence of every defect, or conference acceptance.
