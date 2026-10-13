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
| A provider HTTP 500 left billing uncertain | One explicitly authorized replacement retains the original full reservation and has its own deterministic identity | A local conservative reservation is not the provider invoice |
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

## Current Checkpoint: Not Complete

The first recovery succeeded. Calibration advanced from 63 to **101 of 104
completed observations: 97 valid graphs and four terminal global failures**.
The failures are three duplicate-friendship responses and one response containing
an ID outside the roster. They remain failures, not deduplicated or regenerated
successes. This remaining global-format missingness needs scientific attention
before a larger study, even though the parser correctly rejects it.

A second provider interruption, **HTTP 520**, stopped the next iterative graph
after 139 received replies. It affects GPT-6-Luna, Japan framing, English,
repetition 1. Two later cells have not been attempted. The first replacement
permission did not authorize an unlimited retry loop. Another exact replacement
has been requested, not performed.

| Item | Measured status |
| --- | --- |
| Wording screen | 96 observations; 94 valid graphs, two terminal failures; $5.9461527 |
| Calibration received responses | 11,075 responses, including the partial graph; $7.802919775 |
| Earlier retired HTTP-500 reservation | $0.001172 retained; one authorized replacement succeeded |
| Current unresolved HTTP-520 reservation | $0.000971625 retained; no replacement sent |
| Calibration total charged or reserved | **$7.8050634 of the original $15 allowance** |
| Further calibration accounting since the earlier stop | $2.4773969; no old responses repurchased |
| Separate Hindi probe | Four planned graph cells under $1; not run because completed-calibration clearance is absent |
| Regression checks | 189 offline tests passed in 442.480 seconds; five focused publishing-checkout tests passed in 7.752 seconds |

Costs are conservative recorded-token estimates and retained reservations, not
reconciled invoices. Completed calibration cells alone account for $7.7901589
and 3.156 serial API hours. That excludes the unfinished cell and is **not a
full-study runtime forecast**. Updated main-study forecasts are withheld until
all 104 observations have completed and been inspected. Historical ~$103
forecasts must not be presented as a new quote.

The stopped summary and separate `checkpoint_inspection.json` distinguish partial
verification from whole-stage clearance. They cannot unlock the Hindi probe or
main collection. A passing software suite does not make these four malformed
global responses valid or certify the scientific design.

The checkpoint audit replayed all 101 terminal observations and the 139 received
responses in the unfinished cell, with the private ledger unchanged and zero
API calls. A separate check in the clean publishing checkout replayed all 101
public terminal observations without a key or private ledger, including graph,
metric, saved-event and artifact checks. It does not independently reconcile
provider billing or expose the unfinished cell as a completed graph.

## Exactly What Remains

- [ ] Obtain one bounded decision for the HTTP-520 request, retain its original
  reservation, and preserve all 139 received replies.
- [ ] Complete the interrupted cell and the two unattempted cells, under the same
  calibration allowance; do not replace terminal parser failures.
- [ ] Perform final private accounting and public replay, then issue fresh
  cost/runtime scenarios from the completed calibration.
- [ ] Run and inspect the already authorized separate Hindi comparison. Its
  results cannot retrospectively change either frozen V8 experiment.
- [ ] Resolve the research questions below and authorize a finite main-study
  allocation only after those decisions.

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
