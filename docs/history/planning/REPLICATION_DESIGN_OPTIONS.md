# Replication Revision: Decision Before Collection

**Decision recorded:** retain four models and four methods with eight repetitions,
896 total. The author removed the $50 maximum and authorized $5 additional
calibration only. See [RESEARCH_896_HANDOFF.md](RESEARCH_896_HANDOFF.md).
The alternatives below are historical planning context, not the active protocol.

Status: proposed allocation change, not an approved execution protocol.
The author reports that Astra reviewed the preceding implementation. Changing
the allocation changes the protocol hash and requires an updated review receipt;
it does not authorize API calls or clear the other review gates.

## Why Two Repetitions Cannot Be Fixed by Renaming Outputs

The existing design has 4 models x 4 methods x 7 settings = 112 conditions.
With 224 graphs, each condition receives exactly 2 runs. Seeds do not enlarge
the persona population, and 50 nodes or 1,225 possible ties do not provide
independent replicate networks. More bootstrap draws cannot create more
collected runs either.

The seven settings preserve four country framings in English and four
instruction languages under US framing, counting US-English only once.

## Allocation Choices

| Design | Models | Methods | Settings | Runs per condition | Total networks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current, insufficiently replicated | 4 | 4 | 7 | 2 | 224 |
| Narrower replicated primary study | 2 | 2 | 7 | 8 | 224 |
| Full panel, intermediate allocation | 4 | 4 | 7 | 5 | 560 |
| Full panel, eight-run allocation | 4 | 4 | 7 | 8 | 896 |
| Full panel, ten-run allocation | 4 | 4 | 7 | 10 | 1,120 |

**Recommendation under a firm graph limit:** narrow the primary study rather
than spread the same observations over too many conditions. Two models and
two methods still permit country, attribute, model and instruction-language
questions, but conclusions must name that narrower panel. Choose the models
and methods before seeing fresh results; do not select favorable comparisons.
The earlier four-method archive remains historical, not a substitute for fresh
replicates. No particular pair has been silently selected.

**If all four models and all four methods are mandatory:** increase the graph
target, then test whether the requested precision and cost can both be met.
Retain the cumulative $50 ceiling. Do not promise that 896 or 1,120 will fit,
and do not silently finish whichever cheap conditions happen to fit first and
call the resulting unbalanced study complete.

## What Makes the Replication Defensible

- [ ] Specify the smallest scientifically meaningful difference for each
  primary metric, or the desired interval precision, before testing effects.
- [ ] Use limited, explicitly authorized calibration to measure token costs,
  parse failures and variability of the actual matched condition differences.
  US-English-only variability does not establish variability for every language
  or country contrast. A tiny pilot variance is itself uncertain.
- [ ] Evaluate sample-size sensitivity across plausible variances and effects;
  freeze a feasible design before the main collection. Eight or ten runs are
  planning alternatives, not evidence-backed sufficiency thresholds.
- [ ] Match roster and local randomization settings across comparisons, retain
  every collected run and failed attempt, and analyze at the network-run level.
  Different API requests are repetitions; provider seeds are not guaranteed
  deterministic controls. Record resolved model identifiers and collection time.
- [ ] Distinguish within-roster generation variation from between-roster
  variation. Eight generations on one roster do not establish population
  generality. Multiple rosters require an explicit nested allocation and an
  analysis that respects roster clustering, not simply pooling all graphs.
- [ ] If the budget cannot support the chosen inference, narrow the question
  or report the exploratory limitation. Never stop collecting because a result
  becomes significant or retrospectively claim sufficient power.

These requirements follow sample-size justification based on inferential goals,
effect sizes, precision and resource constraints, not a conference-specific
minimum. Source: Daniel Lakens, [Sample Size Justification (2022)](https://research.tue.nl/en/publications/sample-size-justification/),
DOI: 10.1525/collabra.33267; [author's sample-size chapter](https://lakens.github.io/statistical_inferences/08-samplesizejustification.html).

## What Changes After the Allocation Decision

- [ ] Update protocol dimensions, repetition/roster allocation and total count.
- [ ] Remove hard-coded two-repetition assumptions from manifest, checks,
  analysis coverage and tests; keep exact reviewed scope validation.
- [ ] Regenerate the planned manifest, offline fixtures and review hashes.
- [ ] Update notebook, README, handoff and PDF response matrix consistently.
- [ ] Re-review changed paths and obtain bounded spending authorization.
- [x] Preserve historical adjacency, PNGs, receipts and published measurements.
- [x] No fresh results exist to rewrite; no additional API calls are authorized.

The active 224 protocol remains unchanged until the author chooses which
constraint may change. This document is a proposal, not generated evidence.
