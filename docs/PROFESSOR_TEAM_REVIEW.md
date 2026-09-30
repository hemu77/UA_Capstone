# Review Request for Professor and Project Team

## Requested Decision

Please review the project **through the completed calibration stage**, not as a
finished paper. We request a decision on research scope and validity assumptions
before committing approximately **$101 additional API usage** to the remaining
828 networks. No main batch is running.

The [team knowledge brief](TEAM_KNOWLEDGE_BRIEF.md) explains the original work,
reviewer concerns, model-panel rationale, complete planned scope, completed
implementation and remaining limitations. [Calibration results](../CALIBRATION_RESULTS.md)
contains the measured accounting and verification record.

## Evidence Available for Review

- 68 verified, 50-person current-protocol calibration graphs and 204 offline controls.
- Four model configurations; all four methods; seven country/language settings.
  Full calibration across settings is on GPT-6-Luna, not all four models.
- Corrected parsing/metrics, recorded decision replay, source/roster provenance,
  token usage, budget controls and explicit handling of failed requests.
- 88 passing Python tests; 896 offline fixture checks; maintained notebook loading.
- AI-assisted translation review, clearly distinguished from human validation.
- Preserved earlier results and reviewer-response history, not relabelled new data.

## Questions for Scientific Review

1. Is the intended contribution appropriately limited to model-generated network
   sensitivity on a designed fictional roster, rather than national human behavior?
2. Is the purposeful four-configuration, single-provider panel defensible for
   RQ3, with no architecture-only or cross-vendor generalization claim?
3. Does the instruction-only language manipulation, with fixed English persona
   fields, answer the intended RQ4? What bilingual review is required?
4. Are eight repetitions on one roster adequate for the chosen precision goals,
   or should the allocation change before money is spent?
5. Are the primary outcomes, paired comparisons, uncertainty approach,
   multiplicity treatment and failed-run/correction reporting sufficiently specified?
6. Does the intended venue/contribution require empirical network validation,
   additional rosters, prompt/temperature robustness or method ablations now?
7. Do the documented fixes address the engineering concerns without overstating
   resolution of the reviewers' remaining scientific concerns?

Please record **approve unchanged**, **revise first**, or **defer**, with reasons
and required evidence in [TEAM_APPROVAL.md](TEAM_APPROVAL.md). Approval is a
research decision, not a promise of zero bugs or guaranteed conference acceptance.

## Cost and Timing for the Proposed Next Stage

Remaining 828: approximately **$101.00**, or **$121.20** with 20% cost allowance.
The suggested cumulative ledger ceiling is **$128**, including $6.47334 already
counted. Remaining serial API time is approximately **28.44-42.66 hours**, plus
local processing, analysis and pauses. These estimates assume the present
protocol and transfer Luna treatment ratios to other models; they are not quotes.

If design changes are required, calibration reuse and cost must be reassessed.
Once scientific decisions and a finite budget are explicitly approved, the
execution owner can validate the matching review gates and proceed. Until then,
review/documentation work is offline and paid collection remains stopped.

## Optional Engineering Review Helper

The repository includes a reusable [read-only reviewer](../agents/README.md).
Team members may use its checklist or task prompt to inspect a bounded code/data
change. It cannot authorize spending, supply human bilingual validation or certify
scientific validity. Review findings must cite evidence rather than invent issues.
