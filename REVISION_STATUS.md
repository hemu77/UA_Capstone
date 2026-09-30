# Research Revision: Evidence, Fixes and Remaining Work

**Historical revision log.** Earlier 24-pilot/640-study counts and $50 planning
below describe an earlier stage, not current authorization. Current evidence is
68 completed calibration graphs toward a proposed 896. Read
[the team brief](docs/TEAM_KNOWLEDGE_BRIEF.md) and
[calibration results](CALIBRATION_RESULTS.md) for the present status.

This report tracks implementation, not a promise of acceptance. The supplied
review-action document is feedback to investigate, not an executable specification
or proof that a claimed experiment exists. The authenticated browser opened the
requested architecture-review skill; its evidence/confidence format was adapted
into `.agents/skills/review-architecture/SKILL.md`. One read-only reviewer checked
the bounded implementation. The original repair phase used no paid requests;
the subsequent authorized engineering pilot is tracked below and in its ledger.

**Legend:** `[x]` completed and checked; `[ ]` unfinished or unresolved. Green
software checks do not turn unfinished research into a positive research result.

## Budgeted Implementation Update

- [x] $5 engineering-pilot limit inside a $50 cumulative limit; SQLite reservations
  survive restarts and retain uncertain charges. Source: `paid_study.py`.
- [x] Real provider usage, request settings, resolved model, response ID and
  fingerprint are recorded locally; completed graph/PNG hashes are verified.
- [x] Twenty pilot cells and 640 draft main cells enumerate all fifty personas,
  GPT-6 Luna/GPT-4.1 Mini and all four methods. Pilot completion is tracked separately.
- [x] Viewer header/controls are compact; both graphs use one deterministic union
  force layout. JSON downloads include the displayed algorithmic coordinates.
- [x] Scalar analysis canonicalizes node/neighbor order. Reanalysis version
  `revision-v1.1-canonical-order` changed 54 historical modularity values (largest
  absolute change 0.023504); density, mean clustering and largest-component shares
  did not change. Some LCC path statistics changed when equal-size components
  were tied; canonical order now fixes the tie choice. Earlier scalar tables and
  their manifest are preserved as `*.insertion_order.*`.
- [x] Paid engineering pilot: **24/24 verified**, all fifty personas per network
  (20 original plus four Sol US-English method checks); **$1.886188** conservative
  cost, **2,702 attempts**, zero unresolved requests. Luna has 16, Mini four,
  Sol four. The $5 shared pilot cap and draft 640-run main scope are unchanged.
  Sixteen parser failures are annotated; 454 early responses lack annotations,
  so the full parser-failure rate remains unknown. Raw failed replies and costs
  remain in the ledger. Prompt variants are explicitly labeled.
- [x] Multi-select model/method/country/instruction-language/seed filters drive
  topology dots and demographic-mixing heatmaps from shared Python measurements.
  Missing combinations, valid counts, CSV/SVG/JSON exports and URL-restored
  selections are explicit. Long evidence notes are collapsed, not removed.
- [x] 528 offline controls on the 176 retained historical graphs, with matching
  node/edge counts and explicit degree-rewiring completion flags. These are
  exploratory synthetic baselines, not real-network validation or new LLM runs.
- [ ] Astra review: after actual pilot results, before main protocol freeze.
  Use [ASTRA_REVIEW_HANDOFF.md](ASTRA_REVIEW_HANDOFF.md) with the real pilot artifacts.
- [ ] Five adult rosters, reviewed translations, final uncertainty plan and main
  collection still require scientific work; this update does not mark the project complete.

## 1. What Changed and Why

- [x] **The friendship contract is explicit.** Networks are simple, undirected
  graphs: if either persona selects another, one edge exists. Reversing the order
  of the two IDs does not create a different tie. Historical files cannot recover
  who initiated a tie, reciprocity, or a chronological sequence.
- [x] **Distance and homophily calculations were repaired.** Edge disagreement
  divides by N(N-1)/2, not N(N-1), for undirected graphs. Categorical Coleman
  scores use within-group incident ties and population shares. Age uses numeric
  assortativity, not an arbitrary ten-year categorical ratio. Group-level scores
  are retained so undefined groups can be inspected.
- [x] **Invalid model replies cannot partially alter the graph.** The parser
  validates IDs, duplicate choices, self-links and eligible add/drop targets
  before mutation. Three attempts is the maximum. Permanent configuration errors
  stop immediately, and SDK retries are disabled underneath the explicit loop.
- [x] **New draft prompts separate setting from instruction language.** Parallel
  country statements no longer assign a language to the fictional participants.
  Portuguese and treatment-language retry instructions are included. New output
  prefixes include `revision-v1`, preventing overwrite of historical experiments.
- [x] **Real graph-change events are recorded by the generator.** An offline test
  replays these changes and recovers the final graph. Old graphs have no events;
  their replay is disabled. New paid attempts have a local usage/provenance ledger;
  earlier pilot responses without parser annotations are explicitly counted.
- [x] **The notebook is usable offline.** Original exploratory cells are kept
  behind an explicit opt-in, since some write files or require missing datasets.
  New cells verify hashes and load corrected outputs. Old saved outputs were
  cleared to avoid displaying stale numbers next to repaired code.
- [x] **The 3D viewer reads the shared Python results.** It does not implement a
  competing homophily formula in JavaScript. A shared union-force layout makes the
  same persona occupy the same position in both views. Spatial closeness in that
  display is not a finding about social or geographic distance.
- [x] **Manual edits are isolated.** Sandbox changes exist in browser memory only.
  Research metrics are hidden and downloads are labeled non-research. Reloading,
  discarding edits or changing the reference condition returns to saved evidence.
- [x] **Paid generation is locked before client creation.** A supplied key is not
  a spending limit. The old runner has no enforceable reservation ledger or
  durable failed-attempt accounting. `PAID_GENERATION_READY=False` is an explicit
  release gate for archived runners. Only `paid_study.py` uses the authorized
  budgeted path; the legacy calls remain blocked.
- [x] **Budgeted engineering runner implemented and tested.** Model access,
  output limits, durable reservations, actual usage, raw request/response
  provenance and artifact-checked resume are implemented. Parser outcomes are
  additional annotations, not replacements for raw paid replies. Main research
  collection remains blocked at the protocol/Astra review gate.

## 2. What the New Audit Found

| Historical study | Inventoried | Graph checks passed | Quarantined |
|---|---:|---:|---:|
| Cultural / sequential | 24 | 24 | 0 |
| Additional methods | 72 | 68 | 4 |
| Fixed-US language | 96 | 84 | 12 |
| Total | 192 | 176 | 16 |

The excluded graphs contain **23 self-links**, all in historical nano outputs.
The old validation checked that there were 50 nodes and some edges but did not
reject a person being connected to themselves. We did not delete those links and
call the experiment fixed. The original `.adj` files are unchanged, while
`stats/revision_v1/quarantined_runs.csv` lists the exclusions. Missing/corrupt PNGs
are a separate artifact check and cannot quarantine a structurally valid graph.

This is an **8.33% observed graph-invalidity fraction among saved files**, not an
API refusal or parse-failure rate. Those rates require attempt logs, which are
missing historically. Exclusion is not random across models and conditions, so
the retained results can be biased. Some condition summaries have one available
run instead of two. Nineteen homophily measurements remain undefined, rather
than being silently replaced with zero or rescaled.

The roster itself adds another concern: ages span **0 to 89**, with **nine people
under 18 and four under five**, all carrying political-affiliation fields. An
infant selecting friends as an adult partisan is not a defensible political
social-network model. For the revised study, use adults 18+ if political
affiliation remains a central question; define another design explicitly if
intergenerational or child networks are the intended subject. Preserve the old
roster only for historical reproduction.

## 3. What Each Research Question Means Now

**RQ1: Does country framing matter?** The historical data contain differences.
For sequential GPT-4.1-mini in English, mean largest-component shares over two
seeds are US 0.50, India 0.98, Japan 0.76 and Brazil 0.52. The exact US values are
0.48 and 0.52; Japan varies from 0.52 to 1.00. This explains why pooled means and
individual-run values must never be presented interchangeably. It does not
establish a stable cultural effect. A fixed US roster tests response to framing;
country-grounded rosters test a different question involving population changes.

**RQ2: Which attributes matter?** In the historical US-English mini runs,
political Coleman means are global 0.8317, local 0.9957, sequential 1.0000 and
iterative 1.0000. A high same-group association does not prove causal dominance.
Do not rank a ratio, a Coleman score and age assortativity as if they share a
single scale. Only supplied attributes belong in a prompt-influence analysis;
the study runners did not include names or interests. Controlled masking and
independent, realistic adult rosters are still needed.

**RQ3: Are models consistent?** Matched edge comparisons are now orientation-safe
and seed-aware. Under US-English sequential conditions, disagreement means are
0.1045 (mini/full), 0.2371 (mini/nano) and 0.2429 (nano/full). These are fractions
of all possible ties, not mistakes against a known true social network. Models
can agree because they share training or prompt biases. The current evidence
does not support claims about other providers or all LLMs.

**RQ4: Does translating the prompt matter?** The historical US sequential mini
mean densities are English 0.1718, Hindi 0.1743, Japanese 0.1731 and Spanish
0.1784. These small descriptive differences have no demonstrated significance.
The older prompt changed fictional spoken language too, so it cannot isolate
instruction translation. New neutral context wording removes that particular
code-level confound, but human translation review and new controlled runs are
still required. Portuguese support is not Portuguese experimental evidence.

## 4. Feedback-to-Implementation Checklist

No issue is marked complete merely because a plan mentions it.

| Feedback IDs | Status | Implemented or required action |
|---|---|---|
| A1, A3: other families / broad model panel | [ ] | OpenAI-only scope stated. Cross-provider evidence unresolved; no invented substitute for requested models. |
| A2: full crossed model protocol | [ ] | Draft crossed protocol and offline request estimator exist; no revised paid matrix has run. |
| A4, A5: temperature / prompt wording | [ ] | Prespecified panel in protocol; parameterized ablations and results still needed. |
| B1, B2: replication / uncertainty | [ ] | Actual sample counts shown. No confidence intervals claimed from two seeds; independent rosters and pilot-informed precision design required. |
| C1, C4: benchmark inventory / comparison | [ ] | Eight network IDs observed in saved real-network metrics. The larger claimed benchmark and inclusion flow still need source-level reconciliation. |
| C2, C6, C7: excessive realism claim | [x] | Current framing is an audit of simulation behavior. No universal baseline superiority, true-edge validity or real-culture claim. Old narrative is explicitly archived. |
| C3: shortest paths | [ ] | Raw largest-component path metrics exported with explicit `_lcc` names. Full empirical benchmark figure still pending. |
| C5: feature-similarity baseline | [ ] | Must run demographic-similarity, density-matched random and degree-preserving baselines with the same metric pipeline. |
| D1: corrected homophily | [x] | Shared categorical Coleman and numeric age functions, group-level outputs and hand-checkable tests added. Old ratio results are not relabeled as Coleman. Paper revision still required. |
| D2: LCC inconsistency | [x] | Per-run and explicitly grouped tables separate seeds, methods and contexts; concrete US/Japan values documented above. Unavailable submission-specific figures cannot be retroactively certified. |
| D3: figure readability | [x] | New labeled, fixed-condition comparison PNGs created and inspected. Original paper artwork remains outside this change. |
| E1: US roster used as other populations | [ ] | Limitation made prominent; adult/source-grounded/comparable rosters not yet constructed. |
| E2: culture-language confounding | [ ] | Draft prompt wording repaired. Full crossed, reviewed experiment still pending. |
| E3: persona-to-prompt mapping | [x] | Visible fields, omitted names/interests, IDs and undirected semantics documented in code, notebook and README. |
| F1: local candidate sweep | [ ] | Current repository offers all other personas; feedback mentions k=12. Reconcile the submitted implementation first, then implement paired candidate controls. |
| F2: iterative ablation | [ ] | Current initialization is local, not the sequential initialization described in feedback. Initialization/rounds ablation and source reconciliation remain open. |
| G1, G2: novelty / motivation | [x] | Repository positions the work as an audit and controlled decomposition, not a new proven-realistic generator. Paper needs the same rewrite. |
| G3: mitigation / mechanisms | [ ] | Attribute masking and paired intervention are specified, not executed. |
| G4, G5: clarity / limitations | [x] | Four explicit RQ sections, plain-English definitions, source links and concrete consequences of limitations added. |
| G6: roster size / composition | [ ] | Draft 50/100/150 panel; multiple independent adult rosters still required. |

### Additional issues from the supplied document

- [ ] Conflicting GPT-4o 0.39 versus 0.999 claims, missing GPT-5.1 evidence, and
  publication-specific numbers require their original files. Do not transplant
  a result from a different model or aggregation to fill those gaps.
- [x] No universal political-dominance claim is made. Demographic scores are
  available per retained run, with age reported separately.
- [ ] Benchmark counts, size/density matching, prompt friendship budgets and
  empirical homophily reference values still need a preregistered comparison.
- [x] Historical graphs are undirected, contrary to one feedback assertion.
  Directional claims are withheld rather than computed from invented arrows.
- [ ] Multiple-comparison correction, model snapshots, refusal/retry rates,
  actual usage and human-reviewed translations remain outstanding.
- [x] Primary prompts already omit names; exported personas omit names/interests.
- [ ] Broken submission LaTeX/metadata cannot be repaired in these repository
  files; update the manuscript and submission system separately.
- [x] Responsible-use warning added. Synthetic demographic ties must not guide
  decisions about real people; mitigation effectiveness still needs evidence.

## 5. Cost and Next Gate

`plan_revision_study.py` makes no SDK requests. Its draft pilot is 20 networks,
up to **2,255 logical requests or 6,765 attempts** under the three-attempt policy.
The reduced **640-network** draft is up to **72,160 logical requests before
retries**, excluding population-grounding, broad robustness panels and unselected
country-language interactions. The previous 2,560-network full-cross scenario
is not the current budgeted design. Counts are ceilings, not a power calculation.

Account access was verified for GPT-6 Luna and GPT-4.1 Mini. The estimate is about
$36 with contingency for pilot plus reduced main scope, with a $5 pilot cap inside
$50 cumulative. Actual usage and failed responses remain charged in SQLite.
Next: Astra review of pilot failures/costs, adult roster eligibility, translations,
effect-size targets and uncertainty before any confirmatory collection. A single
roster engineering pilot cannot estimate between-roster variance. Rotate keys
exposed in chat after use; never store them in notebook, browser, repo or report.

## 6. Verification Scope

The runnable Python and JavaScript tests cover repaired arithmetic, parsing,
retry limits, full-50 mocked generation across four methods/five languages,
event replay, budget lock, source/export parity, quarantine, and display helpers.
The notebook executes its maintained section offline while explicitly skipping
32 archived cells. Browser checks cover graph rendering, filtering, persona
selection, sandbox isolation, self-link rejection, downloads and responsive layout.
See [validation report](stats/revision_v1/validation_report.md) for observed results.

The reviewer identified the paid-run authorization gap and missing failure/usage
records; the shared dispatch lock remains for old runners, while the new budgeted
runner implements these controls. A missing-PNG quarantine bug was corrected and tested.
No full unrelated-repository security audit or conference-readiness certificate
is implied. Publication and confirmatory research remain gated; original artifacts and
pre-existing user changes have not been discarded.
