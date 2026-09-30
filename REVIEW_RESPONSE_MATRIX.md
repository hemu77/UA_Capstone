# Review Response and Pre-Spend Decision

## Current Allocation Update

**Current handoff:** all 68 calibration graphs are verified; the 896-network
main batch remains unapproved. [The team brief](docs/TEAM_KNOWLEDGE_BRIEF.md)
explains what is fixed, what remains a research decision, and the ~$101 remaining
cost. [Team approval](docs/TEAM_APPROVAL.md) must precede further paid collection.

The author has now selected **eight repetitions, 896 total graphs**, retaining
all four models and all four methods, and authorized **$5 additional calibration
only**. The old $50 maximum is removed; the main batch remains separately gated.
See [RESEARCH_896_HANDOFF.md](RESEARCH_896_HANDOFF.md). This replaces the earlier
224 allocation discussed below. More repetitions address the count deficiency,
not proof of statistical power, translation equivalence or population validity.
The earlier review record is preserved rather than silently rewritten as a pass.

Evidence: the six-page `reviews.pdf` supplied by the author, read on September
30, 2026. Page numbers below refer to that PDF, not to a new external review.
Author rebuttal statements in the PDF are historical claims, not independently
verified results. No paid API calls were made for this audit.

## Historical 224-Run Decision (Superseded Allocation)

**Do not launch the complete study yet.** Software repairs cannot make missing
scientific evidence appear. The proposed scope is **224 networks total**, not
224 per method: 4 models x 4 methods x 7 settings x 2 repetitions. That is 56
per model and 56 per method, with 50 fictional adults in every graph.

The most important remaining issue is explicit in the final rejection on PDF
page 2: two culture/language runs per condition were insufficient. This reduced
design still has two. It can support a transparent exploratory study, not a
claim that the rejection has been fully resolved. No graph-count threshold
guarantees acceptance, and no software review guarantees zero defects.

## Reviewer Concerns

- [x] **Wrong homophily definition (pp. 3-4): code corrected.** Categorical
  Coleman homophily uses `(within_group_share - population_share) /
  (1 - population_share)`. Group evidence is retained. Undefined groups stay
  undefined rather than becoming zero. Age uses numeric assortativity and is
  not ranked against categorical coefficients as if they measured importance
  on a shared scale. Fresh analysis exposes each attribute's control comparison.
- [x] **Country and language mixed together (pp. 4-6): fresh manipulation
  separated.** Country framing changes only the stated social setting.
  Instruction language changes independently. Candidate attributes remain
  identical English JSON; participants' spoken language is not assigned.
  Invalid country/language inputs are rejected. Portuguese is present in all
  fresh prompt actions. This is instruction-language sensitivity, not a fully
  translated-persona experiment.
- [x] **US-specific people treated as national populations (pp. 2-6): claim
  restricted and roster replaced for fresh work.** The new adults have declared
  synthetic marginals, no US party labels and no US racial classification.
  They are not representative of any country. Removing US labels does not make
  left/center/right or religious categories culturally universal. Country-label
  sensitivity is the defensible target; national friendship behavior is not.
- [x] **Missing attribute-based reference (pp. 4-5): fresh analysis connected.**
  Each collected fresh graph receives node/edge-matched random, equal-weight
  attribute-similarity, and degree-preserving controls using the actual fresh
  roster. Graphs, scores, source IDs and roster hashes are saved separately.
  These controls are reference rules, not empirical friendship observations;
  one control draw is not a null distribution or significance test.
- [x] **Method description differs from implementation (pp. 5-6): fresh
  semantics made explicit.** Local sees all other 49 personas, not a sampled
  neighborhood of 12. Iterative starts from local generation, not sequential,
  then performs three add/drop rounds. Sequential starts with three local-style
  prompts. Events now identify both overall method and actual prompt action.
  These are whole-protocol comparisons, not isolated causal method effects.
- [x] **Inconsistent metric aggregation (pp. 3-4): fresh outputs retain
  individual runs.** Topology is recomputed from saved adjacency, LCC measures
  explicitly refer to the largest connected component, and paired comparisons
  identify both source runs and the direction of subtraction. No pooled mean
  silently replaces two repetitions or hides a missing comparison partner.
- [ ] **Too few repetitions and only one roster (pp. 2, 4-6): not resolved.**
  The 224 matrix preserves breadth at the expense of replication. More repeat
  runs and independently designed rosters would be needed to assess stability
  beyond this roster. Do not treat 1,225 possible edges as 1,225 independent
  experiments, or label two runs a reliable confidence interval.
- [ ] **Realism and contradictory benchmark claims (pp. 2-6): not resolved by
  new code.** A defensible empirical comparison needs a documented dataset
  inventory, provenance, parsing/exclusion rules, network-type compatibility,
  and all declared metrics. The PDF itself records changing benchmark counts.
  Existing archive files and synthetic controls do not verify those counts or
  justify saying generated ties are realistic. Fresh work withdraws that claim.
- [ ] **Model, temperature and wording robustness (pp. 2-6): not demonstrated.**
  Four OpenAI configurations are not cross-vendor evidence. Different supported
  decoding settings also prevent attributing every difference to architecture
  or model size. Temperature, equivalent-wording and open-weight replications
  remain separate, unrun studies. One reviewer's requested 20 models is not a
  universal conference minimum.
- [ ] **Local-neighborhood and iterative ablations (pp. 5-6): not run.**
  Candidate-set sizes, random neighborhoods, initialization and round counts
  need actual experiments if robustness to those choices is claimed. Correct
  documentation of the current settings does not replace these experiments.
- [ ] **Translation equivalence: independent review pending.** A bilingual
  review must check all instructions, retries and the meaning of fixed persona
  fields. Equal JSON bytes and parser tests prove separation, not semantic
  equivalence. Review decisions must match the frozen prompt hashes.
- [ ] **Mechanism, mitigation and contribution beyond prior work (pp. 4-6):
  still a research task.** A more attractive simulation does not establish a
  mechanism or a novel finding. Interpret observed contrasts after collection;
  do not invent explanations, mitigation success or model results in advance.

## Engineering Repairs From the Two Read-Only Reviewers

- [x] Completed runs now reconcile provider IDs, usage, parse receipts, request
  identity and charges with the original spending ledger before further calls.
  A restored ledger missing fresh charges cannot silently reopen the budget.
- [x] An OS-backed exclusive lock serializes fresh generation and analysis;
  a second process stops before dispatch. Process exit releases the lock.
- [x] JSON publication uses a unique temporary file and atomic replacement.
  An interrupted artifact write resumes from paid cached replies, not a new call.
- [x] Preflight binds its starting source fingerprint and fails if sources change
  during checking. It cannot certify an edited file it did not test.
- [x] Replay records the actual first-three local prompts within sequential runs.
  Batched decisions remain batched; no invented per-edge chronology is recorded.
- [x] Fresh analysis is explicit (`revision224.py --analyze`). Historical results
  are never substituted when fresh receipts are absent. A missing study returns
  `NOT_COLLECTED`, not a populated table of fixtures or archive results.

## Historical Gates (See Current Team Approval Checklist)

- [ ] Astra reviews the frozen implementation, tests and this response matrix.
- [ ] Bilingual instruction/retry review is completed and documented honestly.
- [ ] The author accepts the exploratory claim boundary or revises the allocation
  for stronger replication. This is a scientific decision, not a UI preference.
- [ ] Confirm model access, supported settings and current prices before spending.
- [ ] Replace credentials exposed in chat and supply the replacement through
  the local environment, never source files, notebooks or review documents.
- [ ] Obtain explicit permission for at most 16 calibration cells, counted toward
  the 224, with a cumulative $5 ceiling including the existing pilot charges.
- [ ] Inspect actual token usage, retries, truncations, outputs and condition
  differences. Review a conservative completion forecast against the remaining
  cumulative $50 budget, then request separate permission for the full batch.

The cap stops spending; it cannot promise that every planned graph will fit.
If reviewers need additional evidence, do not relabel an exploratory study
"conference ready" simply because software checks pass.

## Historical Verification Record (Before 68-Cell Calibration)

- [x] 60 offline tests passed, including fake-provider interruption/resume,
  missing-ledger charges, receipt tampering/omission, source changes and locking.
- [x] All 224 full-roster fixture cells and 12 matched-control fixtures passed.
  These are software checks, not collected model observations.
- [x] Notebook JSON and code syntax checked; the fresh notebook section executed
  against current hashes and reported zero fresh collected networks.
- [x] Fresh analysis executed offline and returned `NOT_COLLECTED`, as expected.
- [x] Both read-only reviewers rechecked their fixes and reported no remaining
  material findings in their bounded code-review scopes. This is not a guarantee
  about every possible defect or an independent bilingual/empirical review.
- [x] Ledger inspection: 3,150 historical pilot requests, conservative total
  $2.018819775, zero unresolved attempts, no fresh-main charges. This audit added
  no paid calls. Existing NumPy/pandas deprecation warnings remain nonfatal.
