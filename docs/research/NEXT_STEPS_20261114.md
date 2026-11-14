# Next Research Steps: Evidence Before Narrative

**November 14, 2026. Status: reviewed proposal, not newly executed results.**
**Execution follow-up:** the RQ2 mixing, sampling and conditional Copenhagen
subset is now [executed and reported separately](EXTENSION_RESULTS_20261114.md).
Other proposals below remain unrun; this plan's original snapshot is retained.
Repository evidence inspected at `7411deb7b6d9eef1287fab84edf7993d92a300dc`.
Five bounded, read-only research agents contributed in two waves, with at most
three concurrent. Their agreement is not independent human peer review.

## Decision In Plain Words

We have useful findings, but not evidence that our networks accurately reproduce
human friendship in general. The next step is to test which conclusions survive
reasonable changes in measurement, sampling and baseline information, while
keeping the original four research questions central.

Keep **896 primary attempts: 890 valid graphs, including 12 empty, and six
failures**. The six selected supplemental graphs came from nine extra attempts;
they remain outside primary inference and external evaluation. The 896-graph
demo catalog is not the primary analysis population.

No new study calls, dataset downloads, experiments, ledger access or framework
installation were performed for this plan. Existing outputs and frozen source
remain unchanged. Proposed analyses below are post-collection extensions, not
retroactively preregistered tests.

## 1. Keep The Four Questions

The labels below summarize the existing questions; they do not amend the frozen
contract. See the [current brief](../PROFESSOR_RESEARCH_BRIEF.md) and
[original findings](../PRIMARY_ANALYSIS_FINDINGS.md).

| Question | What we can say now | Next useful check | What this cannot establish |
| --- | --- | --- | --- |
| RQ1: country framing | None of the 12 English-language country-framing contrasts passes the original correction on the designated endpoints. | Summarize all existing paired effects, intervals, missingness bounds and block influence, not only selected contrasts. | Nonsignificance is not equivalence or cultural invariance. More external graphs do not isolate framing effects. |
| RQ2: demographic mixing | In each person-level method, politics leads the two marginal mixing scores in all 28 model/setting combinations. Global differs: 6/28 by Coleman and 1/28 by excess share. | Show margins and seed sensitivity; prioritize within-attribute method comparisons over declaring an overall attribute winner. | Neither score identifies independent political importance. Category sizes, correlated attributes and fixed prompt order remain confounded. |
| RQ3: model consistency | The iterative Hindi contrast has the same direction in all four model-specific means. Some other contrasts reverse across models. | Publish the full model-by-contrast matrix on common complete blocks, with descriptive model-removal sensitivity. | Four fixed configurations are not a sample of all LLMs. Deleting a model changes the estimand, not just uncertainty. |
| RQ4: instruction language | Iterative Hindi-minus-English clustering is -0.04528; Holm p = 0.02383; simultaneous 95% interval [-0.08500, -0.00556]. | Retain all 24 comparisons and their sensitivity results; audit translation and instruction equivalence before broader claims. | This concerns these instructions, not Hindi speakers or national behavior. The seven settings are not a language-by-country factorial. |

**Important statistical distinction.** The intervals are Bonferroni simultaneous
t intervals; p-values use Holm correction. Only 23 of 24 contrasts were testable.
The Hindi direction is negative in all eight blocks, but significance remains
in only 3/8 leave-one-block-out fits. This is threshold sensitivity, not a 37.5%
replication probability. The exact sign-flip adjusted p = 0.1875 must be reported,
but its smallest possible two-sided raw p with eight blocks is `2/256 = 0.0078125`,
larger than the first Holm threshold `0.05/24 = 0.0020833`. No hypothesis can pass
that exact-test family with this resolution. It is not a decisive independent
disproof of the effect. Descriptive model sign reversals and block influence are
not automatically inferential evidence either. Assumptions and low precision
still matter.

## 2. Defensible Positives, Not Selected Wins

| Claim card | Evidence and denominator | Challenge and current decision |
| --- | --- | --- |
| C1: Procedure-dependent marginal mixing is reproducible within this panel. | [Attribute ranks](../../outputs/publication_figures/20261111_v4/attribute_ranks.csv): 4 methods x 28 fixed model/setting combinations; 878 nonempty primary graphs have defined mixing scores. | Supported descriptively. Shared edges and correlated roster attributes mean two agreeing scores are not independent confirmations. Do not claim human political preference. |
| C2: The Hindi iterative contrast has directional agreement across the fixed panel. | [Primary effects](../../outputs/primary_analysis/20261105/primary_contrasts.csv), [model means](../../outputs/primary_analysis/20261105/model_contrasts_descriptive.csv), [sensitivity](../../outputs/primary_analysis/20261105/sensitivity.csv); use `iterative:us:hindi`. | Supported within the stated panel, with fragile significance. No equivalence, population or translation-invariant claim. |
| C3: A topology advantage can shrink after a density-informed diagnostic. | [Efficiency records](../../outputs/publication_figures/20261111_v4/efficiency_records.csv): pooled means about 0.119 Global, 0.365 fixed WS and 0.153 target-density ER across 11 records. | Supported as a diagnostic, not causal attribution or fair-transfer victory. Target ER sees information unavailable to transfer baselines. Existing source/metric/size reversals already defeat a universal-winner headline. |
| C4: The study is auditable despite failures. | [Primary attempts](../../outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv), [failure audit](../../outputs/analysis_ready/060dcd3a86d5/failures.csv), frozen contract and manifests. | A research-quality asset, not by itself a novel scientific contribution. Six failures and 12 empty graphs stay visible. |

For C3, the post-hoc 0.06 density split is not a preregistered threshold. FB02
favors Global by 0.0056; FB05 favors ER by 0.0003. A descriptive 0.015 tolerance
does not turn these into exact ties or establish equivalence. Gap-closed ratios
can exceed 100%; prefer absolute paired distances and margins.

## 3. Lessons From The Professor's Five Papers

These are proposed applications, not claims that the papers validate our data.
The [version/access record](../PRIMARY_LITERATURE_REVIEW.md#source-access-and-versions)
identifies author-preprint fallbacks and unverified publisher differences.

| Paper | Apply to our questions | Action, not imitation |
| --- | --- | --- |
| [Chang et al., ICWSM 2025](https://arxiv.org/html/2408.16629v2) | RQ2, RQ3 | Their topology-plus-homophily comparisons are prior art. Test whether our procedure-dependent mixing pattern survives sensible scoring and resource controls; do not present political homophily alone as new. |
| [Piao et al., Nature Communications 2026; author v3](https://arxiv.org/html/2501.05171v3) | RQ2, RQ3 | Treat regulation/refinement rules as part of the system. Compare procedure contracts and resources explicitly. Do not rename final-graph mixing as opinion polarization dynamics. |
| [Papachristou and Yuan, PNAS Nexus 2025; accepted author v7](https://arxiv.org/html/2402.10659v7) | RQ1, RQ3 | Matched human/LLM choices are stronger behavioral evidence than similar topology. A future consented matched-choice study is a separate design, not something supplied by our current graphs. |
| [Helwe et al., ACL 2025](https://aclanthology.org/2025.findings-acl.883/) | RQ1, RQ4 | Keep nationality framing separate from instruction language. Independent bilingual validation and equivalent prompts would strengthen interpretation; questionnaire scores do not validate friendship ties. |
| [Argyle et al., Political Analysis 2023](https://doi.org/10.1017/pan.2023.2) | RQ1, RQ2 | Validate conditional patterns against relevant human data. Census-like marginals and detailed invented personas do not establish population fidelity. |

## 3A. Additional Papers Tagged To Specific Changes

The literature lane read abstracts and relevant methods in the linked full
texts, not every appendix. These four additions are targeted methodological
references, not an exhaustive current novelty search. Their proposed checks
are unrun and do not replace the locked primary endpoints.

| Tag and verified paper | Keep, replace or augment | Concrete application and boundary |
| --- | --- | --- |
| **METRIC-VALIDITY / external context.** O'Bray, Horn, Rieck and Borgwardt, [Evaluation Metrics for Graph Generative Models: Problems, Pitfalls, and Practical Solutions](https://arxiv.org/pdf/2106.01098), ICLR 2022, sections 5-6. | **KEEP** validated weighted W1; **AUGMENT** evaluator sanity checks. | Their MMD findings warn that metric/kernel choices can alter conclusions. Test fixed nested edge deletions on development-graph copies and check density/efficiency responses. Clustering need not change monotonically. Do not replace W1 with MMD just because a ranking improves. |
| **DIVERSITY / RQ3 context.** Thompson, Knyazev, Ghalebi, Kim and Taylor, [On Evaluation Metrics for Graph Generative Models](https://arxiv.org/pdf/2201.09871), ICLR 2022, sections 4.1-4.4. | **AUGMENT** resemblance with sensitivity to lost diversity. | Progressively duplicate a graph within existing instruction strata while preserving sample counts and stratum weights. Check whether our endpoint distributions detect the loss; some may not. This is a deliberate evaluator perturbation, not new observations. Random-GNN distances would need their own calibration and are not the default next step. |
| **SAMPLING / external context.** Leskovec and Faloutsos, [Sampling from Large Graphs](https://cs.stanford.edu/~jure/pubs/sampling-kdd06.pdf), KDD 2006, sections 3.3 and 4.2-4.5. | **REPLACE** the faithful-miniature-community interpretation; **KEEP** the declared sampling contract. | Reuse existing 25/40/50-node results, show full-record versus sampled statistics and sampling fraction n/N. Their empirical sample-size findings are not a universal threshold. A random-walk sample changes the estimand; it is not an automatic repair for unfavorable induced samples. |
| **MIXING / RQ2.** Newman, [Mixing patterns in networks](https://arxiv.org/pdf/cond-mat/0209450), Physical Review E 67, 026126 (2003), section II. | **AUGMENT** with mixing matrices and edge-end marginals; **REPLACE** attribute-importance wording. | Add categorical Newman assortativity as a labeled sensitivity to the two existing measures, using the same primary source cohort and reporting each defined denominator. Its edge-end baseline differs from the roster baseline. Preserve undefined cases and compare on common defined graphs where required. It does not remove attribute correlation or create a causal importance scale. |

Priority: sampling/context audit and RQ2 mixing matrices first; evaluator
perturbation/diversity tests next. Keep all tested metrics and null results,
including checks that expose weak evaluators. Do not select a new metric using
its favorable ranking on evaluation data. Any real modification to analysis code
needs focused tests and a separate, versioned result package before publication.

## 4. New Real-World Data: Suitability Before Rankings

External structural comparison is contextual evidence for the existing study,
especially model/procedure consistency. It does not directly answer RQ1 or RQ4,
and cannot validate RQ2 attributes absent from the observed data. A new
methodological headline outside RQ1-4 requires discussion with the professor,
not a silent change of research question.

| Candidate and official source | Role | Gate or limitation |
| --- | --- | --- |
| [Copenhagen Networks Study](https://www.nature.com/articles/s41597-019-0325-x), [official deposit](https://doi.org/10.6084/m9.figshare.7267433) | First candidate: participant-only Facebook friendship snapshot; separate from Bluetooth/calls/SMS. | One university-student cohort, not multiple populations. Descriptor documents friendship and gender files, but release license, usable bytes and missingness need verification. Joint political/religious labels are not established. |
| [Deezer Europe](https://snap.stanford.edu/data/feather-deezer-social.html) | Optional platform/sparsity stress test, not the main friendship benchmark. | Mutual followers are not friendship nominations; name-inferred gender is not demographic ground truth. At N=28,281 and M=92,752, a uniform 50-node draw has expected 0.284 edges. A mostly empty induced graph is a sampling consequence. Archive license remains to be checked. |
| [SocioPatterns primary-school contacts](https://sociopatterns.org/datasets/primary-school-temporal-network-data/) | Optional explicitly mismatched proximity control. | Children/teachers and 20-second face-to-face contacts are not adult friendship. CC BY-NC-SA terms and attribution apply. Days/classes are not independent populations. Never pool with friendship evidence. |

**Selection decision:** prioritize access and suitability checks for Copenhagen;
do not call three different edge types three replications. A second independent,
licensed friendship community remains an acquisition gap. Do not substitute a
convenient interaction dataset just to increase the dataset count. No candidate
has been downloaded or evaluated in this task.

Before comparing generated graphs with a new source, record its selection date,
official release/version, license, hash, edge definition, node inclusion,
isolates, missingness, time coverage, attribute definitions and dependent units.
Public availability is not permission to redistribute or proof of ethical fit.
For sparse sources, a metadata-based expected-edge check must precede ranking:
`E[M_50] = choose(50,2) * M / choose(N,2)`. This is not an independence assumption
about ties. Deezer's expectation implies at least roughly 71.6% empty uniform
draws by Markov's inequality. The proposed main-benchmark screening floor is
`E[M_50] >= 1`, in addition to documented friendship semantics and suitable roster
coverage: comparisons with fewer than one expected tie are not a useful primary
test of friendship structure here. This is a conservative exclusion rule, not a
universal statistical threshold or sufficient proof of informativeness. Obtain
human approval and freeze it before new comparison outcomes; retain all screened
sources and exclusion reasons in the manifest. Do not repair sparse draws after
seeing results by pruning isolates or switching to BFS neighborhoods.

## 5. Minimum Offline Evaluation Protocol

This is a proposal to finalize before execution, not a sealed protocol or a
claim that all choices are already specified.

1. **Cohorts and endpoints:** preserve 896 attempted, 890 valid, 12 empty, six failed;
   score-defined denominators remain explicit. No supplemental substitutions,
   failure-as-zero coding, or choosing conditions after their performance is seen.
2. **Estimands:** keep the existing model/condition-balanced distributional
   comparison and publish per-model/condition results. Primary question effects
   retain their original paired-block estimands. A benchmark pooled across seven
   instruction settings is not a national-population estimate.
3. **Observation contract:** retain uniform induced n=50 sampling for comparable
   sources and reuse all existing n=25/40/50 diagnostics, not a favorable subset.
   For genuinely new sources, fix those same sizes before comparison outcomes,
   applying them to both generated and observed graphs. The old size results
   remain known evidence, not prospective confirmation. Report full-network
   context and isolate/empty rates. Sampling
   spread is not a population CI; matching n alone does not match sampling frames.
4. **Transfer baselines:** preserve development-fixed ER/BA/WS. Before new target
   outcomes, select at most one closure-aware baseline and, only if identifiable
   on development data, a degree-corrected community baseline. Report fitting data,
   tuning space and failed fits. If a calibration source is needed, designate it
   separately and retain another source for testing. Do not fit to test topology
   and call the result transfer.
5. **Conditional controls:** keep target-density ER G(n,p), exact-edge G(n,m) and
   degree-preserving rewiring in a separate diagnostic tier. They get target
   information on purpose. G(n,p) matches expected density, not each draw's edge
   count. Realized density residual under G(n,m), or degree-Gini residual under
   fixed degrees, is zero by construction and cannot be evidence of fit. Specify
   one rewiring null before execution: exact labeled degrees and simple-graph
   invariants, chain mobility, independent-start agreement, Monte Carlo error and
   longer-chain stability checks. These diagnose failures, not prove uniformity.
   Show total and conditional differences together; conditioning on degrees can
   remove genuine degree-mediated variation.
6. **Metrics and variation:** retain validated metric-specific weighted W1 results.
   Show complete distributions, paired margins, all source records, sampling ranges
   and baseline simulation variation. Predefine any additional statistic and its
   purpose. No optimized composite realism score, uncalibrated edge F1 across
   unrelated node sets, or draw-level population p-values.
7. **Dependence and failures:** independent communities, not nodes, draws or model
   rows, support cross-community generalization. With few communities, describe
   cases instead of manufacturing a precise population interval. Report failures
   and undefined scores beside success-conditional structural distances.
8. **Preservation:** new versioned outputs only. The existing held-out entrypoint
   targets a sealed historical directory; do not rerun it in place for this
   extension. Record code/input hashes, environment, selectors, seeds, decisions,
   exceptions and scope. Human approval precedes a locked execution manifest.

Required before execution: exact new sources/split, inclusion criteria, any
additional baseline family and parameters (or explicitly none, with justification),
Monte Carlo budget/convergence rule, metric
definitions and reproducibility checks. A protocol hash created now would not
make these unresolved choices prespecified.

## 6. Prioritized Work, Staying Inside RQ1-4

| Order | Deliverable | New study calls? | Claim that fails if the check fails |
| --- | --- | --- | --- |
| 1 | Consolidate existing 24-contrast effect, missingness and influence evidence; full model-specific direction matrix. Reuse verified outputs before adding calculations. | No | Broad direction/robustness claims fail on sign reversals or influential blocks; nonsignificant effects remain unresolved. |
| 2 | Separate RQ2 within-attribute method contrasts from cross-attribute rank sensitivity, with all conditions and common-defined counts. Add a degree-conditioned diagnostic only after specifying and validating its null. | No | Political rank stability fails if a politics-minus-competitor margin is nonpositive. A method-direction claim instead fails on reversal of its named within-attribute contrast. Neither is causal attribute importance. |
| 3 | Dataset suitability/license manifest and a frozen external extension, starting with Copenhagen if eligible. | No | Cross-community generality is unsupported until genuinely independent, relevant communities are evaluated. |
| 4 | Evaluate all selected sources with information-tiered baselines, size sensitivity and sampling diagnostics. Publish reversals as well as favorable cases. | No | A universal structural-superiority claim fails on relevant metric/source/size reversals, already observed in current evidence. |
| 5 | If professor approves: multiple independently constructed rosters, counterbalanced field order, validated translations and compute-comparable procedures. | Requires a separate design and any new-call approval | Stronger causal/population or language-general claims remain off limits without these controls. |

For the proposed RQ2 extension, report all six method pairs (Global/Local,
Global/Sequential, Global/Iterative, Local/Sequential, Local/Iterative,
Sequential/Iterative), using second minus first on shared model/setting/seed
support for each attribute. Use excess same-group edge share as the main
descriptive scale for this extension, with Coleman and Newman as sensitivity
measures, not extra discovery opportunities. Keep the historical RQ2 outputs
unchanged. Score-defined/common-support conditioning is not all-attempt
performance; retain failure and empty counts separately. All selection here is
post-collection and must be labeled exploratory.

More seeds on the same roster alone do not repair roster bias. New model versions
alone do not make a four-model result population-general. Design power/precision
for a scientifically useful effect, not a sample size chosen to cross p < .05.

## 7. Adapt Paper Orchestra Without Automating Claims

Inspected the upstream [README](https://github.com/google-research/paper-orchestra),
[CLI](https://github.com/google-research/paper-orchestra/blob/main/paper_writing_cli.py)
and [writing pipeline](https://github.com/google-research/paper-orchestra/blob/main/methods/paper_writer.py).
They separate outline, literature, section-writing and refinement roles, with
citation/figure information and bounded reflection rounds. This is a writing
pipeline, not a guarantee of experimental validity or acceptance. Its CLI can
invoke provider models; it was not installed or run.

Adopt the organizational pattern in this repository, with no new dependencies:

| Stage | Our artifact and acceptance rule |
| --- | --- |
| Evidence first | Claim cards C1-C4 link to existing tables. New claims require exact row/column selection, input/code hashes, units, denominator and uncertainty definition. |
| Literature check | Record publication version, section/figure read, existing contribution and a precise difference. An agent-generated citation is not evidence until its primary source is checked. |
| Competing review | RQ advocate, external-validity skeptic, literature/novelty critic, statistical falsifier and reproducibility adjudicator. Each objection needs evidence or a falsification test, not theatrical disagreement. |
| Figures from evidence | Render from identified numeric tables; captions state cohort, information tier, sample unit, missingness and whether ranges are CIs. Never generate an illustrative picture and call it an empirical plot. |
| Writing last | Draft only supported claim cards. Mark proposed work separately. Unresolved contradictions narrow or block a claim; they are not settled by majority vote or polishing language. |
| Human gate | Professor and authors approve scope, ethics, interpretation and submission. AI review is assistance, not peer review or independent replication. |

### This Round's Disagreements

| Proposal | Critic's challenge | Adjudication |
| --- | --- | --- |
| Make density sensitivity the universal realism headline. | Source, metric and size reversals already contradict it; it could replace the original RQs without approval. | Keep as exploratory evaluation context. Discuss a methodological headline with the professor rather than silently add RQ5. |
| Treat the sign-flip adjusted p as disproving Hindi. | The eight-block exact test cannot reach the first 24-family Holm threshold. | Disclose its resolution, the original result and LOBO sensitivity together. No certainty in either direction. |
| Add Deezer as another friendship validation. | Different tie semantics, inferred gender and mostly empty 50-node draws. | Not main friendship evidence; optional labeled sparsity stress only. |
| Politics leads under two scores, so it is independently important. | Both reuse the same graph and confounded fixed roster; category counts differ. | Keep marginal ordering; prioritize within-attribute contrasts and separate total from conditioned quantities. |

## 8. ICML Target: Decision Gates, Not A Promise

The current [ICML 2027 contribution guidance](https://icml.cc/Conferences/2027/ContributionTypes)
allows evaluation contributions without a new learning algorithm. Our possible
fit is an evaluation contribution, **if** it reveals a generalizable limitation
or a useful validated protocol beyond adding datasets to existing comparisons.
This is a candidate positioning, not established novelty or readiness.

Before an ICML decision, require a precise contribution relative to the five
papers and newer work, a claim tied to RQ1-4 or explicitly approved scope,
information-parity controls, relevant external evidence, honest dependence and
failure handling, reproducible artifacts, and a human expert's critique.
More agents, attractive plots and a polished demo do not substitute for those.

On November 14 the official [dates page](https://icml.cc/Conferences/2027/Dates)
lists January 14, 2027 for abstracts and January 22 for full papers, AoE.
The homepage still carries forthcoming-call language and the call mentions
submission-process updates. Recheck official instructions before planning a
submission; this note does not guarantee dates or acceptance.

The current [author AI policy](https://icml.cc/Conferences/2027/UseOfGenerativeAIByAuthors)
requires disclosure of substantive AI research/writing assistance and human
responsibility. Our synthetic generation, analysis/code assistance, literature
synthesis and figure work need accurate disclosure, not a grammar-only statement.

## 9. Evidence Snapshot And Review Boundary

These table bytes were hashed in this task. Hashes bind this plan's references,
not an independent replay of graph generation or a new raw-data audit.

| Repository-relative file | SHA-256 |
| --- | --- |
| `outputs/primary_analysis/20261105/primary_contrasts.csv` | `b6418b813384ab471273e764188407771eb79a3a071611b5a322b83a4e84324e` |
| `outputs/primary_analysis/20261105/sensitivity.csv` | `cd5a8a45b1073a2235acb0aefaa52a8622da08bf260b35d5d8c58531586c5196` |
| `outputs/primary_analysis/20261105/model_contrasts_descriptive.csv` | `d4ba8e1798286ab61419730fef723de282ddbdebb2611667286b4b4b217f59fe` |
| `outputs/publication_figures/20261111_v4/attribute_ranks.csv` | `89508f1e3b6ebcac7ff1896efdfefa8344448d002064be06531e7d13dc1e8edd` |
| `outputs/publication_figures/20261111_v4/efficiency_records.csv` | `e35d94a0d4a3ac566f610364baacc3b68a1d2d4145b29140c1ce6d4d21a968fd` |

No new performance numbers, human-ground-truth validation, publisher/preprint
equivalence, dataset licenses beyond those explicitly checked, or conference
acceptance probabilities are claimed.
