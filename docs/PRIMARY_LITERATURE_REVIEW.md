# Primary Literature Review: The Professor's Five Papers

Reviewed November 5, 2026. **Exact five-paper set: CONFIRMED by the user.**
This supersedes the earlier unconfirmed, two-reference locating note. Park is
not one of these five. Only clean primary URLs are retained; no email or tracking
links. Paper summaries are paraphrases, bounded to approximately 150-200 words
per paper including associated source facts. Recommendations are our analysis,
not the papers' findings or changes to the collection contract.

## Source Access And Versions

| ID | User-supplied paper | Full-text access verified in this review |
| --- | --- | --- |
| P1 | Chang et al., 2025, [LLMs Generate Structurally Realistic Social Networks but Overestimate Political Homophily](https://ojs.aaai.org/index.php/ICWSM/article/view/35820) | Publisher bot-blocked; full author text [arXiv:2408.16629v2](https://arxiv.org/html/2408.16629v2), March 27, 2025. Methods and C3 below checked there; publisher/preprint byte equivalence not asserted. |
| P2 | Piao et al., 2026, [Emergence of polarization in networks of large language model agents](https://www.nature.com/articles/s41467-026-78228-y) | Publisher abstract readable; accepted-manuscript PDF endpoint was intermittently accessible. Full methods/figures checked in [author preprint v3](https://arxiv.org/html/2501.05171v3), September 7, 2026. Do not silently equate this with the October 8 publisher version. |
| P3 | Papachristou and Yuan, 2025, [Network formation and dynamics among multi-LLMs](https://doi.org/10.1093/pnasnexus/pgaf317) | Publisher full-text/PDF retrieval blocked. Full [author accepted preprint v7](https://arxiv.org/html/2402.10659v7), October 5, 2025, explicitly identifies PNAS Nexus acceptance. Publisher-indexed text corroborates the network-principle experiments; unverified publisher revisions remain possible. |
| P4 | Helwe, Balalau and Ceolin, 2025, [Navigating the Political Compass: Evaluating Multilingual LLMs across Languages and Nationalities](https://aclanthology.org/2025.findings-acl.883/) | Full [ACL proceedings PDF](https://aclanthology.org/2025.findings-acl.883.pdf) read, including methods, Table 2, appendix plots and translation checks. |
| P5 | Argyle et al., 2023, [Out of One, Many: Using Language Models to Simulate Human Samples](https://doi.org/10.1017/pan.2023.2) | Full publisher HTML read: silicon sampling, studies 1-3, figures and discussion. Supplementary analyses not exhaustively audited. |

None is represented as abstract-only: when publisher access failed, the
identified author full text supplied the methods. Version caveats above matter.

## Paper Review Matrix

Figure references below identify evidence to inspect, not figures reproduced in
this repository. Their descriptions were checked against full-text captions.

| Paper | RQ / contribution | Design and baselines | Metrics, figures and convincing evidence | Evidence boundary |
| --- | --- | --- | --- | --- |
| P1 ([sections 3-5, B, C](https://arxiv.org/html/2408.16629v2)) | Can generated friendship graphs reproduce topology without demographic bias? | Fixed 50-person US-demographic roster; 30 graphs/method; six LLMs; three methods: Global, Local, Sequential. Baselines: eight empirical graphs and ER/BA/WS generators. | Density, clustering, LCC fraction, scaled paths, modularity, normalized degrees and observed/expected cross-group ties; standardized mean and KS distances. Figures 3-5 separate topology from mixing; Table B1 quantifies agreement. | Structural agreement coexists with excessive political homophily. Iterative is not a fourth source method. All topology comparisons are undirected. |
| P2 ([author full text](https://arxiv.org/html/2501.05171v3)) | Can interacting agents co-evolve opinions and networks, and how do interventions change polarization? | Main illustrations use 1,000 agents/issue, near-Gaussian initial opinions and a Watts-Strogatz initial network. Self-expression, communication and opinion updating are LLM-driven. Comparisons include unregulated/self-regulated agents and network/individual interventions; robustness spans models and initialization. | Opinion trajectories/distributions, homophilic interaction proportions and polarization levels. Figure 1 shows the system; Figure 2 diagnoses inconsistency and self-regulation; Figures 3-4 connect interaction patterns to intervention outcomes. | Convincing within-system mechanism contrasts, not a held-out human friendship-edge prediction benchmark. Prompts and self-regulation are part of the intervention specification; apparent emergence must be interpreted conditional on them. Timepoints/interactions are not automatically independent system replications. |
| P3 ([accepted author text](https://arxiv.org/html/2402.10659v7)) | Which micro-level link-formation principles and macro-level structures do LLM agents reproduce? | Synthetic principle experiments; empirical friendship, telecommunication and company settings; model, temperature and prompt variants. Preferential-attachment null: random selection. Human comparison uses identical candidate inputs, with 100 social-context and 103 company-context survey responses. | Degree distributions, clustering/small-world behavior and discrete-choice effects for degree, similarity and common neighbors. Figure 1 compares attachment with its null. Figure 6 evaluates human alignment using total-variation distance, Spearman correlations, effect estimates and Borda summaries. | Direct human choices strengthen the evidence beyond attractive topology pictures. They validate constrained tasks, not unrestricted national friendship networks. The authors identify greater within-model consistency than human variability. |
| P4 ([proceedings PDF](https://aclanthology.org/2025.findings-acl.883.pdf)) | Do language and assigned nationality alter expressed political ideology? | Fifteen multilingual LLMs; 50 populous countries and associated languages; nationality-only, language-only and combined scenarios. These are model prompts, not respondents sampled in 50 countries. Back-translation screens language support. | Two Political Compass axes; Table 2 reports centroid distances, dispersion and quadrants; paired Wilcoxon comparisons contrast scenarios. Appendix E Figure 1 visualizes shifts. Appendix B documents repeated responses, majority voting and failure/tie rules. | Language influences expressed ideology more strongly in their comparisons. Smaller models can be more stable. Stability is not correctness; Political Compass output is not friendship homophily. Automated translation screening does not establish cross-cultural construct equivalence. |
| P5 ([publisher full text](https://doi.org/10.1017/pan.2023.2)) | Can demographically conditioned GPT-3 samples reproduce subgroup patterns, termed algorithmic fidelity? | Silicon sampling conditions on real human profiles, including ANES; studies cover partisan descriptors, vote prediction and closed-ended associations. Human texts, judgments and survey responses supply external comparators. | Figure 2 compares descriptor frequencies; Figure 3 examines human evaluations; Table 1 compares vote predictions; Figure 4 compares Cramer's V associations. Study 3 conditions on 11 human responses to predict a twelfth. | Multiple human-data comparisons are more convincing than marginal resemblance alone. Claims concern US public opinion and aggregate correspondence, not individual identity reconstruction or network ties. Conditioning helps only if model conditional distributions are adequate; simulated samples do not independently certify fidelity. |

## Verified Chang Friendship Reference Roster

Names/counts transcribed from [Table C3 and Appendix C.3](https://arxiv.org/html/2408.16629v2).
Counts describe that paper's processed graphs, not a verification of our local
files or a guarantee that raw datasets already use identical transformations.

| Reference | Nodes | Edges |
| --- | ---: | ---: |
| Galesburg | 31 | 63 |
| Hi-tech | 36 | 91 |
| Karate | 34 | 78 |
| Prison | 67 | 142 |
| Tailor 1 | 39 | 158 |
| Tailor 2 | 39 | 223 |
| Moreno freshmen | 31 | 218 |
| Moreno high school | 70 | 274 |

First six: CASOS; final two: KONECT. Tailor retains sociational, not instrumental,
ties; Moreno retains strictly positive weights. Hi-tech requires mutual
nominations. Tailor observations are seven months apart, not independent sites.

That Hi-tech description is the paper's stated tie rule, not a newly applied
filter in our evaluation. Our archived XML with undirected OR projection
reproduces 36 nodes/91 edges; upstream reciprocity processing is not independently
verified. See the explicit caveat and selectors in
[empirical evaluation](../outputs/primary_analysis/20261105/empirical_evaluation.json)
and [completed findings](PRIMARY_ANALYSIS_FINDINGS.md#evaluation-beyond-p-values).

**Our implementation recommendation:** use this as a small heterogeneous
friendship-reference panel, separately labeled exploratory. Verify exact local
file provenance, graph counts, isolates, edge projection, self-loop removal,
weight filtering and source-specific nomination rules before comparing metrics.
Keep transformations explicit; do not force node/edge counts to match by deleting
unexplained records. The parent owns benchmark code and actual audit outcomes.

## How Our Study Differs And What To Do

The current local design is documented in `README.md`,
`docs/METHOD_COMPUTE_FAIRNESS.md` and `POPULATION_REVISION.md`: one roster of 50
designed fictional adults; seven instruction settings; four models and four
procedures; eight planned schedule blocks; 896 planned attempts. This note does
not independently attest final completion or statistical results.

| Reference lens | What ours can contribute | Actionable gap, without overstating scope |
| --- | --- | --- |
| P1: network evaluation | Controlled country-framing/instruction contrasts and an additional refinement procedure. | Add empirical metric distributions and matched classical controls; verify projection/definitions. Compare uncertainty and variability, not just means or one chosen exemplar. |
| P2: evolving interaction systems | Auditable tie-generation procedures, rather than autonomous conversation-driven opinion dynamics. | Do not rename final-graph homophily as polarization dynamics. Show saved formation stages only when real recorded decisions support them; distinguish replay from a behavioral time-series experiment. |
| P3: human link-choice grounding | A reproducible procedure-level model-behavior comparison. | Empirical topology benchmarks do not replace matched human choices. A future consented human task could test identical choices; no new recruitment, paid calls or approval is implied here. |
| P4: language versus nationality | Within-method contrasts can disentangle specified framing from specified instructions. | Keep the seven-setting design explicit: four countries in English plus US in three other languages, not a factorial. Persona JSON stays English; claim named-contract sensitivity, not universal language effects. |
| P5: population and subgroup fidelity | Controlled synthetic-roster sensitivity, independent of claims about a real sampling frame. | Multiple independently grounded rosters and external outcomes are needed for population generality. Invented biography detail is not human evidence; census marginals alone do not validate joint attributes or ties. |

### Real-World Evaluation Priority

The user's priority is empirical evaluation, not only additional simulation.
The bounded comparison is complete as a **post-hoc structural diagnostic** using
existing real-network assets. All eight published node/edge anchors were
verified. See [findings](PRIMARY_ANALYSIS_FINDINGS.md#evaluation-beyond-p-values)
and [scope/provenance caveats](../outputs/primary_analysis/20261105/empirical_evaluation.json).
It reports density, clustering, global efficiency and descriptive ECDF distances,
not KS p-values or cultural ground truth. Actual synthetic controls are
node/edge-matched **G(n,m)** draws, not an implemented ER/BA/WS comparison suite.

1. **Keep empirical references separate.** Preserve the frozen confirmatory
   analysis. Clearly label the new topology benchmark post hoc/exploratory,
   including what was selected after results were available.
2. **Use a common measurement contract.** Declare simple undirected friendship
   projection for comparable topology, while retaining directed nomination
   results separately. State isolate, disconnected-path, empty-graph and
   modularity conventions. Report undefined quantities as undefined.
3. **Show distributions against controls.** Per-metric empirical points/ranges
   and generated distributions are more informative than an attractive graph
   picture. The completed analysis uses G(n,m) references; a broader matched
   ER/BA/WS comparison remains future work. Distances describe agreement; they do not
   by themselves prove interchangeability or adequate power.
4. **Expose size/density confounding.** Record reference sizes and densities;
   assess degree/path scaling and quota-induced restrictions. Do not infer
   improved social reasoning from a statistic largely fixed by edge constraints.
5. **Preserve provenance and dependence.** Record input hashes, source datasets,
   transformations and reference membership. Two waves from the same site are
   dependent evidence, not two independently sampled societies. With only eight
   heterogeneous references, avoid universal population inference.
6. **Separate topology from cultural truth.** None of this small reference panel
   constitutes nationally representative US/India/Japan/Brazil friendship data.
   Country framing on the same fictional roster is a model intervention, not
   country-grounded human behavior. Human bilingual validation remains absent.
7. **Keep resource/failure denominators visible.** Attempts, valid graphs, empty
   graphs, terminal failures and endpoint-defined observations need distinct
   denominators. Iterative has more decisions/refinement than other methods;
   equal repetitions do not establish compute-matched superiority.
8. **Avoid metric substitution.** Our categorical Coleman scores, numeric age
   assortativity and any cross-group ratios require separately stated formulas
   and baselines. Mixing-strength rankings are not causal attribute importance.

### Figure Package Recommendation

The completed empirical context plot is documented in the
[findings and executed notebooks](PRIMARY_ANALYSIS_FINDINGS.md#reproduction-and-reading-order).
The package below remains a recommendation for expanded presentation, not an
inventory of delivered figures. In particular, ER/BA/WS comparisons are future
work; the implemented synthetic references are G(n,m).

- Empirical-reference strip plots with every reference labeled, overlaid with
  generated and matched-control distributions; facet by metric and method.
- Paired within-method instruction contrasts with uncertainty and usable-block
  counts, kept separate from empirical realism distances.
- A compact topology-distance profile with size/density and transformation
  caveats; no single composite winner unless weighting was justified in advance.
- Failure/resource panel beside topology: attempted/valid/empty/failed counts,
  realized cost and decisions. Do not present successful graphs alone.

## Venue Fit: Conditional Assessment

This is a scope judgment, not an acceptance prediction, deadline check or a
recommendation to submit to an already concluded edition. Official pages below
are used as scope examples; verify the actual target edition before submission.

| Venue direction | Official scope evidence | Our readiness judgment / gap |
| --- | --- | --- |
| ICWSM | [Official submission guidance](https://icwsm.org/2026/submit.html) describes original web/social-media research and full, poster, demo and dataset tracks. | Closest topical direction if empirical network evaluation supports a substantive social-computing insight. More conditions alone are not novelty; defend differences from P1 and P3. A demo cannot substitute for scientific validation. |
| ACL / Findings | [Official ACL main call](https://2026.aclweb.org/calls/main_conference_papers/) includes agents, computational social science, multilinguality, bias and evaluation. | Plausible if the contribution centers rigorous instruction-language measurement and reusable NLP evaluation. Translation/construct equivalence, prompt sensitivity and language coverage need stronger treatment; topology-only benchmarking is insufficient for that claim. |
| PNAS Nexus | [Official aims](https://academic.oup.com/pnasnexus/pages/about) emphasize broad interdisciplinary advances. | A possible longer-term direction only with a clear cross-disciplinary finding and stronger external grounding. Current panel benchmarking alone does not establish a general theory or human mechanism. |

Nature Communications and Political Analysis appear in the supplied reference
set, but publication there does not transfer venue readiness to this project.
No separate recommendation for either is made here. First establish what the
empirical comparison actually supports, then choose the venue and claim scope.

## Review Boundary

Only this literature file was edited. No benchmark code, frozen protocol,
historical outputs, paid model calls, SSH sessions, SQLite access or subagents.
No full-paper copies or long quotations were added. This is a targeted methods
and evidence review, not a line-by-line replication or exhaustive supplementary
audit; publisher-version differences for P1-P3 remain explicitly unresolved.
