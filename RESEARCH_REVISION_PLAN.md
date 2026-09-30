# Reviewed Plan: Repair the Research and Build an Interactive Explorer

> Implementation update: see [REVISION_STATUS.md](REVISION_STATUS.md) for the
> current feedback-by-feedback checklist. Offline metric/parser repairs,
> historical reanalysis, the notebook revision and a static Three.js explorer
> are implemented. The stricter audit found 16 invalid historical graphs with
> 23 self-links, plus an age-inappropriate political roster. Original artifacts
> are preserved. Old paid dispatch is locked; a new capped engineering pilot
> completed 24 networks (20 original plus four Sol checks). Source-grounded populations, reviewed translations
> and confirmatory experiments remain incomplete.

## Current Approved Budget Scope

- [x] **Spending control:** $5 pilot ceiling inside $50 cumulative. Failed replies
  are paid attempts, not free retries; uncertain requests retain reservations.
- [x] **Model access:** GPT-6 Luna and GPT-4.1 Mini were checked with this account.
  This replaces the earlier Luna/Sol pricing scenario below.
- [x] **Pilot design:** 20 full-50-person networks across four methods. Both models
  receive US-English; Luna also receives US-Hindi, US-Japanese and US-Portuguese.
- [x] **Requested Sol extension:** four full-50 US-English method checks, using
  the original shared $5 pilot budget; no change to the two-model main scope.
- [x] **Pilot completion:** 24/24 verified, $1.886188 conservative cost, no
  unresolved requests. Use `paid_study.py --verify` for the receipt. Debugging
  prompt fixes remain distinct engineering variants.
- [x] **Analysis selection:** multiple model/method/country/language/seed values
  drive real saved-result plots, exports and explicit missing-combination counts.
- [ ] **Astra checkpoint:** review the actual pilot and protocol before main runs.
  Five defensible adult rosters, translation review and sample-size justification
  remain required. No conference has a universal "enough graphs" threshold.
- [ ] **Reduced main target:** 640 networks = 8 conditions x 2 models x 4 methods
  x 5 rosters x 2 repetitions. Conditions are US/India/Japan/Brazil/neutral in
  English, plus US in Hindi/Japanese/Portuguese. Estimated pilot-plus-main cost is
  about $36 with contingency; actual usage and the hard cap determine feasibility.
- [ ] **Outside this $50 scope:** authentic country-grounded populations,
  full country-language interactions, cross-provider models, large robustness
  panels and validated mitigation. These remain research needs, not deliverables
  that can be marked complete merely by generating more graphs.

The full design below is the feedback-driven roadmap. Work beyond this reduced
scope is deferred, not silently claimed. The live implementation checklist is
in `REVISION_STATUS.md`; the roadmap gates below describe the broader research.

A read-only subagent reviewed the plan twice. Its findings are incorporated below, particularly around experimental controls, replication, costs, failures, and benchmark consistency. This is a reviewed implementation plan, not a completed validation or a guarantee of acceptance.

**Checklist:** `[x] ✅` means something exists or an inspection was completed. `[ ] ❌` means a confirmed problem still needs fixing. `[ ]` means proposed work remains unfinished.

## 1. What We Already Have

- [x] ✅ **An inspected repository and feedback document.** We checked [your GitHub repository](https://github.com/hemu77/UA_Capstone), local code, and the supplied review-action summary. Where the summary and code disagree, we will establish which implementation produced the submitted results rather than assume either description is correct.
- [x] ✅ **Four generation methods are implemented.** Global asks for the whole network; local asks each persona to select connections; sequential exposes information about the developing network; iterative starts with a network and asks personas to revise connections. These generate networks using pretrained models; they do not train new models.
- [x] ✅ **Historical experiment artifacts exist.** Study verification tables contain 24 cultural, 72 additional-method, and 96 language runs. These are useful existing assets, but their old “pass” labels do not prove that every measurement or conclusion is valid.
- [x] ✅ **Analysis scripts, a notebook, tables, and PNGs already exist.** We will repair and reuse them rather than replace the entire project. The interactive viewer will supplement these outputs, not become a separate source of statistical truth.
- [x] ✅ **Specific failures were reproduced.** The edge-distance calculation incorrectly scores completely different undirected graphs as `0.5` rather than `1`, and can treat an edge written backwards as a different edge.
- [ ] **Locate the GPT-5.1 evidence.** Identifiable GPT-5.1 results were not found in the three checked project folders. They remain outside the evidence set until their files, configurations, and provenance can be verified.

## 2. What We Must Fix Before Trusting Results

- [ ] ❌ **Define exactly what a connection means.** The current capstone generator creates an undirected graph: if either persona selects a connection, the friendship edge exists. Preserve that definition for the primary study. New logs will retain who selected whom, but we will not invent historical direction or reciprocity from undirected files.
- [ ] ❌ **Repair measurements with small, hand-checkable examples.** Correct edge orientation and normalization; distinguish ratio homophily from Coleman homophily; measure age using a numeric similarity statistic. Homophily means “similar people connect more often,” but different formulas measure different things. Tests must cover empty graphs, isolates, disconnected graphs, and demographic groups with insufficient variation.
- [ ] ❌ **Separate instruction language from the fictional world.** Currently, changing prompt language also changes what language the participants supposedly speak. That changes two things simultaneously. The revised experiment will change only the instructions and their translated presentation while preserving the social scenario and persona meanings.
- [ ] ❌ **Make generation failures visible.** Reject self-links, duplicate choices, invalid candidates, and impossible requests. Allow at most three attempts per logical request; retry transient service failures and invalid output using a fixed policy, but stop immediately for permanent authentication/configuration errors. Exhausted attempts produce a failed run, not endless regeneration until something looks acceptable.
- [ ] ❌ **Stop mixing incompatible experiments.** Preserve historical outputs unchanged and write corrected analysis separately. Old runs can join a new comparison only when prompts, model versions, roster information, method settings, and retry policies are demonstrably equivalent. Otherwise, label them historical descriptive evidence.
- [ ] ❌ **Rebuild benchmark accounting and comparisons.** Reconcile the feedback’s differing dataset counts with the eight networks in the saved benchmark table. Freeze inclusion rules and document direction, weights, duplicate edges, self-loops, isolates, disconnected paths, and size handling. Compare every prespecified metric against density-matched random, degree-preserving, and demographic-similarity baselines.
- [ ] ❌ **Replace shallow verification with reproducible checks.** Resolve dependency conflicts, execute the notebook without API access, explicitly skip unavailable legacy experiments, and regenerate figures from corrected tables. A PNG looking plausible is not enough: its underlying graph, labels, legend, and reported numbers must agree.

**Green condition:** the software passes the stated tests, and every reported number can be traced back to a graph and configuration. This establishes computational reliability, not yet real-world social validity.

## 3. The Revised Research Design

**Culture needs a more careful definition.** Keep the US, India, Japan, and Brazil, but distinguish a model’s reaction to a country label from actual human cultural behavior.

- [ ] **Run a controlled framing study.** Present identical personas under parallel country-context descriptions, plus a no-country-label control. Do not tell the model that one country emphasizes religion or family and then present the resulting religious clustering as an unexpected discovery.
- [ ] **Run a separate population-grounding study.** Construct country-grounded synthetic rosters using documented demographic distributions and available joint relationships. These test whether results survive population changes; they do not isolate “culture alone.” Sources such as WVS require reporting survey dates, sampling limitations, and variable comparability. [WVS documentation](https://worldvaluessurvey.org/AJDocumentation.jsp?COUNTRY=&CndWAVE=7)
- [ ] **Use culturally defensible attributes.** Do not treat US party labels or racial categories as universally equivalent. Document each field shown to the model, maintain IDs-only primary prompts, and rank only attributes actually supplied. A filename containing “interests” does not mean interests were included in the experiment.
- [ ] **Cross countries and languages instead of pairing each country with one language.** The new core languages are English, Hindi, Japanese, and Brazilian Portuguese; Portuguese support must be implemented. For example, India will be tested with both English and Japanese instructions. Otherwise, language and country effects cannot be separated. Spanish results remain supplementary, and Hindi is not presented as representing every Indian language.
- [ ] **Validate translations.** Record translation versions, back-translation checks, and bilingual review. Keep identifiers, numerical attributes, scenario meaning, and retry instructions consistent. If human language review is unavailable, translation-dependent findings remain provisional.

| Research Question | Plain-English Meaning | Evidence Required |
|---|---|---|
| RQ1: Context | Does changing the country framing change the generated network? | Matched personas and instruction language, with country and neutral controls. |
| RQ2: Attributes | Which supplied attributes are associated with connections? | Comparable homophily measurements, uncertainty, and controlled attribute-masking tests. |
| RQ3: Models | Do the tested OpenAI models behave similarly? | Identical experimental settings and matched roster/order blocks. |
| RQ4: Language | Does translating the instructions change the result? | The same social scenario and personas across reviewed translations. |

- [ ] **Keep method comparisons honest.** Friendship budgets and candidate opportunities affect network density. Analyze the RQs within each method first; compare methods within genuinely comparable budget regimes. Where budgets cannot be matched, describe differences as whole-protocol differences, not proof that the algorithm alone caused them.
- [ ] **Separate repeated generations from different populations.** A roster is one group of 50 synthetic people. Generating four networks from that roster measures generation variability; using five independently constructed rosters also measures population variability. These are different sources of uncertainty and must not be treated as interchangeable.
- [ ] **Prespecify statistical decisions.** Before confirmatory generation, freeze primary metrics, meaningful effect-size targets, comparisons, and a pilot-informed sample-size calculation. Use paired comparisons, roster/run-aware uncertainty intervals, and multiple-testing correction within the four RQ families. Failed runs remain in the failure-rate denominator; missing measurements are not replaced with zero.
- [ ] **Make robustness checks explicit rather than promising them vaguely.** Price a predefined panel covering two equivalent prompt paraphrases; temperatures `0.2`, `0.8`, and `1.0` where supported; local candidate counts `6`, `12`, and all `49`, with random-neighborhood controls; iterative initialization from local versus sequential with `0`, `3`, and `6` rounds; and network sizes `50`, `100`, and `150`. Each panel has a declared scope and remains open if not run.
- [ ] **Distinguish explanation from intervention.** Masking an attribute tests whether exposing it changes model behavior. A separate, paired prompt intervention will discourage unsupported demographic assumptions, using identical held-out rosters and orders for intervention and control. Report whether it reduces demographic concentration and what happens to topology; reduced homophily is not automatically greater realism.

**Models and spending:** use `gpt-6-luna` and `gpt-6-sol` as the proposed newer-model panel, subject to account availability and compatible settings. This remains an **OpenAI-only study**; the cross-provider reviewer concern stays unresolved, and universal LLM claims must be removed. [OpenAI model catalog](https://developers.openai.com/api/docs/models)

**No paid runs before your approval.** First quote a 20-network technical pilot: eight US-English model/method combinations, plus twelve non-English combinations using the cheaper model, all with 50 personas. Exclude this pilot from confirmatory inference. The earlier full proposal of 2,560 networks could require up to **288,640 API requests before retries**, excluding additional controls and robustness studies. It is a pricing scenario, not an automatic launch or a justified sample size. If the adequate study exceeds the budget, narrow the claims and scope explicitly.

## 4. What the Three.js Extension Will Do

**The experience:** choose an experiment, explore its actual network, inspect a persona’s connections, and compare conditions without paying for another API call.

- [ ] **Explore actual saved networks.** Filter by model, method, country context, language, roster, and repetition. Rotate, zoom, select a persona, highlight neighbors, and color nodes by a selected attribute.
- [ ] **Compare without misleading movement.** Show two networks side by side with shared coordinates for the same personas. Highlight shared, added, and removed edges. Disable persona-level edge comparison across different rosters; those comparisons use summary statistics instead.
- [ ] **Explain the numbers beside the graph.** Display definitions, uncertainty where available, source run IDs, and clear labels for historical, validated, incomplete, or sandbox data. The website reads Python-produced metrics rather than implementing competing formulas.
- [ ] **Replay genuine generation history.** New runs record actual decisions: global appears as one graph-producing event; local/sequential show recorded choices; iterative shows additions and removals. Old final-only graphs remain final-only. Animation must not fabricate a history.
- [ ] **Provide a clearly separated playground.** Users can temporarily add or remove edges and inspect the consequences. Label this “manual simulation,” not an LLM result, and never overwrite research files. This interaction is not model training or live paid generation.
- [ ] **Make it accessible and shareable.** Include keyboard-accessible tables, readable legends, mobile support, reduced motion, downloads, and shareable views. Deploy a lightweight static site to GitHub Pages with no API keys in the browser.

**Green condition:** displayed nodes and edges exactly match exported data, replay reconstructs the final graph, and interaction/accessibility checks pass. Viewer readiness and research readiness remain separate.

## 5. Delivery Order and Reviewer Gates

- [ ] **Stage A: Evidence inventory.** Create a checklist mapping every review-document issue to a fix, experiment, justified exclusion, or unresolved limitation. Preserve original results and establish which code produced them.
- [ ] **Stage B: Free computational repairs.** Fix shared generation/analysis code, dependency installation, verification, notebook execution, and historical reanalysis. Add readable comments explaining important decisions rather than commenting every obvious line.
- [ ] **Stage C: Validated exports and explorer.** Build the viewer from corrected historical exports. It can proceed while spending approval is pending, but unvalidated scientific claims must not appear as established findings.
- [ ] **Stage D: Estimate, approved pilot, then confirmatory approval.** Extend the existing runner with dry-run, resume, and cost-cap controls. Record immutable request/response events, parameters, actual usage, failures, model identifiers, and available fingerprints. Reserve the possible cost of in-flight calls before dispatch so the spending limit is enforceable.
- [ ] **Stage E: Approved research and independent review.** Run only the agreed matrix and robustness panels. Keep a read-only reviewer role at the protocol, code/analysis, and final-report gates. Reuse the same review lane where practical; the parent integrates findings and verifies fixes rather than treating a subagent’s approval as proof.
- [ ] **Stage F: Documentation and publication.** Update the README, architecture report, and notebook with beginner-readable explanations, commands, all four RQ answers, limitations, and links from claims to evidence. Inspect figures for readability, scan publishable files for secrets, replace the previously exposed key, and publish through the clean repository only after the relevant gates pass.

**Final status will distinguish three things:** software checks passed, experiments completed, and research claims supported. An attractive viewer or a successful script cannot turn an unresolved scientific question green, and no review process can honestly guarantee zero undiscovered errors or conference acceptance.
