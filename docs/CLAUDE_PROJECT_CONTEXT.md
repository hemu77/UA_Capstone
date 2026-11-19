# Claude Profile: Cross-Cultural and Linguistic LLM Network Study

Updated November 16, 2026, America/Phoenix.

## Project And Links

I am developing **Cross-Cultural and Linguistic LLM Network Study**, a research
and engineering capstone about how language models construct friendship networks
among fictional people. Preserve this established project name. Do not rename
the project "Network Observatory" merely because that phrase remains in the
hosting address or older interface branding.

- Repository: <https://github.com/hemu77/UA_Capstone>
- Deployed Three.js workspace: <https://ua-network-observatory.handm01042024.chatgpt.site/layers.html>
- All valid saved networks: <https://ua-network-observatory.handm01042024.chatgpt.site/layers.html#saved-plots>
- Professor brief: <https://github.com/hemu77/UA_Capstone/blob/main/docs/PROFESSOR_RESEARCH_BRIEF.md>
- Analysis guide: <https://github.com/hemu77/UA_Capstone/blob/main/ANALYSIS_START_HERE.md>
- Notebooks: <https://github.com/hemu77/UA_Capstone/tree/main/analysis>
- Current figure release: <https://github.com/hemu77/UA_Capstone/tree/main/outputs/publication_figures/20261118_v6_2>
- Graph-file index: <https://github.com/hemu77/UA_Capstone/blob/main/NETWORKS.md>
- Deployment documentation: <https://github.com/hemu77/UA_Capstone/blob/main/docs/SIMULATOR_DEPLOYMENT.md>

The work extends Stanford SNAP's LLM social-network code. It combines controlled
experiments, statistical analysis, public-network comparisons and an interactive
evidence explorer. It is not a validated simulation of real people.

## Completed Scope

### Research Questions

| Question | Controlled scope | Interpretation boundary |
| --- | --- | --- |
| RQ1: country framing | Compare US, India, Japan and Brazil framing with English instructions. | Prompt framing, not representative national populations. |
| RQ2: demographic mixing | Compare political-orientation, religion and gender mixing across methods and settings. | Marginal graph patterns, not independent demographic causes. |
| RQ3: model consistency | Compare four fixed model configurations under matched settings. | Not a representative sample of all language models. |
| RQ4: instruction language | Compare English, Hindi, Japanese and Brazilian Portuguese with US framing fixed. | These instructions, not language speakers or cultural traits. |

The design uses one fixed roster of **50 fictional adults**. Models are recorded
as `gpt-4.1`, `gpt-5.6-luna`, `gpt-6-luna` and `gpt-6-sol`; treat these as saved
configuration identifiers, not claims about currently available API products.
Generation methods are **Global, Local, Sequential and Iterative**.

There are seven settings: four country framings in English, plus US framing in
the other three instruction languages. Eight schedule blocks give
**4 models x 4 methods x 7 settings x 8 blocks = 896 primary attempts**. This is
not a full country-by-language factorial. Persona attributes remain in English;
this does not test fully translated personas.

### Collection And Cohort Separation

| Collection | Attempts | Valid graphs | Failed attempts | Analysis use |
| --- | ---: | ---: | ---: | --- |
| Original primary study | 896 | 890, including 12 empty | 6 | Primary inference, evaluation and recorded replay |
| Separate supplements | 9 | 6 selected successes, including 1 empty | 3 | Separately labeled final-graph display only |

**The 896 valid networks in the browser are 890 primary plus 6 supplements, not
896 primary successes.** All failures remain recorded. A valid empty graph
retains its 50 nodes and is not a failed attempt. Supplements never replace
missing observations in statistics, comparisons or primary replay.

Main collection ran on UCLA Hoffman2 using Slurm. Saved responses, graph updates,
hashes, failure records and checkpoint evidence support traceability. The
historical conservative total, including supplemental attempts and retained
reservations, was approximately **$102.635 against a $120 cumulative cap**.
This is not a provider invoice or authorization to spend the remaining amount.

### Statistical Analysis And Public-Network Evaluation

Six analysis notebooks (01-06), a separate readiness notebook (00), and **19
research figures** cover primary inference, comparison plots, failures,
resources, demographic mixing, sampling and external structural evaluation.
Current figures are release **v6.2**, with PNG, SVG, PDF and plotted CSV files.
Earlier sealed releases remain preserved; a layout-only successor is unfinished.

The locked primary family contains **24 contrasts**. Four-model averages are
paired within eight schedule blocks. Blocks, not nodes, edges or model rows,
are the inferential units. P-values use Holm correction; intervals are
Bonferroni simultaneous t intervals. Complete blocks require all four models
in both arms. At least six complete blocks are required; failures and undefined
endpoints remain missing. Twenty-three contrasts were testable.

External comparisons include ten SNAP Facebook records, one SocioPatterns
school friendship survey and a later Copenhagen friendship-network extension.
ER, BA and WS generator settings were fitted on a separate eight-record
development panel, then kept fixed for transfer evaluation. They are not
re-fitted to make the test-set results look favorable.

Metrics include density, clustering, global efficiency, transitivity, degree
inequality, largest-component fraction and demographic mixing. Wasserstein-1
distances compare metric distributions. Instruction settings have explicit
weights; model-specific distances are averaged rather than silently replacing
them with a pooled-model distance. Failed generations are not zero-filled.

Target-density ER, edge-count-matched G(n,m), and finite degree-swap controls
are **target-informed diagnostics**, not equal-information transfer competitors.
Overlapping node draws and overlapping Facebook records are not independent
communities. Min/max spans are descriptive ranges, not confidence intervals.
Copenhagen is conditioned on observed friendship endpoints; a complete
observation roster is unavailable.

## Main Findings To Remember

1. **RQ1:** none of the 12 English-language country-framing contrasts passed
   the original correction on the designated endpoints. Nonsignificance does
   not establish equivalence or cultural invariance.
2. **RQ4:** iterative US-Hindi clustering was **0.04528 lower** than US-English;
   Holm-adjusted **p = 0.02383**, simultaneous interval **[-0.08500, -0.00556]**.
   It was the only significant primary contrast. The direction was negative in
   all eight blocks, but significance survived only **3/8** block-omission fits.
   This is assumption-dependent evidence about the instructions, not Hindi speakers.
3. **RQ2:** politics ranked first in all **28 model/setting cells** for each
   person-level method (Local, Sequential, Iterative) under excess same-group
   edge share, Coleman-type and Newman scores. For Global, politics ranked
   first in **1/28** cells by excess share and **6/28** by the other two scores.
   These results use 878 nonempty primary graphs. One roster, correlated
   attributes, category prevalence and fixed prompt-field order prevent causal
   attribute-importance claims.
4. **RQ3:** the iterative Hindi contrast had the same direction in all four
   model-specific means. Some other contrasts reverse across models; this is
   descriptive heterogeneity, not general model equivalence.
5. **External evaluation:** classical generators remain competitive. Rankings
   change with metric, source and network size. There is no stable overall winner.
6. **Density diagnostic:** target-density ER closed about **86% of the
   fixed-WS-to-Global efficiency-distance gap** across the original 11 records.
   This is not 86% accuracy, a bounded quality score or a causal explanation.
   ER is closer in the school comparison; Global is closer on the Facebook mean.
7. **Sampling:** Copenhagen's mean clustering was approximately **0.315 in the
   full observed record versus 0.080 in induced 50-node draws**. A match to a
   sampled benchmark need not reproduce the full community.

The strongest defensible contribution is an auditable analysis of instruction
sensitivity, method-dependent mixing and evaluation sensitivity. We expose how
claims change with density, sampling, baseline information and missing outcomes.
We do not claim to have invented LLM network generation or established human realism.

Latest statistical audit (November 15): all 24 primary contrasts reproduced
within `3.11e-15`; all 890 adjacency hashes checked; 7,902 RQ2 score values
matched within `4.45e-16`; all 18 paired excess-mixing contrasts and denominators
matched; 30 focused tests passed. These are scoped numerical/software checks,
not human peer review or verified statistical assumptions.

## What The Deployment Shows

The deployment is a **saved-output evidence explorer**, not live agents or new
model generation. Opening it makes no paid study requests.

- Filter primary runs by model, method, framing, instruction language and seed.
- Compare up to six selected primary runs in stacked 3D views, including matched
  model, framing, language and method comparisons.
- Replay saved updates using play/pause, step controls, a timeline and final-state
  navigation. Playback speed is presentation speed, not API elapsed time.
- Select a fictional persona, inspect its actual saved neighbors and follow
  recorded updates affecting its ties.
- Rotate, pan and zoom; change colors and labels; inspect shared versus different
  final ties. Layout positions and layer spacing are visual, not social distances.
- Inspect measurements, coverage, recorded updates and provenance; export selected
  evidence and share a selected view.
- Browse all 896 valid final networks in a separate four-method explorer with
  primary/supplement labels. One selected graph renders at a time; networks are
  not merged into a fictitious 896-network social graph.
- Link to original adjacency/PNG files and research evidence. The current final
  browser uses interactive saved edges, not a deployed wall of PNG tiles.

Supplemental graphs have final-graph viewing only, not invented formation
histories. Calibration and engineering-pilot interfaces are archived and
excluded from the current production package. Raw responses, ledgers,
credentials and operational scripts are not public deployment assets.

The stack includes Python 3.11.9, NetworkX, NumPy, pandas, SciPy, Matplotlib,
Jupyter, JavaScript and Three.js. The recorded pipeline separates experimental
evidence, derived tables, publication figures and public viewer exports.

## Scope Versus Pending Work

### Included In This Update

A **safe versioned primary rebuild command** now requires an explicit NEW direct
child of `outputs/primary_analysis`:

```powershell
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\run_primary_evaluation_v2.py --output outputs\primary_analysis\local_reproduction
```

Existing files/directories, nested release destinations and competing creators
are refused. Analysis failures can mark only the new rebuild incomplete. The
historical `scripts/run_primary_evaluation.py` remains byte-preserved because
published manifests pin it. **Do not execute that historical entrypoint:** it
can overwrite the sealed historical derived directory. Current instructions
point to v2. The fix does not change statistics or regenerate study data.

Full rebuilds need the verified saved-response archives; inspecting/executing
the analysis notebooks from published tables does not require a paid call.

### Pending Engineering And Presentation Work

- The v6.2 reporting release revises figures 02, 10, 15, 18 and 19;
  deployment is separate from the offline notebook release.
- Publish the local established-name correction and synchronize production
  branding. The live page still had the unwanted old display title at the last
  November 16 check; the hosting URL intentionally remains unchanged.
- Synchronize deployment findings/figure assets with the latest reviewed research
  release after verification. GitHub research files and deployment are separate
  release surfaces; a repository commit does not automatically update the site.
- Revalidate affected notebooks and end-to-end browser flows after those changes.
  Do not present earlier passing checks as fresh validation of a future version.

The last documented deployment is **version 7, November 14**. GitHub main was
verified at `340fc6ae32d482968e91953aaa88959e9715065c` before this local update.
That main commit contains the November 15 v6.1 figure integration; the safe
rebuild/context commit and name restoration require a later push to become public.

### Future Research, Not Completed Or Authorized

- Independent persona rosters and counterbalanced prompt-field order.
- Human bilingual instruction-equivalence review and stronger construct validation.
- More independent, semantically compatible real-world communities and a frozen
  held-out behavioral/prediction evaluation plan.
- Compute-comparable generation and carefully validated demographic/degree controls.
- Manuscript framing and venue selection with the professor; no publication guarantee.

These extensions may require separate design, data permissions and spending
approval. Existing evidence does not establish cultural causation, individual
friendship prediction, political polarization dynamics, or ICML readiness.

## How Claude Should Help Me

Explain results in clear language I can use with my professor, while keeping
effect sizes, denominators, limitations and evidence links exact. Challenge
unsupported conclusions instead of manufacturing positive findings. Keep the
four research questions central and distinguish completed work from proposals.

Preserve all primary failures, empty outcomes, supplemental labels, original
receipts and sealed releases. Do not silently substitute selected supplements,
rewrite historical manifests, call providers, query a live ledger, or make new
paid calls. A passing test does not establish scientific validity.

The professor's five papers are references for research framing and evaluation,
not findings we must reproduce. Their review and the extension results are here:

- <https://github.com/hemu77/UA_Capstone/blob/main/docs/PRIMARY_LITERATURE_REVIEW.md>
- <https://github.com/hemu77/UA_Capstone/blob/main/docs/research/NEXT_STEPS_20261114.md>
- <https://github.com/hemu77/UA_Capstone/blob/main/docs/research/EXTENSION_RESULTS_20261114.md>

Prefer focused fixes with regression checks, professional readable figures,
reviewable commits and truthful handoffs. Preserve the established project name.
