# Independent Review Prompt For Claude

Review this repository's primary study and offline analysis critically. This is
a read-only methodological/code audit, not a request to endorse the research.
Do not change frozen evidence, call providers, access credentials/ledgers, start
collection, or merge selectively completed supplements into primary inference.
Do not assume passing tests or five agent reviews establish scientific validity.

Start with `ANALYSIS_START_HERE.md`, `docs/PRIMARY_ANALYSIS_FINDINGS.md`,
`docs/PRIMARY_LITERATURE_REVIEW.md`, the two executed analysis notebooks, and
`outputs/primary_analysis/20261105/evaluation_manifest.json`. Check the manifest
and code hashes. Raw receipt archives live in the linked hash-verified release;
state explicitly which evidence is unavailable rather than assuming it passes.

Audit these questions independently:

1. Does the exact primary cohort remain 896 original attempts, 890 valid,
   six failures and 12 valid empty graphs? Are all six adaptively selected
   supplementary graphs excluded from all primary tests? Verify identifiers,
   row selection and missing-versus-zero handling, not just headline counts.
2. Is the paired schedule block the correct inferential unit? Check four-model
   averaging, shared US-English controls, eight seeds, complete-block rule,
   minimum six, fixed 24-hypothesis Holm family and simultaneous intervals.
   Do not treat models, nodes or dyads as independent replications.
3. Independently reproduce iterative Hindi versus US-English clustering:
   mean about -0.04527889, t/Holm p about .02383122, simultaneous interval
   about [-.08500171, -.00555608]. Reproduce all 24 contrasts and explain
   assumptions, selection, missingness and uncertainty, not merely arithmetic.
4. Verify sign-flip raw .0078125 / Holm .1875 and its symmetry assumption,
   coarse resolution, and exploratory status. Reproduce the 3/8 leave-one-seed
   Holm result using a freshly adjusted full family on each omission. Are
   missingness identification bounds clearly distinguished from CIs? Are
   per-model comparisons also shown on common complete seeds?
5. Check graph/hash validation, 7,120 deterministic equal-n/m references,
   empty graphs and failure handling. Is Monte Carlo SD correctly labeled?
   Do any plots imply confidence, realism or model-adjusted superiority
   unsupported by their reference design?
6. Inspect the eight real-network selectors, tie definitions, published size
   anchors, provenance and undirected projection. Assess convenience sampling,
   repeated Tailor waves, Moreno rating unions, Hi-tech reciprocity caveat,
   node-count/density confounding and absent matched human attributes. Are ECDF
   distances descriptive with no unjustified KS tests or optimized winner?
   Recommend a realistic held-out human evaluation rather than declaring
   structural resemblance to be behavioral or cultural fidelity.
7. Verify resource aggregation includes failed attempts, distinguishes received
   cost from reservations, never reallocates unsupported reservations and never
   mistakes serial API time for wall time. Is unequal compute clearly stated?
8. Execute notebooks top-to-bottom offline if feasible. Do figures agree with
   tables, show suppressed/undefined cells honestly, and retain caveats? Does
   rerendering preserve sealed outputs? Are original 69 source pins and original
   collection artifacts unchanged? Read `analysis_primary/`, the driver and tests.
9. Read the five exact professor-supplied papers from their primary sources.
   Verify the literature review accurately distinguishes static structure,
   dynamic agent behavior, multilingual measurement and human validation.
   Challenge novelty and venue-fit claims; propose concrete new experiments,
   not post-hoc hypotheses retroactively labeled confirmatory.

Return findings first, ordered by severity, with precise file/line references,
reproduction evidence, likely effect on conclusions and the smallest remedy.
Separate code defects, statistical assumptions, unsupported scientific claims,
missing evidence and future-study suggestions. Deduplicate findings. If no
defect is found, say so within the checked scope and list residual limitations.
End with: which claims can be shared now, which need softer wording, and which
require genuinely new real-world data. Never guarantee acceptance or describe
an AI audit as independent human peer review.
