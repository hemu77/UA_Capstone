# Design Review: Branch `r1/social-netw`

**Date:** October 7, 2026
**Branch reviewed:** `r1/social-netw` at `3220ca7` (3 commits ahead of `main` at `7676526`)
**Requested focus:** research scope, translation-validation approach, repetition count, analysis plan
**Reviewer:** Claude (AI coding assistant), requested by the repository owner. This is
an independent, evidence-based design review. It is **not** human bilingual validation,
professor approval, or authorization to spend.

---

## 0. Summary

The engineering work in this branch is strong. Provenance, receipt replay, ledger
reconciliation, fail-closed spending gates, and the honest separation of fixtures,
V5 and V6 evidence are better than most published LLM-simulation studies. The
documentation is careful not to overclaim.

The scientific design still has problems that **more spending will not fix**.
In order of importance:

| # | Finding | Severity | Cheapest fix |
| --- | --- | --- | --- |
| 1 | With 8 repetitions per model × setting cell and Holm over the declared 24-hypothesis family, the smallest detectable effect on the primary outcomes (clustering, modularity) is about **5–20× larger** than the between-condition spread actually observed. | **High** | Make the **model-averaged contrast** the primary estimand (n = 32 per contrast at 8 reps/cell); per-model results become secondary estimates. |
| 2 | The V6 global prompt is degenerate: 22/26 graphs are empty, including **14/14 for GPT-6-Luna**. Under the V5 prompt, the same model and roster produced **0/14** empty graphs. The docs say the cause "cannot be established", but the V5→V6 contrast is strong evidence that the prompt change caused it. | **High** | A predeclared global ablation (≈64 graphs, well under $1) before any "global 32" allocation. |
| 3 | The conservative bounded-mean (Hoeffding) fallback can **never reject** at realistic effect sizes. Its 0% false-positive rate reflects 0% power, not safety. | **High** | Replace it (Section 4.2). |
| 4 | Each instruction language has **one** wording. Language and wording are perfectly confounded, so RQ4 cannot distinguish "Hindi" from "this particular Hindi sentence". | **High** | A translation-variant arm: 2 extra independent translations per language plus 2 English paraphrases, on a cheap method. |
| 5 | The V6 prompt text has no recorded semantic review; `TRANSLATION_REVIEW.md` covers V5 only. Every US-Portuguese prompt contains a grammatical agreement error. | Medium | Fix in the next contract; obtain a short fluent-reader check (about 10 sentences per language). |
| 6 | The research questions promise homophily, but every primary outcome is topology. Age assortativity is the **only** metric whose between-condition spread exceeds the noise floor. | Medium | Either make homophily primary or reword RQ1 as a topology question. |
| 7 | Power figures are quoted at α/6 and α/288, but the declared families are 24 and 84. | Low | Update `plan.md` and related docs. |
| 8 | Several documents contradict each other about which dataset the simulator shows, and `plan.md` says the reviewer's `review_experiments/` bundle was unavailable, although it exists in the working tree. | Low | Doc cleanup (Section 6). |

**Recommendation:** record **"revise first"** in `TEAM_APPROVAL.md`. Do not authorize the
896- or 1,568-graph scenarios as currently specified. The two cheap studies proposed
here (global ablation and translation variants) cost a few dollars together and should
come before any main-collection budget.

---

## 1. What Was Reviewed and How

### Files read

- `README.md`, `plan.md`, `TRANSLATION_REVIEW.md`
- `docs/CALIBRATION_V6.md`, `docs/CALIBRATION_V6_RESULTS.md`,
  `docs/STATISTICAL_ANALYSIS_DECISION.md`, `docs/PROFESSOR_TEAM_REVIEW.md`,
  `docs/TEAM_APPROVAL.md`
- `revision_next.py` (V6 prompt text), `revision224_prompts.py` (V5 prompt text),
  `statistical_review.py`, `outputs/statistical_review_v6/report.json`
- `outputs/calibration_v6/5fb3715550db/network_metrics.csv` (all 104 V6 graphs)
- `stats/revision896_retry_v5/network_metrics.csv` (68 V5 graphs)
- Paid V6 receipts under `outputs/calibration_v6/5fb3715550db/runs/` (prompt text search)
- The untracked external-review bundle `review_experiments/` (scripts e1–e9 and logs)

### Independent computations

All numbers marked *(computed)* below were produced from the committed CSVs with
NumPy/SciPy in a throwaway environment, using the scripts in the Appendix. No API
calls were made and no repository file other than this one was modified.

### Not verified

- The 142-test Python suite and the notebook were **not** rerun. They require the
  pinned Python 3.11.9 environment, which is not installed on this machine.
- Private-ledger reconciliation was not rechecked; it needs the private `budget.sqlite`.
- The variance estimates in Section 3 rest on GPT-6-Luna only (7 settings × 2 reps,
  so about 7 degrees of freedom per pooled SD). Treat them as order-of-magnitude
  planning inputs with roughly ±30% uncertainty, not precise parameters.

---

## 2. Research Scope

### 2.1 What the study can and cannot claim

The docs are careful and correct on the big framing points:

- Country labels are one-sentence prompt framings, not cultural constructs.
- Instruction language is varied while persona JSON stays English, so this is
  instruction-language sensitivity, not "speaker language".
- The seven-setting design (four countries in English, plus US in three other
  languages) is not a factorial and cannot identify country × language interactions.
- One designed 50-person roster does not support population or national-behavior claims.

No changes are needed there. The problems are breadth and alignment.

### 2.2 The scope is too broad for the size of the signal

Current scope: 4 RQs × 4 methods × 4 models × 7 settings. Under the proposed
multiplicity plan that gives, **per method**, 24 (RQ1) + 24 (RQ4) + 84 (RQ3) = 132
primary hypotheses, or **528 across the four methods**, in 12 separately corrected
families with no study-wide error control.

Section 3 shows that the observed between-condition differences are small. A design
that spreads a fixed budget over 528 tests will mostly produce non-significant
results, which (as the docs correctly say) cannot be read as equivalence.

**Recommendation:**

1. **Confirmatory:** RQ1 (country framing, English fixed) and RQ4 (instruction
   language, US fixed), each estimated as a **model-averaged** contrast.
2. **Descriptive / secondary:** RQ2 (which attributes show mixing) and RQ3 (model
   differences). RQ3 alone is 84 hypotheses per method, and "different model
   configurations behave differently" is expected rather than a contribution. Report
   per-model estimates with intervals and a heterogeneity summary instead of 84 tests.
3. Consider fixing **two** methods as confirmatory (for example local and sequential,
   which are cheap and stable) and treating global and iterative as exploratory until
   their issues are resolved. Iterative is about 71% of calibration cost.

### 2.3 The global method: the empties are very likely prompt-caused

`CALIBRATION_V6_RESULTS.md` says: *"Calibration cannot establish why."* The evidence
is stronger than that wording suggests.

*(computed)* Empty global graphs by prompt version:

| Model | V5 prompt: empty / total | V6 prompt: empty / total |
| --- | ---: | ---: |
| GPT-4.1 | 0 / 1 | 4 / 4 |
| GPT-5.6-Luna | 0 / 1 | 3 / 4 |
| GPT-6-Luna | **0 / 14** | **14 / 14** |
| GPT-6-Sol | 0 / 1 | 1 / 4 |

For GPT-6-Luna, the roster, the seven settings and the two-repetition structure are
the same. The V5 minimum density was 0.0196 (24 edges). What changed:

- V5 global: *"Choose friendships among the listed people. Friendships are undirected. …"*
- V6 global added: *"Each person may have zero, one or multiple friends. There is no
  required number of friendships … If the whole network has no friendships, output
  only NONE."*
- Candidate display order is now randomized (a smaller and less plausible cause).

Telling a model that zero friendships is acceptable, and giving it a one-word exit,
predictably moves it to the cheapest compliant answer. The V6 wording turned a
generative task ("propose friendships") into a permissive one ("list friendships if
any exist"). Strictly, prompt wording and display order are confounded, so the
cautious doc wording is technically defensible, but the team should be told that the
prompt is the leading explanation.

**Why this matters for the estimand.** Fictional strangers given no shared context
have no reason to be friends, so `NONE` is arguably the most honest answer to the V6
prompt. The study's question needs a generative framing such as *"Construct a
plausible friendship network among these people, who live in the same community in
{country}."* That is a substantive design choice the team must make explicitly.
Whatever is chosen, the V5 failure mode (near-perfect matchings in non-English
Luna graphs, 48/50 degree-one nodes) should also be guarded against.

**Proposed predeclared ablation (cheap):**

| Arm | `NONE` clause | "No required number" clause | Generative framing sentence |
| --- | --- | --- | --- |
| A (current V6) | yes | yes | no |
| B | no | yes | no |
| C | yes | no | no |
| D | no | no | yes |

Run 4 models × 4 repetitions × 4 arms in US-English = 64 global graphs. At the
measured global cost (V5 nonempty global was about $0.004 per graph), this is well
under $1. Predeclare the decision rule, for example: *"choose the arm with the lowest
empty rate whose degree-one share is below 0.8; ties go to the arm closest to V6."*
Then freeze the winning wording as a new contract and reforecast. Do not mix the
ablation graphs into confirmation data.

Until this is done, **do not buy "global 32"**. Under the current prompt GPT-6-Luna
would return 32 empty graphs per cell, which carries no information about RQ1 or RQ4.

### 2.4 Primary outcomes do not match the research questions

RQ1 asks whether country framing changes "homophily and topology". RQ2 is entirely
about demographic mixing. Yet the proposed primary outcomes are all topology:
density and degree-one share (global), clustering and modularity (others). The
interesting claim for a paper is almost certainly about **who** the model pairs up
(religion, politics, age), not about clustering coefficients.

*(computed)* Ratio of the SD of the seven Luna condition means to the pooled
within-cell SD. With 2 reps per cell, pure noise produces a ratio of about
1/√2 ≈ 0.71, so values well below 0.71 show no detectable condition signal:

| Metric | local | sequential | iterative |
| --- | ---: | ---: | ---: |
| clustering | 0.16 | 0.47 | 0.75 |
| modularity | 0.28 | 0.44 | 0.43 |
| Coleman gender | 0.39 | 0.24 | 0.32 |
| Coleman religion | 0.45 | 0.96 | 0.46 |
| Coleman political orientation | 0.27 | 0.62 | 0.62 |
| **age assortativity** | **1.68** | **1.51** | 0.65 |
| density | 0.19 | 0.24 | 0.45 |

Age assortativity in local and sequential is the only metric clearly above the
noise floor. Sequential religion homophily is borderline. Local clustering, a
proposed primary outcome, shows **less** variation between conditions than noise alone
would create.

This is weak evidence (7 cells, 2 reps), but it points the same way as the
reviewer's V5 analysis (`review_experiments/e2_paired.py`): the homophily
measures, not the topology measures, are where conditions differ.

**Recommendation:** make attribute mixing primary, each measured relative to the
already implemented degree-preserving controls so that the score is not a by-product
of degree sequence. Candidates: age assortativity, plus one categorical Coleman index
chosen in advance with a written rationale. Keep topology as secondary. If the team
prefers topology as primary, reword RQ1 so it no longer promises homophily.

### 2.5 Smaller scope points

- **Nomination quotas.** Quota draws `floor(clamp(Exp(5), 1, 20))` mechanically fix
  most of the density in local, sequential and iterative. The docs already say this
  correctly. Density should never be a primary outcome for those methods.
- **One roster.** More seeds do not address roster generalization. A second roster
  run on one cheap method (local) would cost a few dollars and would let the paper say
  whether effects replicate across rosters. That is optional but valuable.
- **Weak country manipulation.** "The social setting is India." is one sentence.
  Small or null country effects may say more about the strength of the manipulation
  than about the model. State this as a limitation.

---

## 3. Repetition Count

### 3.1 Per-cell repetitions cannot detect the observed effects

*(computed)* Smallest detectable standardized paired effect (d_z) at 80% power,
two-sided, exact noncentral t:

| n | α = .05 | α/6 | **α/24 (declared RQ1/RQ4 family)** | **α/84 (declared RQ3 family)** | α/288 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.156 | 1.650 | **2.093** | **2.560** | 3.097 |
| 16 | 0.749 | 0.986 | 1.166 | 1.331 | 1.499 |
| 32 | 0.511 | 0.652 | 0.751 | 0.836 | 0.917 |
| 64 | 0.356 | 0.447 | 0.509 | 0.561 | 0.609 |

The existing docs' figures (1.156, 1.650, 3.097) are arithmetically correct, but they
quote α/6 and α/288. Neither is a declared family. The relevant figures are 2.09 and 2.56.

Converted to raw units using the pooled within-cell SD from GPT-6-Luna V6:

| Method / metric | Observed SD between settings | Detectable at n = 8, α/24 | Detectable at n = 32, α/24 |
| --- | ---: | ---: | ---: |
| local clustering | 0.006 | 0.109 | 0.039 |
| local modularity | 0.014 | 0.146 | 0.052 |
| sequential clustering | 0.012 | 0.078 | 0.028 |
| sequential modularity | 0.021 | 0.141 | 0.050 |
| iterative clustering | 0.033 | 0.131 | 0.047 |
| iterative modularity | 0.026 | 0.178 | 0.064 |
| local age assortativity | 0.061 | 0.108 | 0.039 |
| sequential age assortativity | 0.084 | 0.165 | 0.059 |

With 8 repetitions per model × setting cell, the design can only detect effects
roughly 5–20× larger than anything seen across the seven calibration settings. This
matches the external reviewer's independent V5 result (`review_experiments/e3`):
the detectable effect at n = 8 exceeded the observed spread for 94% of metric × method
combinations.

### 3.2 Recommended fix: make the model-averaged contrast primary

If the primary estimand for RQ1 and RQ4 is the contrast **averaged over the four
model configurations**, each contrast uses 4 × r graphs per condition. At r = 8 per
cell that is n = 32, and each RQ1/RQ4 family drops from 24 to 6 hypotheses
(3 contrasts × 2 outcomes).

*(computed)* 95% interval half-width at n = 32 with α/6, and repetitions per cell
needed for a ±0.02 half-width (±0.02 is an illustration, **not** a recommended
meaningful effect; the team must set that):

| Method / metric | Half-width at 8 reps/cell (n = 32) | Reps/cell for ±0.02 |
| --- | ---: | ---: |
| local clustering | 0.026 | 12 |
| local modularity | 0.035 | 22 |
| local Coleman religion | 0.031 | 17 |
| local age assortativity | 0.026 | 12 |
| sequential clustering | 0.019 | 7 |
| sequential modularity | 0.033 | 20 |
| sequential Coleman religion | 0.016 | 5 |
| sequential age assortativity | 0.039 | 28 |
| iterative clustering | 0.031 | 18 |
| iterative modularity | 0.042 | 32 |
| iterative age assortativity | 0.085 | 127 |

Interpretation:

- 8 reps/cell is roughly adequate for sequential and borderline for local, **if** the
  primary estimand is model-averaged.
- Iterative is both the most expensive and among the noisiest methods. Its age
  assortativity is far too noisy for confirmatory use at any affordable n. This
  supports making iterative exploratory.
- Pooling assumes that a model-averaged effect is meaningful. Report per-model
  estimates and a heterogeneity measure (for example the model × condition
  interaction variance) alongside it. Do not pool silently.

### 3.3 Allocation process

1. Team sets a **raw-unit precision target** per primary outcome, before data.
2. Re-estimate within-cell SDs from all four models, not Luna alone. GPT-4.1, 5.6-Luna
   and Sol have only one repetition per cell, so a small variance stage (for example
   3 reps per model in US-English for local and sequential) is worth considering.
   Keep it separate from confirmation, as the docs already require.
3. Compute reps/cell per method from steps 1–2, cost it, and freeze it.
4. Global allocation is decided only after the Section 2.3 ablation.

---

## 4. Analysis Plan

### 4.1 What is already right

- The repetition unit is the network generation, not edges, nodes or decisions.
- Missing results stay in the multiplicity family.
- Undefined metrics stay undefined rather than becoming zero.
- A null result is explicitly distinguished from equivalence.
- Estimation-first reporting is preferred.
- The paired-t inflation finding is real. *(computed)* With uniform bounded
  differences at n = 8, paired-t plus Holm over 24 gives a family false-positive rate
  of **11.4%** (20,000 worlds), versus the repository's 12.5% (1,000 worlds). The
  per-test rate at α/24 is 0.0050 instead of the nominal 0.0021. At n = 32 the
  inflation falls to 6.0%.

### 4.2 The bounded-mean fallback has no power

`statistical_review.py` uses a Hoeffding interval with radius
`2 * b * sqrt(log(2m/α) / (2n))`. *(computed)* For m = 24:

| n | Half-width, metric in [−1, 1] | Half-width, modularity in [−2, 2] |
| ---: | ---: | ---: |
| 8 | **1.310** (wider than the whole range) | 2.620 |
| 32 | 0.655 | 1.310 |
| 64 | 0.463 | 0.926 |

To reject at n = 32 with 24 hypotheses, a clustering difference would need to exceed
**0.655**. Observed condition differences are about 0.01–0.03. The method reports 0%
false positives in the stress test because it essentially never rejects anything.
That is 0% power, not safety. The docs acknowledge it "can be very wide", but it
should not be presented as the conservative sensitivity analysis.

**Also note:** an exact sign-flip permutation test does not rescue n = 8. Its
smallest possible two-sided p-value is 2/2⁸ = 0.0078, which is above α/24 = 0.0021,
so no hypothesis in a 24-family could ever be rejected. This is a second argument
for model-averaged contrasts.

**Recommended replacements** (choose one and predeclare it):

1. **Primary:** a linear mixed model per method and outcome:
   `outcome ~ condition + (1 | model) + (1 | model:condition)` (plus a seed/schedule
   term only if pairing is justified, see 4.3). The `condition` coefficients are the
   model-averaged RQ1/RQ4 contrasts. Use Kenward–Roger or Satterthwaite degrees of
   freedom.
2. **Sensitivity:** a permutation test of `condition` labels within model at the
   pooled sample size (n = 32 makes exact or Monte-Carlo permutation feasible), or a
   cluster bootstrap over graphs.
3. Validate the chosen procedure offline with the existing `simulate()` harness, using
   variance components taken from calibration rather than uniform [−1, 1] draws.

### 4.3 Pairing is not justified at n = 8

The plan pairs conditions by shared randomization schedule (seed). That only helps if
outcomes sharing a schedule are correlated. In the reviewer's V5 analysis
(`review_experiments/e2_paired_stdout.txt`), the across-condition correlation for
the same repetition ranged from **−0.73 to +0.89** across metrics, with no consistent
sign. The model's own sampling noise probably dominates the shared schedule.

Pairing has a cost at small n. *(computed)* Critical t at two-sided α/24:

| Degrees of freedom | Critical t |
| ---: | ---: |
| 7 (paired, n = 8) | **4.75** |
| 14 (unpaired, 8 vs 8) | **3.77** |
| 31 | 3.36 |
| 62 | 3.21 |

**Recommendation:** estimate the schedule correlation from calibration plus the
variance stage. Pair only if it is clearly positive for the primary outcome;
otherwise analyse as unpaired or use the mixed model in 4.2. Record the rule before
confirmation data are seen.

### 4.4 The missing-data policy is too brittle

`family_summary` suppresses inference for a whole contrast if **any** repetition is
missing. Two consequences:

- **Global homophily is permanently undefined** under the current prompt: Coleman
  indices, age assortativity and modularity are undefined in empty graphs, so every
  global homophily contrast would be suppressed.
- **One transport failure** in any cell kills that contrast, even though the failure
  has nothing to do with the outcome.

**Recommendations:**

1. **Replace non-informative run failures.** A predeclared rule such as *"a run that
   stops on transport ambiguity, truncation or exhausted corrections is replaced by
   the next seed in the frozen list, and all failures are reported"* is standard
   practice. It is not cherry-picking, because the replacement decision does not
   depend on the network produced. The current "no automatic paid replacement"
   policy is a spending safeguard; it can stay as a pause-for-approval step rather
   than a permanent hole in the data.
2. **Empty graphs need a two-part analysis** if global stays in scope: (a) the
   probability that a graph is non-empty, compared across conditions, and (b) the
   metrics conditional on being non-empty, with that conditioning stated.

### 4.5 Multiplicity

- Holm within each 24- or 84-hypothesis family per method gives **12 separate
  families and about 528 hypotheses** with no study-wide control. The docs say this
  honestly, but a reviewer will notice. Model-averaged primary contrasts (Section 3.2)
  shrink the confirmatory set to 6 per RQ per method, or 24 to 48 in total, which is
  small enough to correct study-wide if the professor prefers.
- Name the primary outcome **per method** in advance (one, not two, if possible).
  Two outcomes per contrast doubles the family for little gain.

---

## 5. Translation Validation

### 5.1 What the current evidence shows and does not show

| Evidence | What it supports | What it does not support |
| --- | --- | --- |
| 72-call screen: revised 0/36 vs original 4/36 failures | The revised Portuguese one-choice wording stops the model repeating IDs | Meaning equivalence; any Hindi or Japanese claim |
| 12 first-response failures in 11,536 decisions (0.104%) | All four languages produce parseable answers | That the four instructions mean the same thing |
| Two AI reviewers (V5 text, `TRANSLATION_REVIEW.md`) | No obvious semantic slips in **V5** wording | Anything about the **V6** sentences actually used; independence from the generator's model family |

The docs are already candid that compliance is not translation accuracy. Two gaps
remain.

### 5.2 Gap 1: one wording per language

With exactly one instruction text per language, every RQ4 contrast compares two
specific sentences. Any difference could come from the language, from incidental
phrasing, or from tokenization of one word. The calibration already shows that
wording details matter a lot: the original Portuguese singular wording failed 4/6
times, the revised one 0/6.

`plan.md` §5B lists a "translation-noise arm" but it is unchecked and has no size.

**Proposed design:**

- For each of Hindi, Japanese and Portuguese, add **two independently produced
  translations** of the V6 English text. Use different sources from the original
  author: for example a different LLM family and a machine-translation system.
- Add **two English paraphrases** written without looking at the translations.
- Run on **local** with GPT-6-Luna (about $0.04 per graph), US setting:
  4 languages × 3 wordings × 4 reps = 48 graphs, roughly $2.
- Analysis: treat wording as a random effect within language. Claim a language
  effect only if it exceeds the between-wording spread within languages. If English
  paraphrases differ from each other as much as English differs from Hindi, RQ4 must
  be reported as wording sensitivity, not language sensitivity.
- Predeclare all of this before running, as was done for the wording screen.

### 5.3 Gap 2: the V6 text was never semantically reviewed

`TRANSLATION_REVIEW.md` reviews V5. The V6 text in `revision_next.py` adds new
singular instructions, new global instructions, a new Hindi context sentence and new
Portuguese country articles. `plan.md` records only a "bounded read-only code review"
for them.

This reviewer's own reading of the V6 text (AI reading, not native-speaker review):

| Language | Text | Observation |
| --- | --- | --- |
| Portuguese | *"O cenário social é os Estados Unidos."* | **Agreement error.** With a plural predicate, Portuguese requires *"O cenário social são os Estados Unidos"*. More natural: *"O contexto social é o dos Estados Unidos."* Confirmed present in **every paid US-Portuguese receipt** (global, local, sequential and iterative, all four models). It comes from the article fix in `revision_next.py` line 29. Because RQ4 holds the country at US, every Portuguese observation carries this error. |
| Portuguese | *"Escolha exatamente um amigo entre os candidatos elegíveis."* | Natural and correct. |
| Hindi | *"…न कि केवल एक-से-एक जोड़ियाँ"* | Reads as "not **only** one-to-one pairs". The English is "not a one-to-one pairing". The nuance shifts slightly from "this is not a matching task" toward "pairs that are not merely one-to-one". Minor. |
| Hindi | Context sentence (`सामाजिक परिवेश {country} है…`) | Understandable; "social environment is the United States" is slightly stiff but clear. |
| Japanese | Global (`…1対1の組み合わせだけを作る課題ではありません…`) | Good. Explicitly says "this is not a task of only making 1:1 combinations", which is closer to the English than the Hindi is. |
| All | `degree`, `mutual`, `ID`, `NONE` kept in English | Reasonable, because the JSON fields are English. Note that non-English prompts are therefore **code-switched**, which is itself part of the "language" treatment. State this in the paper. |

The Portuguese error does not invalidate the collected V6 data (the contract is
frozen and the error is recorded), but it must be fixed in any new contract, and it
should be disclosed if V6 Portuguese results are reported.

### 5.4 Human review is cheap here

The waiver is recorded honestly, but the effort it saves is small. The complete V6
instruction set is about **10 short sentences per language** (context, singular,
plural local, sequential intro, global, iterative add/drop, retry prefix and repair).
One fluent reader per language could check it in about 20 minutes using the
English text side by side. Classmates, a university language department, or a
paid one-off check would all serve. This would remove the single most attackable
limitation in the paper. Recommended procedure:

1. Send the reader the English and target text only (not the study's hypotheses).
2. Ask three questions per sentence: does it mean the same thing; is it natural; is
   anything ambiguous.
3. Record the reader's language background and their verbatim answers in a new
   `TRANSLATION_REVIEW_V6.md`, with any fixes going into the next contract.

A blinded back-translation by a third system (different from both the translation
author and the generator) is a weaker substitute if no human reader is available.

---

## 6. Documentation Consistency

| Location | Problem | Suggested fix |
| --- | --- | --- |
| `README.md` line 34 and line 258; `docs/VIEWER_INTERACTION_REVIEW.md` line 5 | Say the simulator now shows the 104 V6 graphs. | Correct (matches commit `233b7ed`). Keep. |
| `README.md` line 178 and line 184; `plan.md` line 17; `docs/CALIBRATION_V6_RESULTS.md` line 192; `docs/PROFESSOR_TEAM_REVIEW.md` line 4 | Say the simulator was left unchanged and still shows V5 / "the new 104 graphs have not been exported". | Update to match. `PROFESSOR_TEAM_REVIEW.md` matters most, because it is the document the professor reads. |
| `plan.md` line 38 | "Its `review_experiments/` scripts and bundle were not available here." | The bundle exists in the working tree (untracked). Either commit it (it is small and offline-only) and cite it, or reword. Several of its findings (e2 correlations, e3 power, e9 position effects) support this review. |
| `plan.md` line 51 | Power quoted at α/288. | Quote at the declared family sizes (α/24, α/84): 2.09 and 2.56 SD at n = 8. |
| `README.md` line 73 | "Proposed staged allocation: global 32, local 8". | Conflicts with the 22/26 empty-global finding. Mark as superseded pending the global ablation. |
| `docs/STATISTICAL_ANALYSIS_DECISION.md`, stress-test section | Presents the bounded-mean method's 0% false-positive rate as a favourable result. | Add the half-width table from Section 4.2 and state that the method has no practical power at the planned n. |

---

## 7. Recommended Next Steps

In order. Items 1–4 are offline and free; items 5–6 are small paid studies, each
needing its own explicit approval and cap, as the repository already requires.

1. **Decide the estimand.** Model-averaged RQ1/RQ4 contrasts as primary; RQ2 and RQ3
   descriptive; name one primary outcome per method, preferably a homophily measure
   relative to degree-preserving controls.
2. **Replace the bounded-mean fallback** with a mixed model plus a permutation or
   bootstrap sensitivity analysis, and validate it offline with calibration-based
   variance components.
3. **Write the failure-replacement and empty-graph rules** (Section 4.4) and the
   pairing decision rule (Section 4.3).
4. **Fix the documentation inconsistencies** in Section 6 and the Portuguese
   agreement error in the next prompt contract.
5. **Global ablation** (Section 2.3): about 64 graphs, well under $1.
6. **Translation-variant arm** (Section 5.2): about 48 local graphs, roughly $2, plus
   the fluent-reader check (Section 5.4), which costs nothing in API spend.
7. Only then: set raw-unit precision targets, compute repetitions per method
   (Section 3.3), reforecast, and bring a finite budget to `TEAM_APPROVAL.md`.

---

## Appendix A: Reproduction Scripts

All scripts are read-only. Run from the repository root with any Python that has
NumPy, SciPy and pandas. Exact numbers may differ slightly from those above in the
Monte-Carlo parts.

### A.1 Detectable effects, Hoeffding width, paired-t inflation

```python
import numpy as np
from scipy import stats
from scipy.optimize import brentq

def mde(n, alpha, power=.8):
    df, c = n - 1, stats.t.isf(alpha / 2, n - 1)
    return brentq(lambda d: stats.nct.sf(c, df, d * np.sqrt(n)) - power, 1e-3, 8)

for n in [8, 16, 32, 64]:
    print(n, [round(mde(n, a), 3) for a in [.05, .05/6, .05/24, .05/84, .05/288]])

for n in [8, 32, 64]:
    for b in [1, 2]:
        print(n, b, round(2 * b * np.sqrt(np.log(2 * 24 / .05) / (2 * n)), 3))

rng = np.random.default_rng(1)
for n in [8, 32]:
    d = rng.uniform(-1, 1, (20000, 24, n))
    t = d.mean(-1) / (d.std(-1, ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    print(n, 'FWER', (p.min(1) < .05 / 24).mean())

for df in [7, 14, 31, 62]:
    print(df, round(stats.t.isf(.05 / 24 / 2, df), 2))
```

### A.2 Within-cell versus between-setting variation (GPT-6-Luna, V6)

```python
import numpy as np, pandas as pd
d = pd.read_csv('outputs/calibration_v6/5fb3715550db/network_metrics.csv')
L = d[d.model == 'gpt-6-luna'].assign(cell=lambda x: x.culture + '_' + x.language)
for m in ['local', 'sequential', 'iterative']:
    for k in ['avg_clustering_coef', 'modularity', 'coleman_religion', 'age_assortativity']:
        g = L[L.method == m].groupby('cell')[k]
        within = np.sqrt(g.var(ddof=1).mean())
        between = g.mean().std(ddof=1)
        print(m, k, round(within, 4), round(between, 4), round(between / within, 2),
              'MDE n8/a24 =', round(within * np.sqrt(2) * 2.093, 3),
              'MDE n32/a24 =', round(within * np.sqrt(2) * 0.751, 3))
```

### A.3 Empty global graphs, V5 versus V6

```python
import pandas as pd
for path in ['stats/revision896_retry_v5/network_metrics.csv',
             'outputs/calibration_v6/5fb3715550db/network_metrics.csv']:
    g = pd.read_csv(path).query("method == 'global'")
    print(path)
    print(g.assign(empty=g.density == 0).groupby('model')['empty'].agg(['sum', 'count']))
```

### A.4 Portuguese agreement error in paid receipts

```powershell
Select-String -Path outputs/calibration_v6/5fb3715550db/runs/*portuguese*.json `
  -Pattern 'cenário social é os Estados' -List | Measure-Object
```
