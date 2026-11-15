# Population revision: proposed design, not collected evidence

The existing graphs remain an archive. Their 50 synthetic US personas are not
a representative world population. Changing the country named in a prompt
measures a framing response conditional on that roster. More synthetic records
do not by themselves repair the selection assumptions.

## Political affiliation decision

- [x] Preserve Democrat/Republican and all original graphs in the historical US work.
- [ ] Use adults aged 18+ in a new, independently versioned roster.
- [ ] Omit political affiliation from the primary cross-country prompt comparison.
  This narrows the estimand: it measures behavior without an explicit political cue.
  Other attributes may still act as proxies, so this does not remove all bias.
- [ ] Add a matched sensitivity arm with a political cue visible versus hidden.
  Keep other persona fields, order, conditions and sampling scheme fixed.
- [ ] For country-grounded populations, source actual country-specific political
  categories and document who is eligible and how missing/non-affiliated cases work.
  Do not equate parties across countries or silently translate US party names.
- [ ] Interpret within-country political homophily separately unless a reviewed
  measurement mapping supports cross-country comparison. A left/right scale is
  not automatically equivalent across countries either.

## Two distinct questions

1. **Controlled synthetic experiment:** hold adult attributes and identities fixed,
   vary country framing or instruction language separately. Use several independent
   rosters and report sensitivity to composition. Claim prompt-conditioned model
   behavior on this constructed population, not real national friendship patterns.
2. **Population-grounded extension:** build country-specific adult rosters using
   documented sources, including plausible joint distributions, missingness and
   eligibility. Changing both roster and context requires a crossed design to
   separate their effects. Marginal census matching alone is insufficient.

Before generation, review translations, correlations between demographic fields,
impossible combinations, ordering effects, refusal/parse failures and feature
ablations. Pre-specify primary outcomes and treat a generated network, not each
dependent edge, as the replication unit. Astra should review this design before
confirmatory collection. No design guarantees conference acceptance.

## Requested model panel

Proposed: `gpt-5-mini`, `gpt-6-luna`, `gpt-6-sol`, plus
`Qwen/Qwen3-8B` and `mistralai/Mistral-7B-Instruct-v0.3` from Hugging Face.
This is a draft panel, not a claim that all five have been run or are equivalent.
The GPT-5 and HF paths are not yet implemented in the paid runner.

Public HF metadata checked on 2026-09-29 lists Apache-2.0 for both open models.
Qwen lists live providers; Mistral's listed provider reports an error, so its
execution route remains unresolved. HF hosting does not mean free inference.
Review multilingual competence before final selection, then pin revisions,
provider, chat template, quantization and decoding/thinking settings. Keep those
differences visible in the interpretation of model comparisons.

Sources: [GPT-5 mini](https://developers.openai.com/api/docs/models/gpt-5-mini),
[Qwen](https://huggingface.co/Qwen/Qwen3-8B),
[Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

The existing `study_protocol.json` stays unchanged because it identifies already
collected receipts. A new protocol must precede new collection, share the existing
$50 cumulative ceiling, and include hosted HF costs if used. No new paid calls
were made for this population-design and visualization revision.

## Layer laboratory

Open `http://127.0.0.1:8765/layers.html` after building the viewer. Add up to six
saved runs; orbit, separate planes, filter shared/different edges, and select a
person to trace their identity and neighbors. The table and topology plot show
final saved metrics, unaffected by visual edge filtering. A dashed vertical line
connects copies of the same person; it is not a social edge. Layer height and
force-layout coordinates have no geographic or causal meaning.

The archive link keeps the existing filtered plots and pairwise explorer available.
Missing new-model results are not synthesized for display.
