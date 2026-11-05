# Figure Redesign: References, Not Templates

The actual figure pages in the professor's five papers were visually inspected,
not only their captions. Publisher access limits were handled with identified
author versions where needed. No source graphic is copied into this repository.

| Reference inspected | Useful visual principle | Applied here / not copied |
| --- | --- | --- |
| [Chang et al., author v2](https://arxiv.org/pdf/2408.16629v2), PDF p.5, Figs.3-4 | Compact metric facets, visible network-level points, restrained summary marks and a shared legend. | Show paired schedule points and compact panels. Their mean/SE is not substituted for our fixed-family simultaneous intervals. Their pooled node display does not justify node-level inference here. |
| [Piao et al., author v3](https://arxiv.org/pdf/2501.05171v3), PDF pp.17-20, Figs.1-4 | Panel letters, consistent encodings, explicit reference/intervention contrasts, methods in captions rather than oversized banners. | Use coherent panels and neutral reference lines. Do not invent time series, polarization dynamics, intervention effects or their uncertainty for our static graph experiment. Publisher/version equivalence is not asserted. |
| [Papachristou and Yuan, accepted author v7](https://arxiv.org/pdf/2402.10659v7), PDF p.4, Fig.1 | Dashed null/reference lines and structured small multiples make model comparisons interpretable. | Show zero-difference references and distinguish development baselines from target-informed controls. Do not copy their chart geometry, p-values or dynamic-network claims. |
| [Helwe et al., ACL proceedings](https://aclanthology.org/2025.findings-acl.883.pdf), PDF p.16, Appendix E Fig.1 | A repeated panel grid uses consistent axes across scenarios. | Keep framing/language labels and panel scales consistent. Political-compass quadrants and contours do not fit our endpoints and are not reused. |
| [Argyle et al., publisher PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/035D7C8A55B237942FB6DBAD7CAA4E49/S1047198723000025a.pdf/out_of_one_many_using_language_models_to_simulate_human_samples.pdf), PDF pp.8,11, Figs.3-4 | Direct source contrasts, restrained colors and faceted comparisons emphasize substantive correspondence. | Use a quiet palette, legible source labels and common metric units. Our human-network structural comparison is not their matched survey validation. |

## What Was Wrong With The Earlier Set

Large headlines, repeated cohort/disclaimer blocks and oversized white margins
competed with the data. Repeated labels and per-point sample counts crowded the
panels. A long report footer embedded in every image made the plots read like
dashboard screenshots, not manuscript panels. Changing fonts alone would not
solve those problems.

## Replacement Rules

- Preserve all earlier figures and the original primary data/analysis seal.
  The recommended manuscript set is a new, separately labeled derivative.
- Use compact, aligned panels, restrained typography and limited consistent
  colors. Keep captions, assumptions and denominators adjacent in the notebook
  or caption document rather than baking a warning paragraph into every PNG.
- Plot estimates/observations appropriate to the question: full-family
  simultaneous intervals for the locked contrasts; raw paired blocks for
  schedule variation; no CI around descriptive model means or W1 distances.
- Never hide suppressed inference, failed attempts, undefined values or the
  separate supplementary cohort. Visual polish must not become evidence polish.
- Provide PNG for browsing and vector SVG/PDF for manuscript use. Inspect the
  rendered outputs for clipping and ambiguity; statistical correctness alone
  does not make a good figure.

The redesign borrows communication principles, not authority. A paper's visual
style does not establish that our design, assumptions or conclusions are valid.
