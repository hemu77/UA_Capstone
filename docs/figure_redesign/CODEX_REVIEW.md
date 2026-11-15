# Review Corrections: v6.1

Original imported commit: `5c6fe253a89d97ab537fc3ed114b50e6f786d7f3`, parent `b24e20d`.
Its commit, renderer and released artifacts remain intact. Corrective commits are
separate. [PR #3](https://github.com/hemu77/UA_Capstone/pull/3).

## Findings And Fixes

1. **Figure 16 sign agreement:** v6 counted the larger of positive/negative cells,
   not agreement with the displayed mean. Religion, Sequential minus Local, has
   mean -0.001161613 and 12 negative / 16 positive cells. v6.1 correctly says **12/28**.
   All 18 attribute/procedure-pair counts are checked against the paired-cell table.
   The missing-pair note no longer incorrectly attributes every missing pair to Global.
2. **Caption scope:** observed-only Figures 10 and 18 no longer claim a primary LLM
   cohort; Figure 11 is explicitly observed graphs plus conditional controls. Figure
   01 now distinguishes Bonferroni intervals from Holm p-values without claiming
   that they can agree only at the smallest p-value.
3. **Incomplete historical verification:** the current verifier additionally checks
   the original v2 artifacts/source pins and extension source pins. Negative tests
   simulate changed v2 PNG, extension renderer and pinned old checker bytes without
   modifying those files. Original v6 is also verified, including its manifest pin.
4. **Notebook integration:** all six display v6.1. Notebook 06's generator reproduces
   the actual current sources and captions, imports from an `analysis/` working
   directory, and explains that Figure 17 shows ratios, not fractions summing to one.
   Notebook 05 now describes the actual three-score margin panels rather than the
   previous design. Current prose and preview links point to the current figures.
5. **Test isolation:** the preparation test now writes only under a temporary root,
   with a guard that rejects any write outside it. Captured source hashes describe
   the actual code; no frozen implementation was edited.
6. **Repository navigation:** 20 unpinned historical documents moved to calibration,
   planning and legacy folders. Current full-run, supplements and failure catalogs
   have separate entry points. Hash-pinned source/data paths were not moved.
7. **Cross-format rendering:** PDF inspection found diagonal color artifacts in
   Figure 17's raster matrices, despite a correct PNG. Vector cells now preserve
   exact cell boundaries, colors and values in the PDF. Browser inspection also
   found missing DejaVu Sans fonts caused serif substitution in the SVG exports.
   All 19 current SVGs now outline the same glyphs, so no reader-installed font is
   required. This fixes portability, not data or styling; PNG/PDF text remains
   unchanged outside the two corrected figures.

## Release Boundaries

[v6.1 artifacts](../../outputs/publication_figures/20261115_v6_1/) contain all 19 figures
in PNG/SVG/PDF with plotted CSVs. Figures 16 and 17 receive the narrow corrections;
all 19 SVGs receive portable glyph outlines. The other **17 PNG/PDF/CSV sets are
byte-identical** to v6. Figure 17's plotted values are unchanged. The v6.1 manifest binds the original
release, correction code, input hashes and reviewed captions.

Original primary estimates, the 24-hypothesis family, weights, baselines and all
sealed data remain unchanged. Six selected supplements remain excluded. No model
calls, ledger access, new generation, hypothesis tests or deployment were performed.

## Verification

Current entry points (frozen Python 3.11.9, kernel `ua-capstone-final060d`):

```powershell
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\check_figure_release_v6.py
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\check_research_extension.py
rtk proxy .\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

The first checker runs all six notebooks from `analysis/` and verifies 41 exact
embedded PNGs and captions. The second also enforces notebook 06 generator parity.
`check_deep_review.py` now forwards to the current checker. The pinned older
`check_publication_notebooks.py` and v3 `review_publication.py check` remain historical
interfaces whose old-picture requirements do not describe the current notebooks;
their historical evidence is still checked, not bypassed.

Two bounded read-only review lanes inspected numerical/visual correctness and
verification/test integration. All 19 PNGs and rasterized PDFs were inspected,
and all 19 SVGs were rendered in an isolated headless Chrome profile. The reviewer independently
reproduced the main contrasts, matrix ratios and reviewer diagnostics. These are
AI-assisted checks, not independent human peer review.

Final verification passed on November 15, 2026:

- **410 offline tests**, including eight correction regressions; no failures.
- **Six frozen-kernel notebooks**, with 41 hash-matched PNGs and captions.
- Notebook 06 generator parity and the primary/held-out/manuscript semantic checks.
- Original v6 release and **290 additional historical file pins** verified.
- **203 local documentation/notebook links** checked across 20 current files.
- All 19 PNGs, PDF renderings and browser-rendered SVGs inspected.

Windows' frozen kernel emitted the existing ZeroMQ selector-thread warning;
execution still succeeded. The duplicate-ZIP-name warning came from an intentional
malformed-archive test. No frozen roster repair was needed. SVG text is outlined
for portable display; plotted CSVs and notebook captions retain readable text.
No GitHub CI workflow is configured; these are local verification results.

## Interpretation Limits

- One fixed fictional roster, four model configurations and eight paired blocks.
- The one adjusted Hindi contrast remains sensitive to block omission. The exact
  sign-flip diagnostic cannot reach the first Holm threshold with eight blocks;
  its p-value floor is not additional evidence against the contrast.
- Target-density ER uses target information; it is not a transfer competitor.
- Overlapping public records and induced draws are not independent populations.
- RQ2's score ordering is descriptive; neither a conditioned reviewer diagnostic
  nor roster association establishes independent attribute importance.
- Additional reviewer diagnostics in the original v6 report were spot-reproduced,
  but are not new confirmatory tests or a new sealed primary result table.

No publication-readiness guarantee or human behavioral-validation claim follows.
Raw public datasets were not downloaded or reconstructed again during this review.
