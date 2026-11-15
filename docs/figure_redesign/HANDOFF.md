# Handoff for Codex: figure release v6

**Historical handoff from the imported commit.** Follow-up corrections and current
commands are in [CODEX_REVIEW.md](CODEX_REVIEW.md). Do not use the old restore or
checker instructions below as current workflow; the original v6 artifacts remain preserved.

Branch: `claude/figure-redesign-v6`, based on `main` at `b24e20d`. Pull request: see the PR that
carries this file (URL is reported in chat, not written here, to avoid a self-referential hash).
Nothing is merged or deployed. No paid calls, no generation, no ledger queries. Supplements excluded.

## What changed

| Path | Change |
| --- | --- |
| `analysis_figures/v6/` (`style`, `data`, `primary`, `heldout`, `sampling`, `mixing`, `release`) | New renderer for all 19 figures. Reads sealed tables only and re-verifies every historical seal on load. |
| `outputs/publication_figures/20261115_v6/` | New release: 19 x (PNG 300 dpi, SVG, PDF, `_plotted.csv`) + `manifest.json`. Deterministic: two builds were byte-identical. |
| `scripts/run_figure_release_v6.py` | Builds into a NEW directory; refuses an existing one. |
| `scripts/check_figure_release_v6.py` | Verifies the release, all historical seals, executes notebooks 01-06 on the frozen kernel and checks embedded PNG hashes and captions. `--write` refreshes outputs. |
| `tests/test_figure_release_v6.py` | 13 focused tests (cohort guard, counts, seals, display arithmetic, diagnostic tier). |
| `tests/test_publication_figures.py` | One assertion updated: notebooks import `show` from the v6 release. |
| `analysis/01`-`06` notebooks | Display v6 figures and captions; a banner cell was added; notebook 04 now shows all 19. Numerical cells are unchanged. |
| `docs/figure_redesign/README.md` | Findings, per-figure verdicts, references, main/appendix proposal. |
| `ANALYSIS_START_HERE.md`, `analysis/README.md`, `docs/PROFESSOR_RESEARCH_BRIEF.md`, `docs/research/EXTENSION_RESULTS_20261114.md` | Pointer to v6; sign-flip sentence corrected; density caveat added. |

Unchanged: every file under `outputs/` that existed at `b24e20d`, `final_study/`, `text-files/`,
`analysis_primary/`, `analysis_heldout/`, `analysis_extension/`, and the historical renderers and
checkers.

## Figure to notebook

| Notebook | v6 figures displayed |
| --- | --- |
| 01 | 01, 02, 07 |
| 02 | 01-07 |
| 03 | 08, 09, 10, 11, 12, 13, 15 |
| 04 | all 19 |
| 05 | 14 |
| 06 | 16, 17, 18, 19 |

## Commands and results (Python 3.11.9 venv, kernel `ua-capstone-final060d`, Linux)

| Command | Result |
| --- | --- |
| `python -B scripts/check_figure_release_v6.py` | PASS: six notebooks executed, 41 embedded figures match v6, historical seals unchanged |
| `python -B -m unittest tests.test_figure_release_v6` | 13 pass |
| `python -B -m unittest discover -s tests` | 402 run, 401 pass before the one-line test update; the updated test then passes with the v6 and extension suites (44 pass). The full suite was not re-run after that edit. |
| `python scripts/run_research_extension.py --output <new dir>` | PASS; 15 tables equal the published v2 tables to 2e-16; output deleted |
| Two consecutive `run_figure_release_v6.py` builds | byte-identical |
| `python -I -B scripts/check_primary_notebooks.py` | PASS |
| `python -I -B scripts/check_manuscript_notebook.py` | PASS |
| `python -I -B scripts/check_research_extension.py` | FAILS by design: it pins notebook 06 to its generator and to the v2 extension images |
| `python -I -B scripts/check_deep_review.py` | Not validly run: it stopped on the roster line-ending hazard below. After restoring the roster it would still fail by design at the step that requires notebooks 01-05 to embed v3/v5 images |
| `python -I -B scripts/check_publication_notebooks.py` | Failed before this work as well (expects v2 images) |
| `python -I -B scripts/check_heldout_notebook.py` | Not validly run: stopped on the roster hazard. Re-run after `git checkout text-files/revision224_adults.json` |

The historical checkers were left untouched on purpose. Their seal checks are reused inside
`analysis_figures/v6/data.py::load`; only their "notebook embeds the old picture" step is superseded.

## Open risks and what was not done

1. The prose inside notebooks 01-06 and older docs (`PUBLICATION_FIGURES_V2.md`,
   `FIGURE_REVIEW_RESPONSE_V3.md`, `DEEP_REVIEW_RESPONSE.md`, `HELDOUT_REAL_WORLD_EVALUATION.md`,
   `PRIMARY_ANALYSIS_FINDINGS.md`) still describes the old figure designs in places. Only a banner
   and the top-level pointers were updated.
2. Figures were inspected as rendered PNGs at page width. SVG and PDF were produced by the same
   call and checked for presence and determinism, not opened visually.
3. `analysis/README.md` and the brief still embed older preview images below the new note.
4. Notebook files grew (04 is 6.4 MB) because 300 dpi PNGs are embedded.
5. The viewer and gallery (`viewer/`, `NETWORKS.md`) were not touched.
6. The conditional mixing diagnostic in README finding 4 is a reviewer check and is not saved as a
   table.

## Next actions for Codex

1. Restore the roster after any full test run, find the test that rewrites it, and make it write to
   a temporary directory.
2. Decide whether to retire or re-point `check_research_extension.py`, `check_deep_review.py` and
   `check_publication_notebooks.py`: either keep them as historical validators that are expected to
   fail on notebook embedding, or split their seal checks from their embedding checks.
3. Regenerate notebook 06 through its generator in `scripts/check_research_extension.py` if that
   generator stays authoritative (its four display cells must call
   `analysis_figures.v6.release.show`).
4. Bring notebook and document prose in line with the v6 captions, starting with the sign-flip and
   density points in `docs/figure_redesign/README.md` section 1.
5. If the paper keeps an efficiency comparison, add a pre-specified density-matched baseline in a new
   design rather than reading the fixed baselines as competitors.
6. For RQ2, a shuffled-demographics or counterbalanced prompt-order collection is the only way to
   separate attribute from field order; that needs a new, approved design.
