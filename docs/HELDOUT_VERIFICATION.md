# Held-Out Benchmark And Figure Verification

November 5, 2026. **Share with the documented scientific caveats.** This is
scoped engineering/numeric verification, not independent human peer review.

- **59 scoped tests passed** from the clean publication checkout: 37 existing
  primary/readiness checks, three network-catalog checks, 11 held-out checks and
  eight manuscript-figure checks. No historical collector suite was launched.
- A fresh empty-cache fetch retrieved both official public files over HTTPS
  and matched the pinned SHA-256 values. Download limits, redirect restrictions,
  gzip bounds and archive path/link refusal have focused checks. No tar member
  is extracted to disk; raw human-data caches and paper-reference PDFs are not
  published in Git.
- The complete fixed benchmark ran: two source studies, 11 observed source
  records, 352 induced 50-node draws, 1,344 development-only baseline graphs and
  5,632 conditional controls. No conditional controls failed in this run. Source
  counts, overlap and survey exclusions are reported, not hidden by pooling.
- Notebook 03 independently integrates weighted ECDF differences for **all
  2,640 W1 distances**, agreeing within 1e-12. The read-only reviewer separately
  checked a failure-affected 53/56 group, agreeing to 2.22e-16. Development fitting,
  seven-condition weights, empty-graph rules and all-or-missing conditional
  aggregation were reviewed. No held-out parameter fitting was found.
- Three new primary manuscript figures provide nine PNG/SVG/PDF files. Four
  new held-out figures provide 12 files. All seven PNGs were visually inspected;
  deterministic SVG and figure-cleanup tests pass. PDF/SVG exports were not
  separately rendered in a second application. Actual reference figure pages
  from all five professor-supplied papers were inspected; no source graphic was
  copied into the project.
- Notebooks 03 and 04 executed top-to-bottom without errors in both workspaces.
  Published-table reproduction does not need the raw source downloads. Helpers
  verify artifact/code hashes before/after, including previous primary code,
  preserved original artifacts and all 69 frozen source pins. A prospective
  dependency-hash guard gap was fixed and regression-tested before publication.
- All previous primary analysis artifacts, code, source pins and the original
  primary seal remain unchanged. The gallery contains **896 saved networks**,
  explicitly separated into **890 original primary + six supplements**, with
  direct ADJ and original PNG links. Those supplements enter zero benchmark rows.

The same registered Python 3.11.9 environment was used; this is not a fresh
environment-installation test. No paid API/model calls, live-ledger queries,
credential access, scheduler changes or new primary generation were performed.
The public source downloads are not proof of pretraining novelty. Two legacy
convenience studies, dependent node draws, overlapping egonets and incomplete
survey coverage still limit scientific generalization. No full repository-wide
security audit or prospective matched-human behavioral study is claimed.

See [methods/results](HELDOUT_REAL_WORLD_EVALUATION.md),
[figure captions](MANUSCRIPT_FIGURE_CAPTIONS.md),
[visual references](VISUAL_REFERENCE_NOTES.md), and the
[Claude review prompt](CLAUDE_HELDOUT_AND_FIGURE_REVIEW_PROMPT.md).
