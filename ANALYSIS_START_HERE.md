# Analysis Starts Here

Canonical workspace: `C:\Users\Hemu\Projects\UA_Captsone_SocN`.
Use this workspace, not the historical OneDrive checkout.

[Browse all 896 saved network files and plots](NETWORKS.md): 890 primary plus
six separate supplements. Each entry links to its original ADJ and PNG.

Open [the readiness notebook](analysis/00_data_readiness.ipynb) with the
**UA Capstone (frozen Python 3.11)** kernel. It validates and loads prepared
inputs only; it does not run statistical inference or make model calls.

## Statistical Analysis And Evaluations

Open [primary statistics](analysis/01_primary_statistical_analysis.ipynb),
[primary comparison plots](analysis/02_primary_comparison_plots.ipynb), and
[held-out public-data evaluation](analysis/03_heldout_public_evaluation.ipynb).
All three now embed the revised figures directly. [Notebook 04](analysis/04_manuscript_figures.ipynb)
collects all 13 figures, including the largest-component appendix.
The [analysis gallery](analysis/README.md) shows previews without opening a notebook.
The notebooks verify the
separate [evaluation manifest](outputs/primary_analysis/20261105/evaluation_manifest.json)
and do not query a ledger or call models. The original collection outputs stay
unchanged. Read [findings and limitations](docs/PRIMARY_ANALYSIS_FINDINGS.md) and
[the five-paper review](docs/PRIMARY_LITERATURE_REVIEW.md) before interpreting plots.

All primary tests exclude the six selectively completed supplementary graphs.
The real-friendship-network and synthetic-reference comparisons are descriptive,
post-collection checks, not new confirmatory tests or proof of human realism.

## Held-Out Public Data And Revised Figures

The [new held-out evaluation](docs/HELDOUT_REAL_WORLD_EVALUATION.md) uses two
public friendship studies, development-only ER/BA/WS settings, equal-size
50-node draws, explicit missingness weights and separate target-conditioned
controls. It does not pretend that these data were absent from LLM pretraining
or that overlapping subgraphs are independent human samples. No paid calls.

The [revision 2 figure notes and captions](docs/PUBLICATION_FIGURES_V2.md) document
the actual figures inspected in all five supplied papers. Compact dot/interval
panels and distribution curves replace the dense heatmaps; original numerical
results and the first two figure releases remain unchanged. The
[revision 3 review response](docs/FIGURE_REVIEW_RESPONSE_V3.md) explains the
new record-level ranges, matched-size diagnostic and main/appendix split.
Current PNG/SVG/PDF files are in `outputs/publication_figures/20261105_v3/`. Use the
[updated Claude methods-and-figure review prompt](docs/CLAUDE_HELDOUT_AND_FIGURE_REVIEW_PROMPT.md).

Execute all four current notebooks from the published tables and figures:

```powershell
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\review_publication.py check
```

This checks exact embedded PNG hashes as well as old analysis/code/source pins.
To edit and preview plots, use `analysis_figures/publication_v3.py` and run
`scripts/review_publication.py build --cache <public-cache> --dest <unused-directory>`
from the pinned Python environment. The build refuses an existing destination.
A later published revision needs a new version, never silently changed bytes.

Full benchmark reproduction after the environment setup below (not needed for
figure or notebook inspection):

```powershell
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\fetch_heldout_public.py --cache data_external\heldout_20261105
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\run_heldout_evaluation.py --cache data_external\heldout_20261105
```

All four current notebooks can rerun from the published tables without downloading
raw public data. Raw public caches remain outside Git, separate from all
original/supplementary study outputs.

Full offline regeneration after installing the pinned release archives:

For a new environment, first use the repository's `scripts/setup_env.ps1` with
Python 3.11.9, then register that checkout's environment as the notebook kernel:

```powershell
rtk proxy .\.venv\Scripts\python.exe -m ipykernel install --user --name ua-capstone-final060d --display-name "UA Capstone (frozen Python 3.11)"
```

```powershell
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\run_primary_evaluation.py
```

The current notebook checker above executes all four notebooks with the registered
frozen kernel and checks that sealed analysis outputs remain unchanged. A clean GitHub
clone can rerun the notebooks from the published tables without downloading
the large response archive; full resource recomputation needs that archive.

## Prepared Catalogs

- [Primary attempts](outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv):
  896 original observations, including six failures. This is the primary input.
- [All 896 valid graphs](outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv):
  890 original plus six supplementary graphs. Descriptive/visualization use only,
  not a replacement primary sample.
- [All nine failures](outputs/analysis_ready/060dcd3a86d5/failures.csv) and
  [all 905 attempts](outputs/analysis_ready/060dcd3a86d5/all_attempts.csv).
- [Supplement mapping](outputs/analysis_ready/060dcd3a86d5/selected_supplement_map.csv):
  each original failure linked to its selected extra graph; every failed draw stays recorded.
- [Dictionary](outputs/analysis_ready/060dcd3a86d5/DATA_DICTIONARY.md) and
  [integrity/readiness receipt](outputs/analysis_ready/060dcd3a86d5/readiness.json).

Each CSV row supplies project-relative paths and SHA-256 hashes for its JSON,
adjacency and PNG files. Files remain in their intended collection directories:

```text
outputs/final_study/060dcd3a86d5/main/runs/       original: 896 attempts, 890 graphs
outputs/supplement_20261105_v1/runs/            first supplement: 6 attempts, 4 graphs
outputs/supplement_completion_20261105_v1/runs/ completion: 3 attempts, 2 graphs
study_amendments/20261105/                     execution provenance and manifests
outputs/analysis_ready/060dcd3a86d5/            separated analysis catalogs
```

Do not move final-study data into legacy `text-files/`, `plots/`, or older
calibration folders. Failed records remain in their original `runs/` folders;
the failure index gives direct paths rather than relocating evidence.

## Before Interpretation

Retain primary failures and undefined metrics as missing, never zero-fill them.
The 12 valid empty primary graphs are real outcomes, not failed attempts.
The six supplementary graphs were selectively completed and must stay outside
primary inference. Do not claim 896/896 primary success.

Existing [primary analysis rows](outputs/final_study/060dcd3a86d5/main/analysis_rows.json),
[metrics](outputs/final_study/060dcd3a86d5/main/network_metrics.csv),
[contrasts](outputs/final_study/060dcd3a86d5/main/primary_contrasts.csv) and
[missingness](outputs/final_study/060dcd3a86d5/main/missingness_report.json) remain
unchanged. They still require method/assumption and interpretation review.
The recorded analysis has 21 complete-block, two partial-block and one missing
contrast; these labels do not establish significance, power or causal validity.

Do not run a collection/resume command for analysis setup. The original frozen
runner alone cannot replay the separately documented billing exception. This
preparation does not rerun or overwrite existing inferential outputs.

To recheck installation and rebuild identical catalogs without paid calls:

```powershell
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\prepare_study_data.py --archives C:\Users\Hemu\Projects\hoffman2-access\public-release-20261105
```

The installer checks pinned ZIP/manifest hashes, refuses changed existing files,
and copies missing files only. No credentials, ledger queries or new generation.
