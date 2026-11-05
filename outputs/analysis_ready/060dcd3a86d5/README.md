# Analysis-Ready Handoff

Prepared offline; no generation, credentials, ledger queries or new inference.

| Table | Rows | Use |
| --- | ---: | --- |
| primary_attempts.csv | 896 | Primary input; retain six failures as missing |
| primary_valid_graphs.csv | 890 | Original valid graph inventory, not a missingness replacement |
| supplemental_attempts.csv | 9 | All extra draws, including three failures |
| supplemental_valid_graphs.csv | 6 | Separate post-collection graphs |
| all_valid_graphs_descriptive_only.csv | 896 | Descriptive/visualization inventory, NOT primary inference |
| all_attempts.csv | 905 | Full provenance catalog |
| failures.csv / failures.json | 9 | All retained failures |
| selected_supplement_map.csv | 6 | Original failure to selected extra graph |

Graph/PNG/JSON files are unpacked under outputs/final_study/060dcd3a86d5/main/runs/, outputs/supplement_20261105_v1/runs/, and outputs/supplement_completion_20261105_v1/runs/.
Do not move those evidence files into legacy text-files/ or plots/ directories.

Start with analysis/00_data_readiness.ipynb using the project .venv kernel.
Existing primary analysis_rows.json, network_metrics.csv, contrasts and missingness reports remain in the original main directory, byte-for-byte unchanged.
Their 21 complete, two partial and one missing contrasts still need interpretation and assumption review.
Do not run the frozen collector CLI as a setup command: its original replay does not recognize the separately documented billing exception. No new statistical results were calculated here.

readiness.json seals these generated tables by hash. Re-run scripts/prepare_study_data.py --archives <verified-release-folder> for a complete installation and integrity check; different existing evidence is refused, never overwritten.
