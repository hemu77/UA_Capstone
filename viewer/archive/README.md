# Historical Viewer Archive

Repository-only evidence, **not part of the deployed demo**.

| Folder / file | Contents |
| --- | --- |
| `index.html`, `main.js`, `style.css` | Earlier historical/pilot interface source |
| `public/data/networks.json`, `artifacts/` | Historical network and engineering-pilot exports |
| `public/data/calibration.json`, `calibration/` | V5 calibration exports |
| `public/data/revised-calibration.json`, `revised-calibration/` | V6 calibration exports |
| `public/data/presentation.json`, `presentation/` | Earlier pilot presentation figures |

These files were relocated without changing their bytes. They are not primary
results and must not be pooled with the final study. Original evidence in
`outputs/`, `stats/` and frozen source paths was not moved.

This is a source/evidence archive, not a second runnable application. Original
relative imports are preserved; opening the historical HTML directly is not a
supported demo. Use the [current lab](https://ua-network-observatory.handm01042024.chatgpt.site/layers.html)
and [research file map](../../docs/research/README.md) instead.

Historical exporters and `verify_viewer_sources.py` are frozen source. Their
original paths and bytes are preserved; do not invoke them to rebuild this archive
in the current checkout. Use their original baseline checkout for historical
reproduction, never a paid collector.

Run `python scripts/verify_viewer_archive.py` from the repository root to check
every relocated file against baseline Git blobs and every frozen source hash.
Shared graph/replay tests still validate archived fixtures. The production
verifier independently proves these archived payloads are not deployed.
