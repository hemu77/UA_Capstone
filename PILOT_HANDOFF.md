# Engineering pilot handoff / September 30, 2026

## Decision

The pilot is useful as a teammate engineering handoff, not a completed research
validation. The last saved budget summary records **$2.018820** in conservative
charges/reservations under the **$5 pilot ceiling**. That is not a verified invoice.
The GPT-5.6-Luna extension generated **four new graphs with 448 requests** for
**$0.1326314**, using the existing $5 ledger. Total: **28 verified pilots**.
This is an engineering extension, not a completed confirmatory experiment.

Do not launch the $50 study yet. Ask Astra to review the protocol, population,
translation equivalence, model access and sample-size justification first. A
second model's review is an additional check, not a guarantee of correctness.

## Verified data flow

```text
paid_study.py generation graph
  -> outputs/revision_budget_v1/<run>.adj
  -> outputs/revision_budget_v1/<run>.png
  -> outputs/revision_budget_v1/<run>.json (receipt + graph-change events)
  -> verify_completed + export_research_viewer.py
  -> viewer/public/data/networks.json + verified PNG copies
  -> viewer/build.mjs -> viewer/dist
  -> browser graph, replay, RQ measurements and original-image link
```

Existing `text-files`, `plots`, `stats` and `outputs` sources remain in place.
The browser does not infer relationships from PNG pixels. Its force layout is a
display choice; positions are not geography, social distance, or measured time.

- [x] Exact run-ID coverage verified; duplicates and missing/reclassified pilots fail.
- [x] 204 exported graphs match source adjacency, topology and homophily values.
- [x] All 50 displayed persona attribute records match the hashed source roster.
- [x] 28 pilot receipts pass saved hashes, graph checks and final replay parity.
- [x] 28 published PNG copies match their receipt hashes, including build copies.
- [x] Clean manuscript panels are separate from original evidence: 28 PNG/SVG
  pairs, captions outside the images, exact source/URL/hash checks.
- [x] 193 historical source hashes remain unchanged; 16 invalid graphs stay excluded.
- [x] Node test suite covers edge semantics, event replay, matching, missing values,
  filtering and source-version separation.
- [x] Browser checks cover whole-network/persona scope, remove/re-add, attribute
  colors, global one-batch replay, iterative removal and historical replay disabled.
- [x] Phone-width page has no horizontal document overflow; console checks clean.
- [ ] Historical PNG-to-graph equivalence remains unknown.
- [ ] Independent response-to-event reconciliation is not performed by this gate.
- [ ] No claim that all possible defects have been eliminated.

Machine-readable evidence: `outputs/qa/viewer_source_verification.json`.

## What the RQs currently support

| Question | Available inspection | Limit before research claims |
| --- | --- | --- |
| RQ1: country framing, English fixed | Archived US/India/Japan/Brazil conditions and recomputed graph descriptors | New pilots are US-only; one US-structured roster is not four national populations |
| RQ2: demographic mixing | Categorical Coleman scores plus separately labeled numeric age assortativity | Marginal mixing does not establish a dominant cause; attributes correlate and the roster contains minors |
| RQ3: model consistency | Individual final metrics and exact edge overlap between saved runs | Pilot source variants differ; single seeds and unrecorded versions limit attribution |
| RQ4: instruction language, country fixed | Luna pilots in English/Hindi/Japanese/Portuguese; historical language results kept separate | Translation review and replication missing; historical instructions also changed participant language |

All four generation methods remain available. No synthetic run has been invented
to fill an unavailable country, model, language or seed cell.

## Teammate check before approving more spend

- [ ] Reproduce the commands below without an API key.
- [ ] Inspect a global batch, sequential persona journey and iterative removal.
- [ ] Compare receipt, adjacency, event deltas, linked PNG and recomputed values.
- [ ] Independently inspect API request/response provenance locally without
  publishing the private ledger or credentials.
- [ ] Freeze a revised adult-roster protocol, report its population limitations,
  review translations, and decide how political labels are handled.
- [ ] Obtain Astra's bounded scientific review; document any unresolved findings.
- [ ] Agree the actual model panel, replication count, stopping rules and cost
  estimate under the $50 cap before paid collection.

## Reproduce without spending

From the repository root in PowerShell:

```powershell
rtk proxy .\.venv\Scripts\python.exe -B export_research_viewer.py
rtk proxy .\.venv\Scripts\python.exe -B render_pilot_figures.py
rtk npm --prefix viewer test
rtk npm --prefix viewer run build
rtk proxy .\.venv\Scripts\python.exe -B -m unittest test_viewer_sources -q
rtk proxy .\.venv\Scripts\python.exe -B verify_viewer_sources.py
rtk proxy .\.venv\Scripts\python.exe -B -m http.server 8765 --bind 127.0.0.1 --directory viewer/dist
```

Open `http://127.0.0.1:8765/layers.html`. Serve only `viewer/dist`, never the repo
root. The viewer needs neither credentials nor access to the private API ledger.
