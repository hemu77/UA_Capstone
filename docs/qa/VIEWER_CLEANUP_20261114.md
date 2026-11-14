# Current Study And 896-Network Browser

## Changes

The current Three.js lab remains the entry point. Historical datasets and the
old interface are repository-only under `viewer/archive/`; no failed case was
deleted. Root README, research file map and dated history are separated.

The PNG gallery is replaced by an isolated final-graph 3D browser with four
method filters. It has 896 selectable networks: 890 primary and six separately
labeled supplements. Each graph uses its saved adjacency, never inferred PNG
edges or a merge across runs. Primary replay and statistics exclude supplements.
All 50 nodes are retained in 12 empty primary and one empty supplemental graph.
Original PNGs and adjacencies remain linked in the repository.

## Verification

- 42 Node tests passed, including all 890 replay/final parity checks.
- 12 focused Python tests passed for archived consumers and adjacency parsing.
- All 896 final adjacency source hashes verified during offline export.
- Build verifier passed: 890 primary graphs, 896 attempts, 896 final gallery
  graphs, six separate supplements, zero archived deployment payloads.
- 435 relocated files match baseline Git blobs byte-for-byte; all 69 frozen
  source pins match the original contract. Frozen exporters were not modified.
- Edge browser checks passed for primary filters/coverage, empty graphs,
  downloads, legacy-link rejection, compatibility routes, delayed/error replay,
  pointer rotation/pan/zoom, keyboard access and mobile layout.
- New 3D browser checks passed: four groups of 224 descriptive graphs, real
  neighbor counts, canvas rotation/zoom, primary-state isolation, supplements,
  empty graphs, pagination, mobile, catalog retry, checksum refusal, WebGL
  fallback and persisted-page lifecycle handling.

Three bounded read-only agent reviews covered integrity, interface and docs.
Review caught a persisted-page canvas lifecycle bug and sealed exporter edits;
both were fixed before release. These are engineering reviews, not peer review.

## Limits

No paid model calls, ledger access, new generation or statistical recomputation.
No physical-device, Safari or Firefox validation. Back/Forward cache suspension
was checked with page-transition events, not every browser's actual cache policy.
The historical interface is a source archive, not a maintained second app.

Deployment provenance is recorded in [the current handoff](../SIMULATOR_DEPLOYMENT.md)
after publication.
