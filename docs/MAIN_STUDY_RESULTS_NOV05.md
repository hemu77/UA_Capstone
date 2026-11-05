# Main Study Results: November 5, 2026

## Original Collection

| Measure | Verified result |
| --- | ---: |
| Planned and recorded terminal attempts | 896 / 896 |
| Valid saved graph records | 890 |
| Failed terminal attempts | 6 |
| Valid empty graphs, included in 890 | 12 |
| Saved provider-response receipts | 99,632 |
| Conservative charges plus retained reservations | $101.98695215 |
| Cumulative ceiling, including the separate supplement | $120 |

Hoffman2 job 814020 completed with exit 0. Final scheduler/log evidence was
checked November 5 at 10:16-10:17 a.m. America/Phoenix. The downloaded private
archive SHA-256 is
`9c7e01c6ebad0430b12a93c78f2bdfb2ba0cb18f5dce88c2554a391827d277e7`.
This is a private full-backup hash, not the public release archive hash.

The independent local audit reconciles the downloaded ledger, journals,
transport receipts and terminal records; verifies every graph/PNG hash, roster,
density and PNG integrity; and replays all six failures offline. All terminal
graph replays were performed in the completed cluster workflow, not repeated
as part of the local artifact audit. See the [machine-readable audit](../study_amendments/20261105/local-audit.json).
No live ledger queries or new Slurm submissions were needed for retrieval.

Five failures are `invalid_global_response`: GPT-5.6-luna US Portuguese seeds
44000 and 44004, GPT-5.6-luna US Japanese seed 44004, GPT-6-luna US Japanese
seed 44002, and GPT-6-sol US Japanese seed 44005. The sixth is
`local_corrections_exhausted`: iterative GPT-4.1 US Japanese seed 44006.
Their original records remain in the dataset. Failed attempts are not graphs.

## Separate Supplement

The owner authorized one new local observation for each of these six cells,
after offline checks and one independent read-only review. New identities
explicitly link to the original failures. Prompts, settings and seeds remain
frozen, but a new stochastic response is not a repair of the original result.
There is no repeat-until-success policy. Uncertain transport stops rather than
silently purchasing a replacement. Original spending and reservations remain
in the cumulative $120 accounting anchor.

Supplemental outcomes will be recorded here after completion. They are not
included in the original 896-attempt primary analysis. Even six new valid graphs
would mean **896 valid graphs across 902 attempts**, not 896/896 primary success.

## Data and Interpretation

[Download all public results](https://github.com/hemu77/UA_Capstone/releases/tag/main-study-20261105).
`main-study-20261105-public.zip` contains all 896 original terminal JSON records,
890 adjacency files, 890 PNGs, analysis tables and reviewed execution provenance.
The [original manifest](../study_amendments/20261105/original-manifest.json) records
the archive hash and every included file's byte count and SHA-256. Extract into
an isolated checkout using the stored repository-relative paths; never overwrite
an active collection workspace. Private ledgers, credentials, authorization
and raw transport/request journals are deliberately not published.

The repo keeps all adjacency files and analysis tables directly under
`outputs/final_study/060dcd3a86d5/main/`; the larger response records and PNGs are
in the release archive to avoid adding over 2 GB of uncompressed records to Git.
Original frozen `final_study/runner.py` alone cannot replay the exceptional
billing request. The exact [reviewed billing adapter](../study_amendments/20261105/billing-v2/AMENDMENT.md)
is included separately; do not modify/refreeze original source to hide this.
These operational scripts are historical provenance, not a new launch command.

The original analysis reports 21 `COMPLETE_ASSUMPTIONS_REQUIRED` contrasts,
two `PARTIAL_BLOCKS_ASSUMPTIONS_REQUIRED`, and one `MISSING_NO_INFERENCE`.
Review missingness, block coverage, model/method effects and assumptions before
drawing conclusions. Different generation methods use different compute budgets;
this is not an equal-cost superiority experiment. Collection completion does
not establish conference readiness, statistical power or causal generality.
