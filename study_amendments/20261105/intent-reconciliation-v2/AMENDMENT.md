# Single pre-transport intent reconciliation

User authorized diagnosis, repair, bounded review, full offline checks and safe
paid continuation on Nov 4, 2026 Phoenix. Frozen source and contract stay intact.

Job 763678 rolled back its reservation commit before transport. Job 788115
stopped offline when accounting found exactly one extra request journal.
Protected-copy diagnosis found no missing journals and one extra intent:
`e6a61184019d126aa703e8177f68581b11b8746c8e294a1eec41f0c80d31a587`.
Its bytes hash to
`a1ec59a2a52346a2972e751cb33f2f144a7734012ab83a5e6e3bc735be9a7625`;
it has no ledger row or transport receipt. This alone is not permission to send:
actual cached-prefix replay must reproduce the exact original ID/ordinal first.

One n6106 allocation tests and verifies a fresh checkpoint copy with keys unset
and network blocked. Only the exact pinned intent may be archived, byte-for-byte,
under `pretransport_rollbacks/` on that copy. No other file, ledger row, cost,
reservation, saved receipt or request identity changes. Unchanged accounting,
751 terminal replays (746 valid, 5 failed), and cached-prefix verification must
pass. After exact source preservation checks under both existing locks, apply
the same archive operation to the stopped live checkpoint and recheck it before
publishing job-specific proof. No inventory filter or accounting guard bypass.
Immediately before the unchanged paid runner, reacquire the workflow lock and
compare the live evidence manifest, full logical ledger state, authorization and
schema with that exact verified snapshot. Any difference refuses paid launch.

The original v2 paid runner and original $120 cumulative cap remain unchanged;
the existing billing ordinal-44 exception remains the only replacement identity.
No extra API retry or broad reconciliation is permitted. Prices retain their
reviewed Nov 2-8 UTC validity. Submission intent prevents duplicate submission.
Abnormal exits stop; only an orderly checkpoint can use the existing v2 child.

Local tests are not cluster verification. The job may spend its allocation on
verification; no fixed start or completion time is promised. File-only monitoring
must never query the active SQLite database.
