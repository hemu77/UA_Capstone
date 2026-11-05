# Billing interruption execution amendment, 2026-11-02

Nov 3 verification correction: copy job 686844 failed before any live handoff.
Its original-row digest pin had one mistyped hexadecimal digit. Revision 2 pins
the digest from the previously archived request journal and rechecks both live
rows before copying, then again on the fresh backup. Revision 1 and its failed
checkpoint remain intact. The special transport identity, paid code path,
accounting rules, price window, and full replay criteria are unchanged.

Owner confirmed restoration of project credits and requested a safe resume. This
amendment is operational, not a change to the study design. Never edit the frozen
package, re-anchor the budget, change providers, or replace a saved response.
Both rejected requests remain as-is, including their full reservations. One new
request in their explicit lineage is allowed, only after validating the exact
original prompt and replaying ordinals 0-43. No further exception is authorized.

Standard input/output USD per million tokens, rechecked November 2:
- https://developers.openai.com/api/docs/models/gpt-4.1 : 2 / 8
- https://developers.openai.com/api/docs/models/gpt-5.6-luna : 0.2 / 1.2
- https://developers.openai.com/api/docs/models/gpt-6-luna : 0.1 / 0.5
- https://developers.openai.com/api/docs/models/gpt-6-sol : 2 / 10

Apply only to the frozen direct API, default tier and short text requests.
Retain the original 100,000 input bound and 1.25 input multiplier. A local
seven-calendar-day review window, Nov 2 through Nov 8 UTC inclusive, is a
stop date, not a promise that prices cannot change. Stop earlier on pricing,
availability or billing discrepancies. No automatic renewal. All cost checks
use the original $120 ledger, not a new budget for the remainder.

A versioned out-of-tree runner is built only from the pinned original runner.
The operational exception is reported separately and its new request carries
an amendment identifier. The original runner alone cannot verify that new
request; retain this bundle and hashes for final replay. An unchanged fixture or
SQL benchmark is not evidence of successful production collection or speedup.
Before live collection: independent review, exclusive locks, durable copy,
matching accounting and hashes, exact offline replay of all terminal records
and saved prefix, first-missing-request identity, and surviving query indexes.
Abort on any mismatch; never roll back a database over later paid receipts.
