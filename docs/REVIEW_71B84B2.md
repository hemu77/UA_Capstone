# Review Of Raghav's Pre-Run Revision

Reviewed commit: `71b84b2bea6877ab6045b51e8dd50479d4ccb0fa`.
Scope: execution order, partial-study inference, receipt recovery and offline
reporting. No main generation or model API spending was authorized by this review.

## Shared Protocol

Raghav's commit and this follow-up keep the same study:

- Four fixed models, four methods, seven named country/instruction settings,
  the same 50 fictional adults and eight schedule blocks: **896 planned attempts**.
- **$120 additional main-study ceiling**, not a promise that every attempt fits.
  The historical usage estimate remains approximately $102.44 and 26.3 serial API
  hours. This code-only repair adds no model calls.
- Complete one 112-cell schedule block before starting the next. Shuffle cells
  within each block. This improves useful coverage after an operational stop.
- Keep all four models in every paired contrast. Use the unchanged complete-data
  calculation with eight complete blocks; retain the predeclared six/seven-block
  fallback, its PARTIAL label, all 24 hypotheses and all-eight-block missingness bounds.
- Missingness must be unrelated to the contrast outcome for the fallback's
  inference to be defensible. Predeclaring it does not prove that assumption.
  Do not stop collection based on interim p-values.
- Keep valid empty graphs and invalid responses. Do not resample unwanted
  outcomes or silently substitute models.
- Preserve prompts, parser, rates, generation settings and all historical paid
  evidence. Historical calibration does not need regeneration for these fixes.

Pairing removes a shared additive block effect. It does **not** remove arbitrary
provider drift, model updates or condition-specific changes within a block.

## Defects Reproduced And Repaired

| Finding in the reviewed commit | Consequence | Follow-up |
| --- | --- | --- |
| **P1: Offline `analyze` created the main-stage directory before collection.** | A harmless inspection then made `authorize` refuse with `Prior stage evidence exists; do not create a fresh budget anchor.` It also blocked freeze migration. | No existing stage returns `NOT_STARTED` without creating it. Test confirms a subsequent authorization still works. |
| **P2: Missing terminal receipts were called "unattempted."** | A stopped cell can have many paid replies but no terminal graph. Public reports understated work and conflated missing outcomes with no API use. | Report `terminal_results`, `without_terminal_result`, terminal failures and valid-but-undefined endpoints separately. Missing execution status is unknown; only the private ledger establishes request attempts/charges. |
| **P2: An unsuccessful re-analysis left old successful exports visible.** | After receipt/source corruption, old COMPLETE status and old p-values could still appear current. CSV failure could also occur after success was written. | Invalidate existing derived reports before verification; publish success only after all exports finish. Paid receipts and graph artifacts remain unchanged. |
| **P2: A journal with `row_sha256: null` matched a nonexistent pre-parse digest.** | Recovery accepted a checkpoint that was neither an intent nor a real historical hash, contrary to its strict repair policy. | Require an actual pre-parse digest; unknown/null checkpoints are refused without rewriting or sending another request. |
| **P2: Reconciliation logs were checked only for a received request ID.** | A changed repair hash, run, ordinal or previous checkpoint could remain in the audit log while launch continued. A bounded read-only AI reviewer reproduced this; no duplicate API request was observed. | Validate every log entry against the received ledger/transport evidence, distinguish pre-parse from post-parse hashes, and reject duplicate/unknown repair provenance before continuing. |

The tests first failed for these defects and then passed after the fixes.
Duplicate terminal records are also rejected rather than silently overwritten.
The reporting contract is versioned separately; historical results are not relabeled.
Recovery tests interrupt both before and after log/journal writes, repeat the
interruption and confirm unchanged ledger rows plus exactly one mocked paid POST.

## Review Decision

The allocation and intended recovery/statistical rules are aligned with Raghav's
revision. The defects above required changes, not a new research design.

Raghav has **not yet confirmed this follow-up**. His authored commit is evidence
of the protocol he implemented, not approval of subsequent edits. The PR comment
requests confirmation of the exact follow-up and the shared terms above.
Professor approval and the owner's eventual launch instruction are separate.

Before eventual launch, use the current verification report, a locally configured
credential, an unsynced workspace and the existing price-validity gate. The
October 18 price check expires after October 25; it must cover the **whole run**.
No API key belongs in this document or the repository.

Tests establish specific engineering behavior. They cannot guarantee the
missingness assumptions, adequate power, research validity or conference acceptance.
