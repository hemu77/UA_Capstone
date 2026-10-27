# Hoffman2 Execution

This deployment moves the reviewed collector to a CPU batch job. It does not
run GPT models on a GPU, introduce parallel API calls, or change the study.
The allocation remains 896 attempts, eight repetitions per condition, and one
$120 main-study budget shared across every continuation.

## What Changed

Contract v6.3 adds an optional scheduler deadline. `launch --walltime-seconds`
checks the same monotonic deadline before each new cell and before each new
request intent. A received response finishes settlement and checkpointing before
a pause is considered. Recovery does not restart the job clock. A planned pause
returns `PAUSED_AT_CHECKPOINT` and exit 75; partial cells have no invented terminal
result. Resume replays their saved replies without purchasing them again.

`scripts/hoffman2/run.sbatch` is included in the frozen source hashes. It requests
24 hours and gives the collector 23 hours, leaving an hour for checkpointing,
archiving, and scheduling. Only exit 75 submits a continuation, with an `afterok`
dependency so it cannot overlap its parent. Other failures stop for inspection.
Slurm's generic automatic requeue is explicitly disabled; only the controlled
checkpoint continuation is allowed to schedule the next collector job.
An additional coordinator lock protects the whole job, including its final
verification and archiving. Never launch a second local or remote collector.

This is graceful walltime handling, not a guarantee against forced termination,
hardware failure, or a filesystem hang. An uncertain provider request can still
need one conservatively charged replacement under the existing recovery policy.
No refund or exactly-once provider billing is assumed.

## Deployment Requirements

- Use the current production Slurm host and verify its SSH fingerprint against
  [UCLA's connection documentation](https://www.hoffman2.idre.ucla.edu/Using-H2/Connecting/Connecting.html).
- Confirm the account's actual allocation with `h2 whoami`. No GPUs are requested.
- Recreate Python 3.11.9 and every version in `requirements.lock.txt`; do not
  re-freeze the protocol to accept installation differences.
- Use a dedicated, owner-only workspace containing `repo/`, `env/`, `logs/`,
  `private/`, and cache directories. A Windows virtual environment cannot be copied
  to Linux. Keep caches outside an over-quota home directory.
- Transfer the exact public source bytes plus the eight historical Hindi checks'
  private authorization and SQLite accounting. Public Git alone is not sufficient.
- Pass the offline tests, 112-case preflight, private readiness, and storage probes
  on the actual compute filesystem before authorizing the first paid job.
- Verify cross-node file locking, SQLite transactions, and atomic replacement.
  Never run independent budget copies or delete a ledger to recover a failed job.
- Enter a fresh OpenAI key locally through a hidden terminal prompt. The batch
  wrapper reads `private/openai.key`, outside the repository, owned by the account
  with mode 600. Its value must never appear in Git, job arguments, or logs.

## Submit And Resume

After deployment and verification, from the login node:

```bash
umask 077
sbatch --account=<approved-account> --export=NONE \
  --chdir=<workspace>/repo --output=<workspace>/logs/main.%j.log \
  <workspace>/repo/scripts/hoffman2/run.sbatch <workspace>
```

The wrapper supplies the unchanged `--approve-usd 120` and
`--approval-reference owner-eight-repetitions-120` on every continuation.
It does not create a new budget per job. Do not use a job array, run concurrent
copies, or add a generic retry loop around a failed submission or collector.

At a planned pause, the job archives `outputs/final_study/` and the contract into
`results/checkpoint-<job-id>.tgz`. At full terminal coverage it independently
replays and analyzes the collection, then writes `results/final-verified.tgz`.
These archives contain private accounting, but not the API key; do not publish
the archives wholesale. Retrieve them to the canonical local workspace and
verify before publishing only the intended public artifacts.

Scratch is temporary storage. Download checkpoints and the final archive; do not
treat cluster scratch as a permanent backup. Never delete the source evidence
until a verified second copy exists. Keep the laptop collector stopped while the
Hoffman2 collector owns the study.

Price evidence currently expires November 1, 2026, inclusive. A later launch
requires refreshed evidence before any main attempt. Scheduler queue time is not
included in the historical 26.3-hour API estimate. Hosting the serial collector
on Hoffman2 does not itself establish a speedup.

## Verification Status

October 27: both login hosts accepted the owner's interactive authentication.
Slurm environment job 512477 completed with exit `0:0`, installing Python 3.11.9
and the pinned requirements; `pip check` passed and unauthenticated OpenAI HTTPS
returned the expected 401. This was a connectivity check, not model access.

Deployment job 516713 verified the archive/source hashes and installed a fresh
checkout. Storage job 516739 passed on two different compute nodes, checking
exclusive `flock`, SQLite writer contention and reopening, and visibility of an
atomic replacement. It does not establish power-loss durability.

A fresh local publishing checkout passed 284 offline tests with sockets blocked.
The final environment-hardening change then passed all 75 runner/wrapper tests,
including a regression proving that inherited `PYTHONOPTIMIZE` cannot disable
assertions. All eight wrapper tests subsequently passed, including disabled
scheduler requeue. Six separate deployment guard tests also passed, with malformed
archives, stale/incomplete reports, echoed-key fallback, attempted-main updates
and failed temporary-folder allocation/resolution rejected. Those local
guard fixtures use a stand-in lock; the two-node check above uses the real lock.

Compute job 516740 was stopped before paid execution because shared-storage test
fixtures were slow. Replacement job 517152 ran 286 tests with disposable fixtures
on node-local storage, but failed nine checkpoint-path checks. New compute job
518030 confirms `/tmp` resolves to `/work/tmp`; fixture paths under that alias
triggered the strict noncanonical-journal guard. The validation wrapper now
resolves its temporary root and asserts that it is canonical before testing.
The resubmission helper uses accounting history for completed jobs rather than
requiring them to remain in the live queue. No frozen collector or historical guard was
weakened, and actual study evidence/accounting remain on shared storage.
Compute job 518030 then completed with exit `0:0` in 28 minutes 59 seconds:
**286 tests passed, zero skips, 112 preflight cases, and all eight historical
Hindi observations replayed with their accounting intact**. The complete private
evidence inventory was unchanged. The report matches contract
`060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f`.
New model API calls: zero. The original nine path failures are resolved by the
full rerun, not waived. No API key was transferred with the source bundle.
Private key entry and live model-access checks remain before main submission.
The key helper requires matching successful validation reports and Slurm exit
codes, then checks access to all four models before creating an owner-only key
file outside the repository and submitting the main job once. A durable launch
intent blocks blind resubmission after an uncertain submission outcome.

The official standard input/output rates were checked again on October 27 and
still match the frozen rates: [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1),
[GPT-5.6-Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[GPT-6-Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), and
[GPT-6-Sol](https://developers.openai.com/api/docs/models/gpt-6-sol).
The existing 1.25 input-cost multiplier covers the documented cache-write
premium. The frozen November 1 expiry and $120 main ceiling were not extended.

This record does not assert that paid generation finished. Passing engineering
checks is not a guarantee of research validity, provider uptime, or acceptance.
