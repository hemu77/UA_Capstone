# Hindi Alternative-2 Candidate

Status: offline-tested, AI-authored candidate; its separately approved live probe
has not run yet. No human bilingual review is claimed. The frozen V8 experiment
is unchanged.

## Observed Failure

The current 96-cell wording screen retained two terminal GPT-4.1 Hindi
`alternative_2` failures (cells 053 and 058). Both exhausted the existing local
correction limit. One saved response was `6,29` followed by `6,41` on another
line, when persona 30 had been asked to nominate exactly two people.

The strict parser correctly rejected it: the response was not a single list
of two unique nominees. Flattening lines, taking the first pair, or removing
duplicates would invent a choice the model did not unambiguously make.
Serialization ambiguity is a repair hypothesis, not an established causal diagnosis.

## Candidate Change

`revision_hindi_candidate.py` retains the original Hindi context, selection
count and exclusions. It appends the same clarification to initial and retry
instructions. Its English meaning is:

> Return IDs of selected people, not friendship pairs. The entire response
> must occupy one line, with no new line. For one selected person, return only
> that person's ID; for multiple people, return their distinct IDs on the same
> line, separated by commas.

There are no example persona IDs to anchor choices. Country framing, personas,
candidate ordering, quotas and parser rules are unchanged. The original
requests, receipts, failure records and V7/V8 source files remain untouched.

## Validation And Use

- [x] Offline checks cover four country contexts, counts 1, 2 and 49, and retries.
- [x] Parser checks reject the observed multiline answer, duplicate IDs,
  self-links, unknown IDs and incorrect counts without mutating the graph.
- [x] A valid two-person answer still creates exactly the requested ties.
- [x] The owner authorized up to $1 on October 13 for four separate GPT-4.1
  Hindi local graphs: original versus clarified wording, two matched repetitions.
- [ ] Complete and inspect that versioned probe before any claim of improved
  model compliance; no historical failed observation is replaced.
- [ ] A future protocol must explicitly decide whether to adopt the candidate;
  it is not automatically imported by either frozen V8 stage.

Run the offline checks with:

```powershell
rtk proxy .venv/Scripts/python.exe -B -m unittest tests.test_hindi_candidate
```

The 96-cell screen must keep its original denominator and failures. The current
104-cell calibration prospectively uses `reference`, not `alternative_2`;
this candidate does not silently alter that choice or spend its allowance.
Passing these tests verifies construction and parsing, not translation accuracy
or live-provider reliability.

## Review Record

Three offline tests passed after the missing serialization clarification first
failed the rendering assertions. A bounded, read-only GPT-6.1-Sol review found
no blocking issue in the candidate, tests or disclosures. It agreed with the
English gloss, but did not independently rerun the tests or inspect the paid
failure receipts. This is AI review, not native-speaker certification.

Hash checks confirmed `revision_v7_prompts.py`, `repair_v8.py` and
`repair_v8_io.py` were unchanged while the paid screen continued.

## Separately Approved Probe

`probe_hindi_candidate.py` reuses the V8 budget, recorded-response replay,
strict parser, finite corrections and local-checkpoint I/O safeguards. Its own
stage is `hindi_candidate_probe`, not another wording/calibration allowance.
Four graphs use the full 50-person roster, GPT-4.1, US framing and Hindi.
Fresh matched seeds are 34000 and 34001; the order is original/clarified for
the first pair and clarified/original for the second. Actor order, candidate
order and quotas match within each pair. The original arm is exactly frozen
alternative 2; the clarified arm appends only the documented format instruction.

Paid execution is serialized after all 104 calibration observations have been
completed and inspected. The shared workflow lock plus an exact-summary gate
prevent a stopped/partial calibration from enabling probe approval. The original
ledger is required; no new ledger, reset allowance or replacement is permitted.
Source hashes and rendered prompts form a separate experiment identity.

```powershell
rtk proxy .venv/Scripts/python.exe -B probe_hindi_candidate.py preflight
# Exact-version review and the existing $1 owner permission are required.
rtk proxy .venv/Scripts/python.exe -B probe_hindi_candidate.py approve
rtk proxy .venv/Scripts/python.exe -B probe_hindi_candidate.py execute
rtk proxy .venv/Scripts/python.exe -B probe_hindi_candidate.py verify
rtk proxy .venv/Scripts/python.exe -B probe_hindi_candidate.py inspect
```

Two repetitions only support a small operational check. Even four successful
graphs would not prove the clarification caused improved compliance or establish
language equivalence. Main-study wording will not change automatically.

Before payment, the reviewer caught a missing completed-calibration prerequisite.
The targeted test failed before the guard, then passed after it. The final suite
passed **184 offline tests in 461.988 seconds**, with network connections blocked;
it includes rejection of missing, partial and stale calibration clearance,
tampered allowances, replay without additional requests, and retained failures.
