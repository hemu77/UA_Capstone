# Main Collection Preparation: October 25, 2026

The owner instructed us to merge PR #1 and prepare the main collection, reporting
that the professor supports proceeding and can provide GPU access. This is an
owner-reported decision, not a recorded GitHub review from the professor or
Raghav. No formal GitHub review submissions or inline review threads were present
when checked. The owner explicitly authorized merging.

## Collection And Hardware

The adopted scope remains 4 models x 4 methods x 7 settings x 8 repetitions:
**896 fresh attempts**, all with the same 50 fictional adults, **224 per method**.
The additional main-study ceiling remains **$120**, with a historical usage
forecast of **$102.44 and 26.3 serial API hours**. Allow roughly 40 hours plus local
processing and recovery; neither cost nor completion time is guaranteed.

These four models use hosted OpenAI APIs. A university GPU does not replace API
billing or accelerate the provider's inference. GPU access can support a separate
open-weight-model extension once its hardware, models and design are specified.
It is not required to start this adopted collection.

The single execution workspace is now:

`C:\Users\Hemu\Projects\UA_Captsone_SocN`

The October 20 relocation verified all 79,425 copied files byte-for-byte, including
private accounting. Local environment launchers were then updated for that path.
The publishing checkout remains a public-artifact workspace and must not launch
paid generation. Never run two execution copies.

## Price Evidence

Official model pages fetched and checked October 25, 2026. Standard text prices
per million tokens are unchanged:

| Model | Input USD | Output USD | Official source |
| --- | ---: | ---: | --- |
| GPT-4.1 | 2.00 | 8.00 | [Model page](https://developers.openai.com/api/docs/models/gpt-4.1) |
| GPT-5.6-Luna | 0.20 | 1.20 | [Model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| GPT-6-Luna | 0.10 | 0.50 | [Model page](https://developers.openai.com/api/docs/models/gpt-6-luna) |
| GPT-6-Sol | 2.00 | 10.00 | [Model page](https://developers.openai.com/api/docs/models/gpt-6-sol) |

The existing conservative accounting, including its input multiplier, is
unchanged. These are standard short-context rates, not Batch/Flex discounts or
GPU costs. GPT-4.1's January 14, 2027 shutdown was also reconfirmed in the
[official deprecations](https://developers.openai.com/api/docs/deprecations).

Contract v6.2 refreshes the internal price-evidence window to **October 25 through
November 1, inclusive**. This is our seven-day freshness policy, not a provider
price guarantee. Prices are checked before every request. Finish collection
within that window; otherwise the runner stops. A later refresh must happen
before collection starts, never by changing an active study's contract.

Only the version, source hash for protocol.py and price-check dates change from
v6.1. Prompts, parser, roster, methods, model settings, rates, analysis, order,
repetitions, spending ceiling and historical evidence stay identical. Migration
accepts the exact unexecuted v6.1 hash and refuses if its main folder exists.

## Launch State

Main generation has not started and no main authorization/ledger is created by
these preparation checks. No new model API spending occurred. A fresh local
`OPENAI_API_KEY` is still required; neither process nor Windows user environment
contained one when checked. Do not publish credentials or private accounting.

From the execution workspace, use `.venv\Scripts\python.exe` for the commands in
[Final Collection](FINAL_COLLECTION.md#run-and-inspect). Keep the machine plugged
in and awake. The plugged-in sleep setting was already disabled at relocation.
The frozen contract and private readiness must pass before the launch command.

The professor's scientific priorities remain part of the paper work: report
effect sizes and consistency, compare against suitable empirical networks and
baselines, and restrict claims to the tested instruction contracts. These
comparisons have not been completed by merging code. RQ2/RQ3 remain descriptive;
eight repetitions do not guarantee statistical power or venue acceptance.
