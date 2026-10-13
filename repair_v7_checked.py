"""Resume frozen V7 generation with JSON-canonical event verification only.

The paid experiment has already started. Its prompts, requests, ledger identities
and source contract stay frozen; this separately fingerprinted verifier changes
only the tuple/list representation used when checking serialized events.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import repair_v7 as frozen

FROZEN_RUNNER_SHA = 'b99a97a548af0c51ae5090ea6b177227a9855fda27f2bcf8e0f67dc71b14bbf1'
IMPORTED_SHA = frozen.base.sha(__file__)
ORIGINAL_VERIFY = frozen.verify_run


def check_source():
    if (frozen.base.sha(frozen.ROOT / 'repair_v7.py') != FROZEN_RUNNER_SHA
            or frozen.base.sha(__file__) != IMPORTED_SHA):
        raise ValueError('Verification adapter source changed; inspect before resuming.')


def verify_run(path, cell, spec, budget=None):
    check_source()
    original_run = frozen.base.run_cell

    def replay_json_events(*args, **kwargs):
        graph, events, decisions, schedule = original_run(*args, **kwargs)
        # The existing V6 verifier uses this same on-disk JSON normalization.
        return graph, frozen.paid.clean_numbers(events), decisions, schedule

    with patch.object(frozen.base, 'run_cell', replay_json_events):
        return ORIGINAL_VERIFY(path, cell, spec, budget)


def execute(stage, recipe=None):
    check_source()
    spec = frozen.contract(stage, recipe)
    evidence = dict(version='v7-json-event-verifier-1', contract_sha256=frozen.digest(spec),
        verifier_source_sha256=IMPORTED_SHA, frozen_runner_sha256=FROZEN_RUNNER_SHA,
        change='Verification only: canonicalize event tuples to JSON lists using paid.clean_numbers.',
        prompts_changed=False, request_identity_changed=False, main_authorized=False)
    path = frozen.folder(spec) / 'verification_adapter.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != evidence:
        raise ValueError('Existing verification adapter evidence differs.')
    frozen.write_json(path, evidence)
    with patch.object(frozen, 'verify_run', verify_run):
        return frozen.execute(stage, recipe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=list(frozen.CAPS))
    parser.add_argument('--recipe', type=Path)
    args = parser.parse_args()
    try:
        recipe = json.loads(args.recipe.read_text(encoding='utf8')) if args.recipe else None
        print(json.dumps(execute(args.stage, recipe), indent=2))
    except Exception as error:
        parser.exit(1, type(error).__name__ + ': stopped; inspect saved evidence, never repurchase automatically.\n')
