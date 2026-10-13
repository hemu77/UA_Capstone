"""Four prospective Hindi graphs under a separate $1 cap; never replace V8 data."""
import argparse
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import repair_v8 as v
import repair_v8_io as io
import revision_hindi_candidate as candidate

STAGE = 'hindi_candidate_probe'
BASE_CONTRACT = v.contract
BASE_REVIEW = v.stage_review
CALIBRATION_CONTRACT = BASE_CONTRACT('calibration')
SOURCES = {name: v.r.base.sha(v.r.ROOT/name) for name in
           [Path(__file__).name, 'revision_hindi_candidate.py']}


def prompt(wording, country, actor, count, *, retry=False):
    if wording == 'clarified':
        return candidate.hindi_alternative_2(country, actor, count, retry=retry)
    if wording != 'original':
        raise ValueError('Unknown Hindi probe wording.')
    render = candidate.frozen.treatment_retry if retry else candidate.frozen.treatment_prompt
    return render('local', 'hindi', country, actor, count, wording='alternative_2')


def catalog():
    return [dict(wording=w, count=n, system=prompt(w, 'us', '0', n),
                 retry=prompt(w, 'us', '0', n, retry=True))
            for w in ('original', 'clarified') for n in (1, 2, 8, 49)]


def contract(stage=STAGE, recipe=None):
    if stage != STAGE or recipe is not None:
        raise ValueError('Only the separately approved four-cell Hindi probe is supported.')
    if any(v.r.base.sha(v.r.ROOT/name) != sha for name, sha in SOURCES.items()):
        raise ValueError('Hindi probe source changed; stop and review.')
    base = BASE_CONTRACT('wording_probe')
    # Counterbalance execution order; both arms share fresh schedules within a pair.
    design = [dict(model='gpt-4.1', method='local', culture='us', language='hindi',
                   repetition=rep, seed=34000+rep, wording=wording, global_arm=None)
              for rep, wording in [(0, 'original'), (0, 'clarified'),
                                   (1, 'clarified'), (1, 'original')]]
    return {**base, 'version': 'hindi-candidate-probe-1', 'stage': STAGE,
            'sources': {**base['sources'], **SOURCES}, 'design': design,
            'catalog_sha256': v.r.digest(catalog()), 'dependencies': [],
            'additional_cap_usd': 1., 'recipe': None,
            'selection_rule': 'Original versus clarified alternative_2, two fresh matched repetitions; no replacement or automatic adoption.',
            'interpretation': 'Post-diagnostic engineering probe; two repetitions cannot establish efficacy or translation equivalence.'}


@contextlib.contextmanager
def adapter(cell, spec):
    if cell not in v.r.cells(spec):
        raise ValueError('Unplanned Hindi probe cell.')
    proxy = SimpleNamespace(**vars(v.r.v6))
    proxy.system_prompt = lambda method, language, country, actor=None, count=None: prompt(
        cell['wording'], country, actor, count)
    proxy.retry_prompt = lambda method, language, country, actor=None, count=None, preserving=False: prompt(
        cell['wording'], country, actor, count, retry=True)
    with patch.object(v.r.base, 'revised', proxy), patch.object(v.r.base, 'contract', contract):
        yield


@contextlib.contextmanager
def isolated():
    # Scope module overrides to this serial process; frozen source files stay unchanged.
    with patch.dict(v.r.CAPS, {STAGE: 1.}), patch.object(v, 'contract', contract), \
            patch.object(v, 'catalog', catalog), patch.object(v, 'adapter', adapter), \
            patch.object(v, 'stage_review', stage_review):
        yield


def stage_review(spec):
    review = BASE_REVIEW(spec)
    # approve/execute invoke this while holding the shared workflow lock.
    previous = v.r.folder(CALIBRATION_CONTRACT)
    paths = [previous/'summary.json', previous/'inspection.json']
    if not all(path.exists() for path in paths):
        raise ValueError('Verified completed V8 calibration required before Hindi probe spending.')
    summary, inspection = map(v.read, paths)
    fingerprint = v.r.digest(CALIBRATION_CONTRACT)
    if (summary.get('contract_sha256') != fingerprint
            or summary.get('status') not in {'COMPLETE_REVIEW_REQUIRED', 'COMPLETE_WITH_INVALID_RESPONSES'}
            or summary.get('completed_cells') != 104
            or inspection.get('status') != 'PUBLIC_REPLAY_VERIFIED'
            or inspection.get('contract_sha256') != fingerprint
            or inspection.get('summary_sha256') != v.r.digest(summary)):
        raise ValueError('Verified completed V8 calibration required before Hindi probe spending.')
    return review


def run(action):
    with isolated():
        if action in {'execute', 'verify'}:
            if action == 'execute':
                return io.execute(STAGE)
            return v.execute(STAGE, verify_only=True)
        if action == 'inspect':
            import inspect_repair_v8 as inspector
            return inspector.inspect(STAGE)
        if action == 'approve':
            result = v.approve(STAGE)
            path = v.r.folder(contract()).parent/'authorization.json'
            record = v.read(path)
            record['basis'] = 'Owner approved an additional $1 for four original/clarified Hindi graphs on 2026-10-13; no main collection.'
            v.r.write_json(path, record)
            return {**result, 'basis': record['basis']}
        if action == 'preflight':
            with v.isolated():
                return v.r.preflight(STAGE)
        raise ValueError('Unknown Hindi probe action.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'approve', 'execute', 'verify', 'inspect'])
    try:
        print(json.dumps(run(parser.parse_args().action), indent=2))
    except Exception as error:
        parser.exit(1, type(error).__name__+': stopped; inspect evidence; no automatic API retry.\n')
