"""Eight English global-task diagnostics sharing the original $5 allowance.

This is a new hypothesis after the 64-clause screen, not replacement evidence.
The frozen payment/transport engine is reused in a serial, scoped adapter.
"""
import argparse
import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import repair_v7 as r
import repair_v7_checked as checked

INSTRUCTION = ('Construct a hypothetical friendship network for a simulation using the fictional people below. '
    'Generate a plausible realization of friendships, rather than recovering relationships already recorded '
    'in the candidate data.\n\n')
SOURCE = Path(__file__).name
IMPORTED_SHA = r.base.sha(__file__)
ORIGINAL_CONTRACT, ORIGINAL_ADAPTER = r.contract, r.adapter
ORIGINAL_CATALOG, ORIGINAL_FOLDER = r.catalog, r.folder


def origin_folder():
    return ORIGINAL_FOLDER(origin_contract())


def origin_contract():
    # Recomputing the parent identity must not inherit this extension's catalog.
    with patch.object(r, 'catalog', ORIGINAL_CATALOG):
        return ORIGINAL_CONTRACT('global_probe')


def catalog():
    row = next(row for row in ORIGINAL_CATALOG() if row['method'] == 'global'
        and row['language'] == 'english' and row['country'] == 'us' and row['global_arm'] == 'a0_b0')
    return [{**row, 'system': INSTRUCTION + row['system']}]


def contract(stage='global_probe', recipe=None):
    if stage != 'global_probe' or recipe is not None:
        raise ValueError('Only the approved eight-call English task probe is supported.')
    checked.check_source()
    if r.base.sha(__file__) != IMPORTED_SHA:
        raise ValueError('Extension source changed; restart and review the new identity.')
    original = origin_contract()
    return {**original, 'version': 'v7-explicit-global-task-1',
        'sources': {**original['sources'], SOURCE: IMPORTED_SHA,
                    'repair_v7_checked.py': checked.IMPORTED_SHA},
        'origin_contract_sha256': r.digest(original), 'catalog_sha256': r.digest(catalog()),
        'design': [row for row in original['design'] if row['global_arm'] == 'a0_b0' and row['repetition'] < 2],
        'interpretation': 'Post-diagnostic English task-interpretation probe; not bilingual or confirmatory evidence.'}


def folder(spec):
    return r.DESTINATION / 'global_task_probe' / r.digest(spec)[:12]


@contextlib.contextmanager
def adapter(cell, spec):
    with ORIGINAL_ADAPTER(cell, spec):
        original_prompt = r.base.revised.system_prompt
        r.base.revised.system_prompt = lambda *args, **kwargs: INSTRUCTION + original_prompt(*args, **kwargs)
        yield


@contextlib.contextmanager
def isolated():
    # The frozen engine already holds the cross-workflow lock during execution.
    with patch.object(r, 'contract', contract), patch.object(r, 'catalog', catalog), \
            patch.object(r, 'folder', folder), patch.object(r, 'adapter', adapter), \
            patch.object(r, 'verify_run', checked.verify_run):
        yield


def preflight():
    with isolated():
        return r.preflight('global_probe')


def origin_evidence():
    origin = origin_folder()
    approved = json.loads((origin.parent / 'authorization.json').read_text(encoding='utf8'))
    summary = json.loads((origin / 'summary.json').read_text(encoding='utf8'))
    original = origin_contract()
    if (approved['contract_sha256'] != r.digest(original) or approved['additional_usd'] != 5.
            or approved['cumulative_ceiling_usd'] != approved['ledger_start_usd'] + 5.
            or approved.get('main_authorized') is not False
            or summary.get('status') != 'COMPLETE_REVIEW_REQUIRED' or summary.get('completed_graphs') != 64
            or summary.get('contract_sha256') != r.digest(original)):
        raise ValueError('Completed original diagnostic and unchanged original allowance required.')
    return approved, r.digest(summary)


def authorize():
    """Bind the extension to the old ceiling, never grant another $5."""
    with r.base.frozen.workflow_lock():
        spec = contract()
        with isolated():
            r.stage_review(spec)
        origin, summary_sha = origin_evidence()
        path = folder(spec).parent / 'authorization.json'
        if path.exists():
            raise ValueError('Extension authorization exists; cannot reset the original cap.')
        budget = r.paid.Budget(r.LEDGER, require_existing=True)
        try:
            r.accounting(budget, origin, origin_contract())
            if (any(path.parent.glob('*/requests/*.json')) or any(path.parent.glob('*/runs/*.json'))
                    or any(path.parent.glob('*/transport/*.json'))
                    or any(json.loads(row[0]).get('cell', {}).get('frozen_source_sha256', {}).get(SOURCE)
                           for row in budget.db.execute('SELECT request FROM attempts'))):
                raise ValueError('Extension attempts already exist; reconcile, never re-anchor.')
            approval = {**origin, 'contract_sha256': r.digest(spec),
                'origin_authorization_sha256': r.digest(origin), 'origin_summary_sha256': summary_sha,
                'basis': 'Owner approved eight explicit-task graphs within the remaining original $5 allowance.',
                'baseline': {row[0]: r.digest(row) for row in budget.db.execute(
                    'SELECT id,phase,cost,status,request,response FROM attempts')}}
            r.write_json(path, approval)
            return {k: v for k, v in approval.items() if k != 'baseline'}
        finally:
            budget.db.close()


def execute():
    spec = contract()
    r.write_json(folder(spec) / 'summary.json', dict(status='VALIDATION_PENDING',
        contract_sha256=r.digest(spec), main_authorized=False))
    origin, summary_sha = origin_evidence()
    approved = json.loads((folder(spec).parent / 'authorization.json').read_text(encoding='utf8'))
    if (approved.get('origin_authorization_sha256') != r.digest(origin)
            or approved.get('origin_summary_sha256') != summary_sha
            or any(approved.get(key) != origin[key] for key in
                   ['ledger_start_usd', 'cumulative_ceiling_usd', 'additional_usd'])):
        raise ValueError('Extension must retain the original allowance and diagnostic evidence.')
    with isolated():
        return r.execute('global_probe')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'authorize', 'execute'])
    args = parser.parse_args()
    try:
        print(json.dumps(globals()[args.action](), indent=2))
    except Exception as error:
        parser.exit(1, type(error).__name__ + ': stopped; inspect evidence, never repurchase automatically.\n')
