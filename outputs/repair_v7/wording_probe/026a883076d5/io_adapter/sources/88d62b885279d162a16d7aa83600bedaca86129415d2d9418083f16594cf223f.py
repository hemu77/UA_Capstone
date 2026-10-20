"""Offline-only repair of one inspected post-parse checkpoint; never buys replies."""
import argparse
import json
import sqlite3

import calibration_v6 as c

REQUEST_ID = 'b876c19986d015bfcf3407534e995fc6f0d9f704f5a4e433406c7bcf345fe104'
CONTRACT = '5fb3715550db87bb86061bf9f3c1484ddc3203b6a1b989758e96fa8ff1772b3d'
BEFORE = '367c37dbc9064eddbccced82c7bf3552934549f037dbe9c99818f9c5140af301'
AFTER = 'e99e6be99537ed1be8ce1d24f96e7cb6e05218a022d88f868064d3f832a72102'


def validate_transition(row, saved):
    """Only adding the independently replayable parse result may differ."""
    if row is None or row[3] != 'received' or saved.get('request_id') != row[0]:
        raise ValueError('Only a received, identified response can be reconciled.')
    response = json.loads(row[5])
    if response.get('parse') != {'valid': True, 'error': None, 'duplicate_edges': []}:
        raise ValueError('Expected the inspected successful parse result.')
    response.pop('parse')
    before = list(row)
    before[5] = json.dumps(response, ensure_ascii=False)
    if c.frozen.digest(before) != saved.get('row_sha256'):
        raise ValueError('Something other than the parse addition changed.')
    return dict(request_id=row[0], row_sha256=c.frozen.digest(row))


def replace_checkpoint(path, recovery, evidence, replacement):
    backup = recovery.with_name('checkpoint_original.json')
    if recovery.exists() or backup.exists():
        raise ValueError('Recovery evidence already exists; inspect rather than overwrite.')
    original = path.read_bytes()
    c.write_json(backup, json.loads(original))
    if backup.read_bytes() != original:
        raise ValueError('Checkpoint backup is not byte-identical.')
    c.write_json(recovery, evidence)
    c.write_json(path, replacement)


def reconcile(apply=False):
    with c.frozen.workflow_lock():
        spec = c.contract()
        if c.frozen.digest(spec) != CONTRACT:
            raise ValueError('This recovery is restricted to the inspected contract.')
        approved, target = c.review(spec), c.folder(spec)
        # Reuse the ledger's read methods, but make every database write impossible.
        budget = c.paid.Budget.__new__(c.paid.Budget)
        budget.db = sqlite3.connect(c.LEDGER.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            total = budget.spent()
            rows = {r[0]: r for r in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
            if budget.unresolved_count() or any(c.frozen.digest(rows.get(rid)) != value
                                               for rid, value in approved['baseline'].items()):
                raise ValueError('Uncertain attempts or changed historical ledger.')
            c.verify_probe(budget)
            path = target/'requests'/(REQUEST_ID+'.json')
            saved = json.loads(path.read_text(encoding='utf-8'))
            row = rows[REQUEST_ID]
            if saved.get('row_sha256') != BEFORE or c.frozen.digest(row) != AFTER:
                raise ValueError('Not the exact inspected checkpoint; no automatic general repair.')
            replacement = validate_transition(row, saved)
            journal_ids, journal_hashes = set(), {}
            for other in (target/'requests').glob('*.json'):
                checkpoint = json.loads(other.read_text(encoding='utf-8'))
                journal_hashes[other.name] = c.sha(other)
                journal_ids.add(other.stem)
                expected = AFTER if other.stem == REQUEST_ID else checkpoint.get('row_sha256')
                if (checkpoint.get('request_id') != other.stem or other.stem not in rows
                        or rows[other.stem][3] != 'received' or c.frozen.digest(rows[other.stem]) != expected):
                    raise ValueError('Another checkpoint requires inspection.')
            relevant = {rid for rid, value in rows.items()
                        if json.loads(value[4]).get('cell',{}).get('calibration_contract_sha256') == CONTRACT}
            if relevant != journal_ids:
                raise ValueError('Journal does not cover exactly the calibration attempts.')
            cell = json.loads(row[4])['cell']
            manifest = c.cells(spec)
            if approved['scope'] != 'extended104' or cell not in manifest:
                raise ValueError('Unapproved recovery cell.')
            prefix = [dict(request_id=rid, request=json.loads(value[4]), response=json.loads(value[5]))
                      for rid, value in rows.items() if json.loads(value[4]).get('cell',{}).get('run_id') == cell['run_id']]
            prefix.sort(key=lambda r: r['request']['ordinal'])
            if len(prefix) != 45 or prefix[-1]['request_id'] != REQUEST_ID:
                raise ValueError('Expected exactly the inspected 45-response prefix.')
            for ordinal, item in enumerate(prefix):
                if (item['request']['ordinal'] != ordinal or item['request']['cell'] != cell
                        or item['request_id'] != c.request_id(cell, ordinal, item['request']['messages'])):
                    raise ValueError('Prefix identity is inconsistent.')
            replay = c.ReplayCaller(prefix)
            try:
                c.run_cell(cell, replay)
            except ValueError as error:
                if str(error) != 'Replay lacks a recorded request.' or replay.ordinal != 45:
                    raise
            else:
                raise ValueError('Expected an incomplete graph, not a completed replacement.')
            evidence = dict(status='VERIFIED_OFFLINE_NOT_APPLIED', request_id=REQUEST_ID,
                contract_sha256=CONTRACT, source_sha256=c.sha(__file__), original_checkpoint=saved,
                replacement_checkpoint=replacement, replayed_responses=45, paid_requests=0,
                journal_sha256_before=journal_hashes,
                ledger_rows_sha256_before=c.frozen.digest(rows),
                ledger_before_usd=total, cause='Observed PermissionError; exact filesystem holder unknown',
                main_authorized=False)
            if apply:
                recovery = target/'checkpoint_recovery.json'
                replace_checkpoint(path, recovery, evidence, replacement)
                c.verify_accounting(budget, approved, target, {x['run_id'] for x in manifest})
                after_rows = {r[0]: r for r in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
                if after_rows != rows or budget.spent() != total:
                    raise ValueError('Offline recovery unexpectedly changed accounting.')
                after_journals = {p.name:c.sha(p) for p in (target/'requests').glob('*.json')}
                changed = [name for name in set(journal_hashes)|set(after_journals)
                           if journal_hashes.get(name) != after_journals.get(name)]
                if changed != [path.name]:
                    raise ValueError('Recovery changed more than the one inspected journal.')
                evidence.update(status='RECONCILED_NO_PAID_CALLS', ledger_after_usd=budget.spent(),
                    ledger_rows_unchanged=True, changed_journals=changed,
                    original_checkpoint_backup_sha256=c.sha(recovery.with_name('checkpoint_original.json')))
                c.write_json(recovery, evidence)
            return evidence
        finally:
            budget.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = reconcile(args.apply)
    print(json.dumps({k:v for k,v in result.items() if k != 'journal_sha256_before'}, indent=2))
