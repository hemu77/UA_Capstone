"""One owner-authorized HTTP-500 replacement; frozen prompts and costs stay intact."""
import argparse
import contextlib
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import repair_v8 as v
import repair_v8_io as io

RID = '186b5ebdcc30c2fbe5e5e4da4232b3c120c7e188f2935e5d8c58fde6234e78c0'
CONTRACT = '176533db70415212b5b53e72afd3ba1400dc50ba2c6847f235d19ae65202a3b8'
BEFORE = 'c25a37af86ce420be50595c11336770e2898b01f601d9764ba4f9ddd0aca46cd'
NONCE = 'owner-oct13-v8-http500-one-replacement'
REASON = 'Owner authorized one exact replacement on 2026-10-13; original full reservation retained; no main study.'
SOURCE_SHA = v.r.base.sha(__file__)
BASE_ID, BASE_RECEIPTS = v.r.base.request_id, v.receipts


def replacement_id():
    return hashlib.sha256(json.dumps([RID, 'replacement', NONCE]).encode()).hexdigest()


def target():
    io.check_source()
    if v.r.base.sha(__file__) != SOURCE_SHA or v.r.digest(v.contract('calibration')) != CONTRACT:
        raise ValueError('Recovery source or contract differs.')
    return v.r.folder(v.contract('calibration'))


def validate_original(row):
    if v.r.digest(row) != BEFORE or row[0] != RID or row[3] != 'uncertain':
        raise ValueError('Not the explicitly inspected uncertain request.')
    request, response = json.loads(row[4]), json.loads(row[5])
    if (request['cell']['repair_contract_sha256'] != CONTRACT
            or BASE_ID(request['cell'], request['ordinal'], request['messages']) != RID
            or response.get('error_type') != 'InternalServerError'
            or response['diagnostics'].get('http_status') != 500):
        raise ValueError('Recovery identity or HTTP-500 evidence differs.')


def evidence():
    record = v.read(target()/'transport_recovery'/'recovery.json')
    original, retired = record['original_row'], record['retired_row']
    validate_original(original)
    response = json.loads(retired[5])
    if (record.get('source_sha256') != SOURCE_SHA or record.get('nonce') != NONCE
            or record.get('replacement_id') != replacement_id()
            or retired[:3] != original[:3] or retired[3] != 'abandoned' or retired[4] != original[4]
            or response.get('previous_response') != json.loads(original[5])
            or response.get('previous_status') != 'uncertain'
            or response.get('authorization') != REASON
            or response.get('replacement_authorization') != NONCE
            or response.get('retained_reservation_usd') != original[2]
            or response.get('accounting') != 'full_reservation_retained_no_replay'
            or response.get('billing_verified') is not False):
        raise ValueError('Retirement evidence differs.')
    return record


def retire():
    """Offline, idempotent retirement; crash recovery never grants another nonce."""
    with v.r.base.frozen.workflow_lock():
        root = target()
        approved = v.read(root.parent/'authorization.json')
        if (approved['contract_sha256'] != CONTRACT or approved['additional_usd'] != 15.
                or approved['cumulative_ceiling_usd'] != approved['ledger_start_usd']+15.):
            raise ValueError('Original calibration allowance differs.')
        budget = v.r.paid.Budget(v.r.LEDGER, total_cap=approved['cumulative_ceiling_usd'],
            pilot_cap=5., require_existing=True, allow_higher_cap=True)
        try:
            folder = root/'transport_recovery'
            snapshot = folder/'private_snapshot.json'
            rows = {r[0]: r for r in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
            journal_path = root/'requests'/(RID+'.json')
            if not snapshot.exists():
                validate_original(rows[RID])
                if budget.unresolved_count() != 1 or v.read(journal_path).get('row_sha256') != BEFORE:
                    raise ValueError('Another unresolved request or changed checkpoint.')
                v.r.write_json(snapshot, dict(original_row=rows[RID], original_journal=v.read(journal_path),
                    original_summary=v.read(root/'summary.json'), source_sha256=SOURCE_SHA,
                    other_rows={rid: v.r.digest(row) for rid, row in rows.items() if rid != RID}))
            saved = v.read(snapshot)
            validate_original(saved['original_row'])
            if saved['source_sha256'] != SOURCE_SHA or any(v.r.digest(rows.get(rid)) != sha for rid, sha in saved['other_rows'].items()):
                raise ValueError('Historical ledger or recovery source changed.')
            if rows[RID][3] == 'uncertain':
                validate_original(rows[RID])
                budget.abandon(RID, REASON, replacement_authorization=NONCE)
            retired = v.r.base.ledger_row(budget, RID)
            record = dict(original_row=saved['original_row'], retired_row=retired,
                source_sha256=SOURCE_SHA, nonce=NONCE, replacement_id=replacement_id(), main_authorized=False)
            path = folder/'recovery.json'
            if path.exists() and v.read(path) != v.r.paid.clean_numbers(record):
                raise ValueError('Existing retirement record differs.')
            v.r.write_json(path, record)
            evidence()
            current_journal = v.read(journal_path)
            expected = dict(request_id=RID, row_sha256=v.r.digest(retired))
            if current_journal not in [saved['original_journal'], expected]:
                raise ValueError('Unexpected original journal transition.')
            with io.local_writes(root, 'retire_exact_http500'):
                v.r.base.write_json(journal_path, expected)
            return dict(status='ONE_REPLACEMENT_AUTHORIZED', retained_reservation_usd=retired[2],
                replacement_id=replacement_id(), paid_calls=0, main_authorized=False)
        finally:
            budget.db.close()


def check_retired(budget, record):
    row = v.r.base.ledger_row(budget, RID)
    if v.r.paid.clean_numbers(row) != record['retired_row']:
        raise ValueError('Retired ledger row differs.')


def accounting(budget, approved, spec):
    record = evidence()
    check_retired(budget, record)
    current = {row[0]: v.r.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
    if budget.unresolved_count() or any(current.get(rid) != sha for rid, sha in approved['baseline'].items()):
        raise ValueError('Historical ledger changed or unresolved requests remain.')
    relevant, allowed = set(), {cell['run_id'] for cell in v.r.cells(spec)}
    for rid, status, raw in budget.db.execute('SELECT id,status,request FROM attempts'):
        request = json.loads(raw)
        cell = request.get('cell', {})
        if cell.get('repair_contract_sha256') == v.r.digest(spec):
            if cell.get('run_id') not in allowed or (status != 'received' and rid != RID):
                raise ValueError('Unapproved or unresolved stage request.')
            if rid == replacement_id() and request.get('replaces_request_id') != RID:
                raise ValueError('Replacement provenance missing.')
            relevant.add(rid)
    journal = {}
    for path in (target()/'requests').glob('*.json'):
        item = v.read(path)
        if item.get('request_id') != path.stem or item.get('row_sha256') != current.get(path.stem):
            raise ValueError('Request journal differs; no automatic replacement.')
        journal[path.stem] = item
    if relevant != set(journal):
        raise ValueError('Ledger/journal request set differs.')


@contextlib.contextmanager
def recovered():
    record = evidence()
    original = json.loads(record['original_row'][4])

    def request_id(cell, ordinal, messages):
        rid = BASE_ID(cell, ordinal, messages)
        if rid == RID:
            if any(original[k] != val for k, val in [('cell', cell), ('ordinal', ordinal), ('messages', messages)]):
                raise ValueError('Replacement request differs.')
            return replacement_id()
        return rid

    def receipts(budget, cell, spec, root):
        check_retired(budget, record)
        # Only the logical replay excludes the retired attempt. Real budget accounting includes it.
        class ReceiptRows:
            def execute(self, sql, params):
                expected = "SELECT id,phase,cost,status,request,response FROM attempts WHERE json_extract(request,'$.cell.run_id')=?"
                if sql != expected:
                    raise ValueError('Frozen receipt query changed.')
                return budget.db.execute(sql+' AND id<>?', (*params, RID))
        return BASE_RECEIPTS(SimpleNamespace(db=ReceiptRows()), cell, spec, root)

    with patch.object(v.r.base, 'request_id', request_id), patch.object(v, 'receipts', receipts), \
            patch.object(v, 'ORIGINAL_ACCOUNTING', accounting):
        yield record


def run(action):
    if action == 'retire':
        return retire()
    with recovered() as record:
        if action == 'inspect':
            import inspect_repair_v8 as inspector
            writer = v.r.write_json
            def publish(path, report):
                if path == target()/'inspection.json' and report.get('status') == 'PUBLIC_REPLAY_VERIFIED':
                    report.update(retained_uncertain_reservation_usd=record['retired_row'][2],
                        total_charge_or_reservation_usd=report['conservative_usd']+record['retired_row'][2],
                        transport_recovery_sha256=v.r.digest(record))
                return writer(path, report)
            # Publish clearance and retained-cost evidence atomically under the inspector lock.
            with patch.object(v.r, 'write_json', publish):
                return inspector.inspect('calibration')
        if action == 'execute':
            return io.execute('calibration')
        if action == 'verify':
            return v.execute('calibration', verify_only=True)
        raise ValueError('Unknown action.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['retire', 'execute', 'verify', 'inspect'])
    print(json.dumps(run(parser.parse_args().action), indent=2))
