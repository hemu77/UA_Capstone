"""Offline checkpoint audit. A partial report never grants completion clearance."""
import contextlib
import json
import math
import socket
import sqlite3
from unittest.mock import patch
from PIL import Image

import repair_v8_transport as recovery

PENDING_ID = 'f577020a10e58e11a3fee44bfce1fda5bbf6feca9f613c11afc8f57942fb213a'
PENDING_SHA = '915a6d7ae4b973b162197908e5d91a90eeae7dc4e7da48f37a6a65b9fba2456b'


def inspect():
    v = recovery.v
    with v.r.base.frozen.workflow_lock():
        root = v.r.DESTINATION/'calibration'/recovery.CONTRACT[:12]
        v.r.write_json(root/'checkpoint_inspection.json', dict(status='VALIDATION_PENDING',
            main_authorized=False, completion_clearance=False))
        return _inspect()


def _inspect():
    v = recovery.v
    with recovery.recovered(), v.isolated(), \
            patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline checkpoint audit')), \
            patch.object(v.r.paid, 'OpenAI', side_effect=AssertionError('No API client')):
        spec, root = v.contract('calibration'), recovery.target()
        fingerprint = v.r.digest(spec)
        cells = v.r.cells(spec)
        if v.read(root/'contract.json') != spec or v.read(root/'planned_cells.json') != cells:
            raise ValueError('Saved contract or design differs.')
        summary = v.read(root/'summary.json')
        if summary.get('status') != 'STOPPED' or summary.get('contract_sha256') != recovery.CONTRACT:
            raise ValueError('Expected the stopped exact-version calibration.')
        budget = v.r.paid.Budget.__new__(v.r.paid.Budget)
        budget.db = sqlite3.connect(v.r.LEDGER.resolve().as_uri()+'?mode=ro', uri=True)
        with contextlib.closing(budget.db):
            rows = {row[0]: row for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
            pending = rows[PENDING_ID]
            if v.r.digest(pending) != PENDING_SHA or budget.unresolved_count() != 1:
                raise ValueError('Unknown pending request; inspect separately.')
            approved = v.read(root.parent/'authorization.json')
            if any(v.r.digest(rows.get(rid)) != sha for rid, sha in approved['baseline'].items()):
                raise ValueError('Historical baseline differs.')
            journals = {}
            for path in (root/'requests').glob('*.json'):
                item = v.read(path)
                if item.get('request_id') != path.stem or item.get('row_sha256') != v.r.digest(rows.get(path.stem)):
                    raise ValueError('Journal/ledger mismatch.')
                journals[path.stem] = item
            relevant = {rid for rid,row in rows.items() if json.loads(row[4]).get('cell', {}).get('repair_contract_sha256') == fingerprint}
            if set(journals) != relevant:
                raise ValueError('Request sets differ.')
            if any(json.loads(rows[rid][4])['cell'] not in cells for rid in relevant):
                raise ValueError('Unplanned ledger cell.')
            expected = {cell['run_id'] for cell in cells}
            if any(path.stem not in expected for kind in ['runs', 'failures'] for path in (root/kind).glob('*.json')):
                raise ValueError('Unexpected terminal evidence.')
            if any(path.stem not in rows for path in (root/'transport').glob('*.json')):
                raise ValueError('Orphan transport evidence.')
            results, hashes = [], {}
            for cell in v.r.cells(spec):
                name = cell['run_id']+'.json'
                path, failure_path = root/'runs'/name, root/'failures'/name
                if path.exists() and failure_path.exists():
                    raise ValueError('Conflicting terminal evidence.')
                if not path.exists() and not failure_path.exists():
                    continue
                cached = v.receipts(budget, cell, spec, root)
                if path.exists():
                    record = v.checked.verify_run(path, cell, spec, budget)
                    with Image.open(path.with_suffix('.png')) as png:
                        png.verify()
                    failure = None
                else:
                    record = v.read(failure_path)
                    caller = v.ResumingCaller(cached)
                    with v.adapter(cell, spec):
                        _, failure = v.run_observation(cell, caller)
                    if failure != record.get('failure') or caller.replay.ordinal != len(cached):
                        raise ValueError('Failure replay differs.')
                if record['cell'] != cell or record['requests'] != cached:
                    raise ValueError('Terminal receipt differs.')
                results.append(v.summary_row(cell, record, failure))
                hashes[cell['run_id']] = v.r.base.sha(path if path.exists() else failure_path)
            if results != summary['results'] or len(results) != summary['completed_cells']:
                raise ValueError('Stopped summary differs from replay.')
            request = json.loads(pending[4])
            cell = request['cell']
            prefix = sorted([row for row in rows.values() if json.loads(row[4]).get('cell') == cell and row[3] == 'received'],
                key=lambda row: json.loads(row[4])['ordinal'])
            for ordinal, row in enumerate(prefix):
                saved = json.loads(row[4])
                if (saved['ordinal'] != ordinal or saved['settings'] != v.r.base.settings(cell['model'], cell['method'])
                        or row[0] != v.r.base.request_id(cell, ordinal, saved['messages'])):
                    raise ValueError('Prefix request identity differs.')
                v.r.verify_transport(root, row[0], json.loads(row[5]))
                response = json.loads(row[5])
                usage = response['usage']
                if any(type(usage[k]) is not int or usage[k] < 0 for k in ['prompt_tokens', 'completion_tokens']):
                    raise ValueError('Invalid prefix usage.')
                rates = spec['rates'][cell['model']]
                charge = (usage['prompt_tokens']*rates[0]*1.25 + usage['completion_tokens']*rates[1])/1e6
                if response['conservative_charge_usd'] != row[2] or not math.isclose(charge, row[2], rel_tol=0, abs_tol=1e-12):
                    raise ValueError('Prefix charge differs.')
            class Prefix(v.ResumingCaller):
                def __call__(self, model, messages):
                    if self.replay.ordinal == len(prefix):
                        if (model != request['model'] or messages != request['messages']
                                or self.replay.ordinal != request['ordinal']):
                            raise ValueError('Pending prompt differs.')
                        raise RuntimeError('EXPECTED_PENDING_BOUNDARY')
                    return super().__call__(model, messages)
            caller = Prefix([dict(request_id=row[0], request=json.loads(row[4]), response=json.loads(row[5])) for row in prefix])
            with v.adapter(cell, spec):
                try:
                    v.run_observation(cell, caller)
                except RuntimeError as error:
                    if str(error) != 'EXPECTED_PENDING_BOUNDARY':
                        raise
                else:
                    raise ValueError('Partial cell unexpectedly complete.')
            if {rid: v.r.digest(row) for rid,row in rows.items()} != {
                    row[0]: v.r.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}:
                raise ValueError('Ledger changed during read-only verification.')
            report = dict(status='PARTIAL_REPLAY_VERIFIED_PENDING_TRANSPORT', contract_sha256=recovery.CONTRACT,
                inspector_sha256=v.r.base.sha(__file__),
                completed_observations=len(results), valid_graphs=sum(row['valid_graph'] for row in results),
                terminal_failures=sum(not row['valid_graph'] for row in results), planned_observations=len(spec['design']),
                pending_prefix_received=len(prefix), pending_request_id=PENDING_ID, retained_pending_usd=pending[2],
                charge_or_reservation_usd=budget.spent()-approved['ledger_start_usd'],
                receipt_sha256=hashes, stopped_summary_sha256=v.r.digest(summary),
                ledger_unchanged=True, paid_calls=0, main_authorized=False, completion_clearance=False)
            v.r.write_json(root/'checkpoint_inspection.json', report)
            return {k:x for k,x in report.items() if k != 'receipt_sha256'}


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
