"""Offline replay audit of the interrupted V5 call, using a disposable ledger copy."""
import contextlib
import hashlib
import io
import json
import random
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import revision224 as study
from paid_study import Budget, PaidCaller

INTERRUPTED = '659cc446cedb665c4735858c9bd74d243ed79ecf4a076cbb3f5a9574a8c2d9a9'


def verify():
    config = study.load_config()
    ledger = study.ROOT / 'outputs/revision_budget_v1/budget.sqlite'
    with tempfile.TemporaryDirectory() as folder:
        with contextlib.closing(sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True)) as source:
            before = source.execute('SELECT COUNT(*),SUM(cost) FROM attempts').fetchone()
            with contextlib.closing(sqlite3.connect(Path(folder) / 'copy.sqlite')) as destination:
                source.backup(destination)
        budget = Budget(Path(folder) / 'copy.sqlite', require_existing=True)
        try:
            status, raw = budget.db.execute('SELECT status,request FROM attempts WHERE id=?', (INTERRUPTED,)).fetchone()
            request = json.loads(raw)
            cell = request['cell']
            assert request['ordinal'] == 194
            original = cell['frozen_source_sha256']
            current = study.source_hashes()
            for name, old_hash in original.items():
                if old_hash != current[name]:
                    assert name in {'paid_study.py', 'revision224.py'}
                    assert hashlib.sha256((study.DESTINATION / 'source_snapshot' / name).read_bytes()).hexdigest() == old_hash
            receipts = {}
            for path in (study.ROOT / study.RESULTS).glob('*.json'):
                record = json.loads(path.read_text(encoding='utf-8'))
                assert record['cell']['frozen_source_sha256'] == original
                study.verify_receipt(path, record['cell'], study.prompts.adult_roster(), budget)
                receipts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if status != 'abandoned':
                budget.abandon(INTERRUPTED, 'Disposable offline fixture only', replacement_authorization='offline-recovery-audit')
            client = Mock()
            client.chat.completions.with_raw_response.create.side_effect = RuntimeError('OFFLINE stop before transport')
            caller = PaidCaller(client, budget, cell, execution_source_sha256=current)
            random.seed(cell['seed'])
            study.np.random.seed(cell['seed'])
            roster = study.prompts.adult_roster()
            with patch.object(study.generation, 'get_system_prompt', study.prompts.system_prompt), \
                    patch.object(study.generation, 'get_user_prompt', study.prompts.user_prompt), \
                    patch.object(study.shared, 'get_llm_response', caller), \
                    patch.object(study.generation, 'update_graph_from_response', caller.parse_response), \
                    patch.object(study.shared.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                try:
                    study.generation.generate_network(cell['method'], study.prompts.DEMOS, roster,
                        list(roster), cell['model'], mean_choices=5, num_iter=3,
                        culture_context=cell['culture'], prompt_language=cell['language'], events=[])
                except RuntimeError as error:
                    assert 'unresolved' in str(error)
                else:
                    raise AssertionError('Expected the mock to stop at the lost request.')
            assert caller.ordinal == 195 and len(caller.request_ids) == 195
            assert client.chat.completions.with_raw_response.create.call_count == 1
            assert caller.retained_abandoned_request_ids == [INTERRUPTED]
            assert caller.last_request_id != INTERRUPTED
            originals = [row[0] for row in budget.db.execute(
                "SELECT id FROM attempts WHERE status='received' AND json_extract(request,'$.cell.run_id')=? ORDER BY json_extract(request,'$.ordinal')",
                (cell['run_id'],))]
            assert caller.request_ids[:194] == originals
        finally:
            budget.db.close()
        with contextlib.closing(sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True)) as source:
            assert source.execute('SELECT COUNT(*),SUM(cost) FROM attempts').fetchone() == before
    result = dict(status='PASS', paid_calls=0, cached_replies_reused=194,
        first_mock_transport_ordinal=194, completed_receipt_sha256=receipts,
        protocol_sha256=study.digest(config), source_sha256=current,
        generation_source_sha256=original, ledger_unchanged=True)
    study.write_json(study.DESTINATION / 'recovery_replay_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))


if __name__ == '__main__':
    verify()
