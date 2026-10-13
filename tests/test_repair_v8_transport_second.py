"""Two independently authorized transport losses never become a general retry policy."""
import contextlib
import io
import json
import sqlite3
import unittest
from unittest.mock import patch

import httpx
import repair_v8_transport_second as second
from tests import test_repair_v8_transport as fixtures


class SecondRecoveryTests(fixtures.RecoveryTests):
    def two_losses(self):
        spec, root, state, first_row = self.scenario('local')
        first, v = second.first, second.v
        first.retire()
        cell = v.r.cells(spec)[0]
        fixture = v.r.base.FixtureCaller(cell)
        with v.adapter(cell, spec):
            v.r.base.run_cell(cell, fixture)
        texts = {json.dumps(q['messages'], sort_keys=True): q['text'] for q in fixture.requests}
        state['fail_second'] = True
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            state['posts'] += 1
            if state['fail_second'] and state['posts'] >= 4:
                return httpx.Response(520, json={'error': {'message': 'fixture', 'type': 'server_error'}}, request=request)
            return self.response(request, text=texts[json.dumps(json.loads(request.content)['messages'], sort_keys=True)])
        self.client(handler)
        with first.recovered(), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'unresolved'):
            v.execute('wording_probe')
        with contextlib.closing(sqlite3.connect(v.r.LEDGER)) as db:
            row = db.execute("SELECT id,phase,cost,status,request,response FROM attempts WHERE status='uncertain'").fetchone()
            prefix = db.execute("SELECT id,phase,cost,status,request,response FROM attempts WHERE status='received'").fetchall()
        self.assertEqual(len(prefix), 2)
        self.enterContext(patch.object(second, 'RID', row[0]))
        self.enterContext(patch.object(second, 'BEFORE', v.r.digest(row)))
        self.enterContext(patch.object(second, 'target', return_value=root))
        self.enterContext(patch.object(second, 'INSPECTION_ROOT', root))
        return root, state, first_row, row, prefix

    def test_two_reservations_exact_replay_and_crash_recovery(self):
        root, state, old, row, prefix = self.two_losses()
        v, first = second.v, second.first
        # RED guard: the first approval does not cover the second failed request.
        with first.recovered(), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'unresolved'):
            v.execute('wording_probe')
        self.assertEqual(state['posts'], 4)
        abandon = v.r.paid.Budget.abandon
        def interrupted(budget, *args, **kwargs):
            abandon(budget, *args, **kwargs)
            raise OSError('fixture stop after committed retirement')
        with patch.object(v.r.paid.Budget, 'abandon', interrupted), self.assertRaisesRegex(OSError, 'committed retirement'):
            second.retire()
        retired = second.retire()
        self.assertEqual(retired, second.retire())
        self.assertAlmostEqual(retired['retained_reservation_usd'], old[2]+row[2])
        state['fail_second'] = False
        with second.recovered(), contextlib.redirect_stdout(io.StringIO()):
            result = v.execute('wording_probe')
            verified = v.execute('wording_probe', verify_only=True)
        self.assertEqual(result, verified)
        self.assertEqual(result['completed_graphs'], 1)
        self.assertEqual(state['posts'], 52)
        with contextlib.closing(sqlite3.connect(v.r.LEDGER)) as db:
            for saved in prefix:
                self.assertEqual(saved, db.execute('SELECT id,phase,cost,status,request,response FROM attempts WHERE id=?', (saved[0],)).fetchone())
            self.assertEqual(db.execute("SELECT count(*) FROM attempts WHERE status='abandoned'").fetchone()[0], 2)
        import inspect_repair_v8 as inspector
        inspect = inspector.inspect
        with patch.object(inspector, 'inspect', side_effect=lambda stage: inspect('wording_probe')):
            report = second.run('inspect')
        self.assertEqual(report['valid_graphs'], 1)
        self.assertAlmostEqual(report['retained_uncertain_reservation_usd'], old[2]+row[2])
        self.assertAlmostEqual(report['total_charge_or_reservation_usd'], report['conservative_usd']+old[2]+row[2])
        self.assertEqual(len(report['transport_recovery_sha256']), 2)
        self.assertEqual(v.read(root/'inspection.json'), report)
        record_path = root/'transport_recovery_second'/'recovery.json'
        record = v.read(record_path)
        record['nonce'] = 'wrong'
        v.r.write_json(record_path, record)
        with self.assertRaisesRegex(ValueError, 'Second retirement evidence'), second.recovered():
            pass
        with self.assertRaisesRegex(ValueError, 'Second retirement evidence'):
            second.run('inspect')
        self.assertEqual(v.read(root/'inspection.json')['status'], 'VALIDATION_PENDING')
        v.r.write_json(root/'inspection.json', report)
        with patch.object(first, 'evidence', side_effect=ValueError('Frozen contract changed')):
            with self.assertRaisesRegex(ValueError, 'Frozen contract changed'):
                second.run('inspect')
        self.assertEqual(v.read(root/'inspection.json')['status'], 'VALIDATION_PENDING')

    def test_second_replacement_loss_stops_without_chain(self):
        _, state, _, _, _ = self.two_losses()
        second.retire()
        with second.recovered(), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                second.v.execute('wording_probe')
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                second.v.execute('wording_probe')
        self.assertEqual(state['posts'], 5)


for cls in (fixtures.RecoveryTests, fixtures.fixtures.ExecutionTests):
    for name in vars(cls):
        if name.startswith('test_'):
            setattr(SecondRecoveryTests, name, None)
del cls


if __name__ == '__main__':
    unittest.main()
