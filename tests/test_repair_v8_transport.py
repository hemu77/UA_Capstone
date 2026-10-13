"""Exact one-request recovery with fake transport; never real API traffic."""
import contextlib
import io
import json
import sqlite3
import unittest
from unittest.mock import patch

import httpx
import repair_v8_transport as recovery
from tests import test_repair_v8 as fixtures


class RecoveryTests(fixtures.ExecutionTests):
    # Reuse the workspace fixture, not the parent class's unrelated test cases.
    def scenario(self, method='global'):
        spec, root, approved = self.setup_workspace(methods=(method,))
        v = recovery.v
        state = {'posts': 0, 'fail': True}
        texts = {}
        if method == 'local':
            cell = v.r.cells(spec)[0]
            fixture = v.r.base.FixtureCaller(cell)
            with v.adapter(cell, spec):
                v.r.base.run_cell(cell, fixture)
            texts = {json.dumps(q['messages'], sort_keys=True): q['text'] for q in fixture.requests}
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            state['posts'] += 1
            if state['fail'] and (method == 'global' or state['posts'] > 1):
                return httpx.Response(500, json={'error': {'message': 'fixture', 'type': 'server_error'}}, request=request)
            text = texts[json.dumps(json.loads(request.content)['messages'], sort_keys=True)] if texts else '0, 1\n1, 2'
            return self.response(request, text=text)
        self.client(handler)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'unresolved'):
            v.execute('wording_probe')
        with contextlib.closing(sqlite3.connect(v.r.LEDGER)) as db:
            row = db.execute("SELECT id,phase,cost,status,request,response FROM attempts WHERE status='uncertain'").fetchone()
        self.enterContext(patch.object(recovery, 'RID', row[0]))
        self.enterContext(patch.object(recovery, 'BEFORE', v.r.digest(row)))
        self.enterContext(patch.object(recovery, 'CONTRACT', v.r.digest(spec)))
        self.enterContext(patch.object(recovery, 'target', return_value=root))
        # The production recovery retains its existing $15 allowance.
        approved = v.read(root.parent/'authorization.json')
        approved.update(additional_usd=15., cumulative_ceiling_usd=15.)
        v.r.write_json(root.parent/'authorization.json', approved)
        self.enterContext(patch.dict(v.r.CAPS, wording_probe=15.))
        self.enterContext(patch.object(v, 'contract', return_value=spec))
        return spec, root, state, row

    def test_exact_replacement_then_read_only_resume(self):
        spec, root, state, row = self.scenario()
        v = recovery.v
        # RED evidence: frozen runner cannot continue an uncertain attempt.
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            v.execute('wording_probe')
        self.assertEqual(state['posts'], 1)
        first = recovery.retire()
        self.assertEqual(first, recovery.retire())
        state['fail'] = False
        with recovery.recovered(), contextlib.redirect_stdout(io.StringIO()):
            result = v.execute('wording_probe')
            verified = v.execute('wording_probe', verify_only=True)
            import inspect_repair_v8
            public = inspect_repair_v8.inspect('wording_probe')
        self.assertEqual(result, verified)
        self.assertEqual(public['valid_graphs'], 1)
        self.assertEqual(state['posts'], 2)
        inspect = inspect_repair_v8.inspect
        writer = v.r.write_json
        def fail_publication(path, value):
            if path.name == 'inspection.json' and value.get('status') == 'PUBLIC_REPLAY_VERIFIED':
                self.assertEqual(value['retained_uncertain_reservation_usd'], row[2])
                self.assertAlmostEqual(value['total_charge_or_reservation_usd'], value['conservative_usd']+row[2])
                raise OSError('fixture publication failure')
            return writer(path, value)
        with patch.object(inspect_repair_v8, 'inspect', side_effect=lambda stage: inspect('wording_probe')):
            with patch.object(v.r, 'write_json', fail_publication), self.assertRaisesRegex(OSError, 'publication'):
                recovery.run('inspect')
            self.assertEqual(v.read(root/'inspection.json')['status'], 'VALIDATION_PENDING')
            report = recovery.run('inspect')
            self.assertEqual(v.read(root/'inspection.json'), report)
            self.assertIn('transport_recovery_sha256', report)
        with contextlib.closing(sqlite3.connect(v.r.LEDGER)) as db:
            rows = db.execute('SELECT id,cost,status FROM attempts').fetchall()
        self.assertEqual(len(rows), 2)
        self.assertIn((row[0], row[2], 'abandoned'), rows)
        record = recovery.evidence()
        record['nonce'] = 'wrong'
        v.r.write_json(root/'transport_recovery'/'recovery.json', record)
        with self.assertRaisesRegex(ValueError, 'Retirement evidence'), recovery.recovered():
            pass

    def test_lost_replacement_stops_without_third_post(self):
        _, _, state, _ = self.scenario()
        recovery.retire()
        with recovery.recovered(), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                recovery.v.execute('wording_probe')
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                recovery.v.execute('wording_probe')
        self.assertEqual(state['posts'], 2)

    def test_crash_after_retirement_preserves_nonempty_prefix(self):
        _, _, state, row = self.scenario('local')
        v = recovery.v
        with contextlib.closing(sqlite3.connect(v.r.LEDGER)) as db:
            prefix = db.execute("SELECT id,phase,cost,status,request,response FROM attempts WHERE status='received'").fetchall()
        self.assertEqual(len(prefix), 1)
        abandon = v.r.paid.Budget.abandon
        def interrupted(budget, *args, **kwargs):
            abandon(budget, *args, **kwargs)
            raise OSError('fixture stop after committed retirement')
        with patch.object(v.r.paid.Budget, 'abandon', interrupted), self.assertRaisesRegex(OSError, 'committed retirement'):
            recovery.retire()
        recovery.retire()
        state['fail'] = False
        with recovery.recovered(), contextlib.redirect_stdout(io.StringIO()):
            result = v.execute('wording_probe')
            v.execute('wording_probe', verify_only=True)
        self.assertEqual(result['completed_graphs'], 1)
        self.assertEqual(state['posts'], 51)
        with contextlib.closing(sqlite3.connect(v.r.LEDGER)) as db:
            unchanged = db.execute('SELECT id,phase,cost,status,request,response FROM attempts WHERE id=?', (prefix[0][0],)).fetchone()
            retained = db.execute('SELECT cost,status FROM attempts WHERE id=?', (row[0],)).fetchone()
        self.assertEqual(unchanged, prefix[0])
        self.assertEqual(retained, (row[2], 'abandoned'))

    def test_orphan_and_insufficient_budget_are_not_retried(self):
        _, root, state, row = self.scenario()
        recovery.retire()
        v = recovery.v
        orphan = root/'transport'/'orphan.json'
        v.r.write_json(orphan, {})
        with recovery.recovered(), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Orphan'):
            v.execute('wording_probe')
        orphan.unlink()
        original_reserve = v.r.paid.Budget.reserve
        def capped(budget, *args, **kwargs):
            budget.total_cap = row[2]
            return original_reserve(budget, *args, **kwargs)
        with recovery.recovered(), patch.object(v.r.paid.Budget, 'reserve', capped), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(v.r.paid.BudgetExceeded):
            v.execute('wording_probe')
        self.assertEqual(state['posts'], 1)
        self.assertFalse((root/'requests'/(recovery.replacement_id()+'.json')).exists())


# Avoid collecting inherited tests twice while retaining their fixture helpers.
for name in list(vars(fixtures.ExecutionTests)):
    if name.startswith('test_'):
        setattr(RecoveryTests, name, None)


if __name__ == '__main__':
    unittest.main()
