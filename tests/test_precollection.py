"""Safety and analysis checks for the offline, post-review handoff."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch


class AuditTests(unittest.TestCase):
    def test_public_inventory_excludes_private_journals_not_receipts(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['requests/private.json', 'transport/private.json',
                         'transport_recovery/private_snapshot.json', 'runs/orphan.json']:
                audit.write_json(root/audit.CALIBRATION/name, {})
            self.assertEqual(audit.public_artifacts(root),
                             {(audit.CALIBRATION/'runs/orphan.json').as_posix()})

    def test_locked_report_delete_marks_separate_state_failed(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            audit.write_json(out/'report.json', {'status': 'OFFLINE_AUDIT_PASSED_NOT_MAIN_READY'})
            with patch.object(Path, 'unlink', side_effect=PermissionError('sharing lock')), \
                    patch.object(audit, 'check_pins') as check:
                with self.assertRaisesRegex(PermissionError, 'sharing lock'):
                    audit.run(output=out)
                check.assert_not_called()
            self.assertEqual(audit.read(out/'audit_state.json')['status'], 'FAILED')
            with self.assertRaisesRegex(ValueError, 'failed, pending or stale'):
                audit.verified_report(out)

    def test_stale_lock_file_does_not_preserve_clearance(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'workflow.lock').touch()
            audit.write_json(out/'report.json', {'status': 'OFFLINE_AUDIT_PASSED'})
            with patch.object(audit, 'check_pins', side_effect=ValueError('changed source')):
                with self.assertRaisesRegex(ValueError, 'changed source'):
                    audit.run(output=out)
            self.assertEqual(audit.read(out/'report.json')['status'], 'FAILED')

    def test_live_lock_does_not_allow_second_writer(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with audit.workflow_lock(out):
                with self.assertRaisesRegex(RuntimeError, 'already holds'):
                    audit.run(output=out)

    def test_early_failure_removes_stale_clearance(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            audit.write_json(out/'report.json', {'status': 'OFFLINE_AUDIT_PASSED'})
            with patch.object(audit, 'check_pins', side_effect=ValueError('changed source')):
                with self.assertRaisesRegex(ValueError, 'changed source'):
                    audit.run(output=out)
            self.assertEqual(audit.read(out/'report.json')['status'], 'FAILED')

    def test_failed_pending_write_never_runs_replay(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            audit.write_json(out/'report.json', {'status': 'OFFLINE_AUDIT_PASSED'})
            with patch.object(audit, 'write_json', side_effect=OSError('disk full')), \
                    patch.object(audit, 'check_pins') as check:
                with self.assertRaisesRegex(OSError, 'disk full'):
                    audit.run(output=out)
                check.assert_not_called()
            self.assertFalse((out/'report.json').exists())

    def test_changed_file_rejected_before_import(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'fake.py'
            source.write_text('raise RuntimeError("must never import")', encoding='utf8')
            pins = {'sources': {'fake.py': audit.sha(source)}}
            source.write_text('raise RuntimeError("changed")', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'Pinned file changed'):
                audit.verify_files(root, pins['sources'])
            # Fresh interpreter: an import-time self-hash cannot conceal the edit.
            code = ('from pathlib import Path; import precollection_audit as a; '
                    f'a.verify_files(Path({str(root)!r}), {pins["sources"]!r}); '
                    'raise RuntimeError("dependency import reached")')
            child = subprocess.run([sys.executable, '-B', '-c', code], cwd=audit.ROOT,
                                   capture_output=True, text=True, timeout=30)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn('Pinned file changed', child.stderr)
            self.assertNotIn('RuntimeError: dependency import reached', child.stderr)

    def test_orphan_receipt_invalidates_clearance(self):
        import precollection_audit as audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/audit.CALIBRATION
            data.mkdir(parents=True)
            audit.write_json(root/'pins.json', dict(version='post-review-pins-1',
                            main_authorized=False, sources={name: '' for name in
                            ['precollection_spec.json', 'requirements.lock.txt', '.python-version']}, artifacts={}))
            audit.write_json(data/'orphan.json', {})
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'PIN_FILE', root/'pins.json'), \
                    patch.object(audit, 'verify_files'):
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    audit.run(output=root/'new-audit')
            self.assertEqual(audit.read(root/'new-audit/report.json')['status'], 'FAILED')


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        import precollection_analysis as analysis
        self.a = analysis
        self.spec = analysis.load_spec()

    def fixture(self):
        rows = []
        for cell in self.a.design(self.spec):
            # Shared schedule and fixed model offsets must cancel in contrasts.
            offset = (cell['seed'] % 4)*.02 + self.spec['models'].index(cell['model'])*.01
            effect = .1 if cell['culture'] == 'india' else 0
            rows.append(dict(**cell, valid_graph=True, edges=10,
                             density=.2+offset+effect, avg_clustering_coef=.2+offset+effect))
        return rows

    def test_design_and_fixed_panel(self):
        rows = self.fixture()
        self.assertEqual(len(rows), 3584)
        result = self.a.primary_contrasts(rows, self.spec)
        self.assertEqual(len(result), 24)
        india = [r for r in result if r['culture'] == 'india']
        for row in india:
            self.assertEqual(row['complete_schedule_blocks'], 32)
            self.assertAlmostEqual(row['complete_block_mean'], .1)
            self.assertIsNone(row['holm_p'])  # constant differences cannot establish precision
        self.assertEqual(len({r['hypothesis_id'] for r in result}), 24)

    def test_missing_model_never_reweights_panel_or_family(self):
        rows = self.fixture()
        target = next(r for r in rows if r['method'] == 'global' and r['culture'] == 'india')
        target.update(valid_graph=False, edges=None, density=None, avg_clustering_coef=None)
        result = self.a.primary_contrasts(rows, self.spec)
        row = next(r for r in result if r['method'] == 'global' and r['culture'] == 'india')
        self.assertEqual(row['complete_schedule_blocks'], 31)
        self.assertEqual(row['status'], 'MISSING_NO_INFERENCE')
        self.assertIsNone(row['holm_p'])
        self.assertLess(row['missingness_lower'], .1)
        self.assertGreater(row['missingness_upper'], .1)
        self.assertEqual(len(result), 24)

    def test_empty_is_not_invalid(self):
        rows = self.fixture()
        rows[0].update(edges=0, density=0., avg_clustering_coef=0.)
        counts = self.a.denominators(rows, ['density'])
        group = next(r for r in counts if r['method'] == rows[0]['method'])
        self.assertEqual(group['attempted'], 896)
        self.assertEqual(group['valid'], 896)
        self.assertEqual(group['nonempty'], 895)
        self.assertEqual(group['endpoint_defined'], 896)

    def test_missing_and_duplicate_cells_rejected(self):
        rows = self.fixture()
        for invalid in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                self.a.primary_contrasts(invalid, self.spec)

    def test_unknown_contract_and_infinite_endpoint_rejected(self):
        rows = self.fixture()
        rows[0]['contract'] = 'another-experiment'
        with self.assertRaises(ValueError):
            self.a.primary_contrasts(rows, self.spec)
        rows = self.fixture()
        rows[0]['density'] = float('inf')
        with self.assertRaises(ValueError):
            self.a.primary_contrasts(rows, self.spec)

    def test_no_hidden_promotion_or_spend(self):
        self.assertFalse(self.spec['main_authorized'])
        self.assertEqual(self.spec['primary_family_size'], 24)
        self.assertNotIn('religion', str(self.spec['primary_endpoints']))
        changed = copy.deepcopy(self.spec)
        changed['main_authorized'] = True
        with self.assertRaises(ValueError):
            self.a.validate_spec(changed)
        for field, value in [('version', 'other'), ('status', 'APPROVED'), ('claim', 'general language'),
                             ('confirmation_reuses_calibration', True), ('precision_scenario_half_width', .99)]:
            changed = copy.deepcopy(self.spec)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.a.validate_spec(changed)
        self.assertTrue(all(r['planning_target_half_width'] == .03 for r in self.a.precision_scenarios(self.spec)))

    def test_family_inference_and_informative_missingness(self):
        rows = self.fixture()
        for row in rows:
            if row['culture'] == 'india':
                # Nonzero variance at the schedule level, not model-row level.
                for metric in ['density', 'avg_clustering_coef']:
                    row[metric] += .01*((row['seed'] % 3)-1)
        result = self.a.primary_contrasts(rows, self.spec)
        india = next(r for r in result if r['method'] == 'local' and r['culture'] == 'india')
        self.assertLess(india['holm_p'], .05)
        self.assertGreater(india['simultaneous_ci'][0], 0)
        # Deliberately erase high outcomes: a significant available-case result
        # must not be promoted into full-design inference under informative loss.
        for row in rows:
            if row['culture'] == 'india' and row['seed'] % 3 == 2:
                row.update(valid_graph=False, edges=None, density=None, avg_clustering_coef=None)
        result = self.a.primary_contrasts(rows, self.spec)
        self.assertTrue(all(r['holm_p'] is None for r in result if r['culture'] == 'india'))

    def test_bimodal_null_and_model_heterogeneity(self):
        rows = self.fixture()
        for row in rows:
            if row['culture'] == 'india':
                model_effect = [-.03, -.01, .01, .03][self.spec['models'].index(row['model'])]
                for metric in ['density', 'avg_clustering_coef']:
                    row[metric] += -.1 + model_effect + (.03 if row['seed'] % 2 else -.03)
        result = self.a.primary_contrasts(rows, self.spec)
        india = next(r for r in result if r['method'] == 'local' and r['culture'] == 'india')
        self.assertAlmostEqual(india['complete_block_mean'], 0.)
        self.assertGreater(india['holm_p'], .99)
        self.assertAlmostEqual(india['per_model']['gpt-4.1']['mean'], -.03)


class CandidateTests(unittest.TestCase):
    def test_duplicate_only_canonicalization_preserves_raw(self):
        from precollection_candidate import canonical_global
        raw = '0, 1\n1, 0\n0, 1\n1, 2'
        result = canonical_global(raw, ['0', '1', '2'])
        self.assertEqual(result['raw_response'], raw)
        self.assertEqual(result['edges'], [['0', '1'], ['1', '2']])
        self.assertEqual(result['repeated_pairs'], 2)
        self.assertEqual(result['reversed_pairs'], 1)

    def test_invalid_line_is_never_dropped(self):
        from precollection_candidate import canonical_global
        for text in ['0, 1\n1, 99', '0, 1\n1, 1', '0, 1\nNONE', '0, 1\nhello', '01, 2', '0,1,2']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                canonical_global(text, ['0', '1', '2'])
        self.assertEqual(canonical_global('NONE', ['0', '1'])['edges'], [])
        parsed = canonical_global('0, 1\n\n1, 2', ['0', '1', '2'])
        self.assertEqual(parsed['interior_blank_lines'], 1)
        self.assertEqual(parsed['edges'], [['0', '1'], ['1', '2']])

    def test_hindi_change_is_isolated(self):
        import repair_v8
        from precollection_candidate import candidate_catalog
        before = repair_v8.catalog()
        candidate = candidate_catalog()
        self.assertFalse(candidate['main_authorized'])
        for old, new in zip(before, candidate['rows']):
            if old['language'] == 'hindi' and old['method'] == 'global':
                self.assertNotEqual(old['system'], new['system'])
                self.assertIn('यथार्थसंगत और विश्वसनीय', new['system'])
            else:
                self.assertEqual(old, new)
        self.assertEqual(before, repair_v8.catalog())


if __name__ == '__main__':
    unittest.main()
