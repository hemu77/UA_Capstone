"""A lost reply keeps its cost; an explicitly approved replacement pays once."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import revision224 as study
from paid_study import Budget
import test_revision224_execution as execution_tests


class RecoveryTests(unittest.TestCase):
    workspace = execution_tests.ExecutionTests.workspace

    def test_explicit_replacement_preserves_charge_cache_and_receipt(self):
        with self.workspace() as (config, root, ledger, client, factory):
            client.chat.completions.create.side_effect = ConnectionError('offline lost reply')
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                study.execute(config, 1, 'calibration')
            budget = Budget(ledger, require_existing=True)
            old_id, old_cost = budget.db.execute('SELECT id,cost FROM attempts').fetchone()
            budget.abandon(old_id, 'Explicit one-attempt recovery within original cap',
                           replacement_authorization='offline-review-1')
            budget.db.close()
            client.chat.completions.create.side_effect = None
            study.execute(config, 1, 'calibration')
            path = next((root / study.RESULTS).glob('*.json'))
            original = path.read_bytes()
            record = json.loads(original)
            self.assertEqual(record['retained_abandoned_request_ids'], [old_id])
            self.assertNotEqual(record['requests'][0]['request_id'], old_id)
            budget = Budget(ledger, require_existing=True)
            self.assertAlmostEqual(budget.spent(), old_cost + record['requests'][0]['conservative_charge_usd'])
            self.assertEqual(budget.unresolved_count(), 0)
            budget.db.close()
            factory.reset_mock()
            study.execute(config, 1, 'calibration')
            factory.assert_not_called()
            self.assertEqual(original, path.read_bytes())
            for invalid_sources in [None, {'paid_study.py': 'tampered'}]:
                tampered = {**record, 'execution_source_sha256': invalid_sources}
                study.write_json(path, tampered)
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    study.execute(config, 1, 'calibration')
            study.write_json(path, record)
            report = study.calibration_report(config)
            self.assertEqual(report['unresolved_attempts'], 0)
            self.assertAlmostEqual(report['networks'][0]['retained_reservation_usd'], old_cost)
            record['retained_abandoned_request_ids'] = []
            study.write_json(path, record)
            with self.assertRaisesRegex(ValueError, 'incomplete|retained'):
                study.execute(config, 1, 'calibration')

    def test_replacement_requires_authorization_and_matching_request(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = Budget(Path(folder) / 'budget.sqlite')
            request = {'cell': {'run_id': 'fixture'}, 'ordinal': 0, 'model': 'fixture',
                       'messages': [], 'settings': {}}
            budget.reserve('old', 'main', .01, json.dumps(request))
            budget.abandon('old', 'Retire without resend')
            with self.assertRaisesRegex(RuntimeError, 'authorization'):
                budget.replacement_id('old', request)
            budget.reserve('authorized', 'main', .01, json.dumps(request))
            budget.abandon('authorized', 'One replacement', replacement_authorization='test-review')
            budget.reserve('another', 'main', .01, json.dumps(request))
            with self.assertRaisesRegex(ValueError, 'already been used'):
                budget.abandon('another', 'Cannot reuse', replacement_authorization='test-review')
            with self.assertRaisesRegex(ValueError, 'differs'):
                budget.replacement_id('authorized', {**request, 'ordinal': 1})
            budget.db.close()

    def test_source_compatibility_is_exact_and_never_permits_prompt_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            current = study.source_hashes()
            original = {**current, 'paid_study.py': hashlib.sha256(b'original').hexdigest()}
            snapshot = target / 'source_snapshot'
            snapshot.mkdir()
            (snapshot / 'paid_study.py').write_bytes(b'original')
            approval = dict(reviewed=True, source_sha256=current, generation_source_sha256=original,
                protocol_sha256=study.digest(study.load_config()),
                roster_sha256=study.digest(study.prompts.adult_roster()), runtime_versions=study.runtime_versions())
            with patch.object(study, 'DESTINATION', target):
                study.write_json(target / 'source_compatibility.json', approval)
                self.assertEqual(study.generation_source_hashes(study.load_config()), original)
                changed = {**current, 'paid_study.py': 'changed'}
                with patch.object(study, 'source_hashes', return_value=changed):
                    with self.assertRaisesRegex(ValueError, 'compatibility'):
                        study.generation_source_hashes(study.load_config())
                approval['generation_source_sha256']['revision224_prompts.py'] = 'changed'
                study.write_json(target / 'source_compatibility.json', approval)
                with self.assertRaisesRegex(ValueError, 'prompt|generation'):
                    study.generation_source_hashes(study.load_config())
