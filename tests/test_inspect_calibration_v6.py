import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from inspect_calibration_v6 import failure_rates, forecasts


class InspectionTests(unittest.TestCase):
    def test_first_response_denominator_excludes_corrections(self):
        base = dict(model='m', method='local', culture='us', language='portuguese',
                    prompt_method='local', requested_count=1)
        rows = [{**base, 'attempt': a, 'valid': valid}
                for a, valid in [(1, False), (2, True), (1, True)]]
        result = failure_rates(rows)[0]
        self.assertEqual(result['first_decisions'], 2)
        self.assertEqual(result['first_failures'], 1)
        self.assertEqual(result['correction_attempts'], 1)
        self.assertEqual(result['first_failure_rate'], .5)

    def test_country_and_effective_prompt_are_separate_compliance_strata(self):
        base = dict(model='m', method='sequential', culture='us', language='english',
                    prompt_method='local', requested_count=1, attempt=1, valid=True)
        rows = [base, {**base, 'culture':'india', 'valid':False},
                {**base, 'prompt_method':'sequential'}]
        result = failure_rates(rows)
        self.assertEqual(len(result), 3)
        self.assertEqual({(r['culture'],r['prompt_method'],r['first_failure_rate']) for r in result},
                         {('us','local',0.), ('india','local',1.), ('us','sequential',0.)})

    def test_record_retains_actual_early_sequential_prompt(self):
        import inspect_calibration_v6 as inspector
        cell = dict(run_id='r', model='m', method='sequential', culture='us',
                    language='english', repetition=0)
        decision = dict(method='sequential', prompt_method='local', requested_count=1,
                        attempt=1, valid=True)
        row = inspector.decisions_from_record(dict(cell=cell, decisions=[decision]))[0]
        self.assertEqual(row['method'], 'sequential')
        self.assertEqual(row['decision_method'], 'sequential')
        self.assertEqual(row['prompt_method'], 'local')

    def test_incomplete_calibration_has_no_main_forecast(self):
        self.assertEqual(forecasts([], complete=False), [])

    def test_all_settings_forecast_counts_and_no_reuse(self):
        models = ['gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol']
        methods = ['global', 'local', 'sequential', 'iterative']
        settings = [('us', 'english'), ('india', 'english'), ('japan', 'english'),
                    ('brazil', 'english'), ('us', 'hindi'), ('us', 'japanese'), ('us', 'portuguese')]
        rows = [dict(model=model, method=method, culture=culture, language=language,
                     conservative_usd=1., api_seconds=3600.)
                for model in models for method in methods for culture, language in settings
                if model == 'gpt-6-luna' or culture == 'us']
        results = forecasts(rows, complete=True)
        self.assertEqual([r['networks'] for r in results], [896, 1568])
        self.assertEqual([r['conservative_usd'] for r in results], [896., 1568.])
        self.assertEqual(results[0]['unmeasured_country_transfer_networks'], 288)
        self.assertEqual(results[1]['unmeasured_country_transfer_networks'], 504)
        self.assertEqual(results[0]['serial_api_hours'], 896.)
        self.assertTrue(all(r['calibration_reuse'] is False for r in results))

    def test_missing_cell_cannot_silently_lower_forecast(self):
        with self.assertRaises(ValueError):
            forecasts([], complete=True)

    def test_rejected_notebook_report_cannot_feed_the_next_cell(self):
        cells = json.loads(Path('analyze_networks.ipynb').read_text(encoding='utf-8'))['cells']
        load, display = (''.join(c['source']) for c in cells[-2:])
        namespace = {'display': Mock(), 'v6_report': {'old': 'previous execution'}}
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                target = Path('outputs/calibration_v6/5fb3715550db')
                target.mkdir(parents=True)
                (target/'inspection.json').write_text(json.dumps({
                    'contract_sha256':'wrong', 'hypothetical_main_forecasts':[{'networks':896}]
                }), encoding='utf-8')
                with self.assertRaises(AssertionError):
                    exec(load, namespace)
                self.assertIsNone(namespace['v6_report'])
                exec(display, namespace)
                namespace['display'].assert_not_called()
            finally:
                os.chdir(original)
