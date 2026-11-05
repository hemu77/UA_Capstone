"""Fast regression: publication seal must reject code changes mid-evaluation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('primary_driver', ROOT/'scripts/run_primary_evaluation.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class DriverTests(unittest.TestCase):
    def test_mutated_code_cannot_receive_completed_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root/'scripts/run_primary_evaluation.py'
            script.parent.mkdir()
            script.write_text('# driver')
            code = root/'analysis_primary/core.py'
            code.parent.mkdir()
            code.write_text('# original')
            source = root/driver.SOURCE
            source.mkdir(parents=True)
            for name in ('summary.json','analysis_rows.json','analysis_summary.json',
                         'primary_contrasts.json','primary_contrasts.csv','network_metrics.csv','missingness_report.json'):
                (source/name).write_text('{}')
            output = root/'derived'

            def change_code(*args):
                code.write_text('# changed during computation')
                return []

            with patch.object(driver,'ROOT',root), patch.object(driver,'OUTPUT',output), \
                 patch.object(driver,'__file__',str(script)), \
                 patch.object(driver,'run_core',return_value=({'sources':{}},None,{})), \
                 patch.object(driver,'run_graph_evaluation',return_value={}), \
                 patch.object(driver,'run_empirical_evaluation',return_value={}), \
                 patch.object(driver,'render_figures',side_effect=change_code):
                with self.assertRaisesRegex(ValueError,'code changed'):
                    driver.main()
            manifest = json.loads((output/'evaluation_manifest.json').read_text())
            self.assertEqual(manifest['status'],'FAILED_INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
