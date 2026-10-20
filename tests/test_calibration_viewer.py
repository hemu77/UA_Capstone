"""The visualization export must fail closed without modifying study evidence."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import export_calibration_viewer as viewer


class CalibrationViewerTests(unittest.TestCase):
    def test_review_path_is_portable_but_not_ambiguous(self):
        reviewed = {'reports_sha256': {'stats\\study\\analysis_report.json': 'checked'}}
        self.assertEqual(viewer.reviewed_report_hash(reviewed, 'stats/study/analysis_report.json'), 'checked')
        reviewed['reports_sha256']['stats/study/analysis_report.json'] = 'other'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            viewer.reviewed_report_hash(reviewed, 'stats/study/analysis_report.json')

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        config = viewer.study.load_config()
        self.cell = viewer.study.calibration_cells(config)[0]
        source = viewer.study.ROOT / viewer.study.RESULTS / (self.cell['run_id'] + '.json')
        self.path = Path(self.folder.name) / source.name
        for suffix in ['.json', '.adj', '.png']:
            shutil.copyfile(source.with_suffix(suffix), self.path.with_suffix(suffix))
        self.roster = json.loads((viewer.study.ROOT / config['persona_file']).read_text(encoding='utf-8'))
        self.sources = viewer.study.generation_source_hashes(config)
        self.receipt_hash = viewer.sha256(self.path)

    def check_record(self):
        return viewer.checked_record(self.path, self.cell, self.roster, self.receipt_hash, self.sources)

    def test_original_graph_replays_and_metrics_match(self):
        record, graph = self.check_record()
        self.assertEqual(len(graph), 50)
        self.assertEqual(record['metrics']['density'], graph.number_of_edges() / 1225)

    def test_changed_receipt_and_png_are_rejected(self):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'Receipt differs'):
            self.check_record()
        self.receipt_hash = viewer.sha256(self.path)
        self.path.with_suffix('.png').write_bytes(b'not the original PNG')
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            self.check_record()

    def test_changed_contract_is_rejected(self):
        self.cell = {**self.cell, 'language': 'portuguese'}
        with self.assertRaisesRegex(ValueError, 'frozen calibration contract'):
            self.check_record()

    def test_consistently_rehashed_but_wrong_metrics_are_rejected(self):
        record = json.loads(self.path.read_text(encoding='utf-8'))
        record['metrics']['density'] = 1
        self.path.write_text(json.dumps(record), encoding='utf-8')
        self.receipt_hash = viewer.sha256(self.path)
        with self.assertRaisesRegex(ValueError, 'measurements differ'):
            self.check_record()


if __name__ == '__main__':
    unittest.main()
