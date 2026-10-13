"""Public derivatives must preserve inspected evidence without publishing API logs."""
import json
import unittest
from unittest.mock import patch
import networkx as nx
import export_revised_viewer as viewer


class RevisedViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = viewer.calibration.contract()
        cls.target = viewer.calibration.folder(cls.spec)
        cls.roster = viewer.calibration.revised.previous.adult_roster()

    def test_complete_inspection_matches_every_receipt(self):
        report = viewer.checked_inspection(self.target, self.spec)
        self.assertEqual(report['verified_networks'], 104)

    def test_stale_inspector_or_tampered_receipt_fails_closed(self):
        original = viewer.calibration.sha
        for suffix in ['inspect_calibration_v6.py', '.json']:
            with self.subTest(suffix=suffix), patch.object(viewer.calibration, 'sha',
                    side_effect=lambda p: 'changed' if str(p).endswith(suffix) else original(p)):
                with self.assertRaises(ValueError):
                    viewer.checked_inspection(self.target, self.spec)

    def test_public_empty_graph_preserves_null_and_allowlists_events(self):
        for cell in viewer.calibration.cells(self.spec):
            if cell['method'] != 'global':
                continue
            path = self.target / 'runs' / (cell['run_id'] + '.json')
            graph = nx.read_adjlist(path.with_suffix('.adj'))
            if graph.number_of_edges():
                continue
            record = viewer.calibration.verify_receipt(path, cell)
            run = viewer.public_run(path, cell, record, graph, self.spec, self.roster)
            self.assertEqual(run['edges'], [])
            self.assertIsNone(run['age_assortativity'])
            self.assertEqual(run['source_sha256'], record['artifacts']['.adj'])
            self.assertNotIn('requests', run)
            self.assertNotIn('decisions', run)
            self.assertEqual(set(run['events'][0]), {'step','method','persona','added','removed','attempts'})
            json.dumps(run, allow_nan=False)
            return
        self.fail('Expected a saved empty global graph in this calibration.')
