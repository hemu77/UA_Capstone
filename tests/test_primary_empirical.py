import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_primary.empirical import REFERENCES, ks_distance, read_reference, run_empirical_evaluation, source_fingerprints
from analysis_primary.core import load_inputs


class EmpiricalTests(unittest.TestCase):
    def test_cross_checkout_fingerprints_preserve_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ratings'
            lf = b'1 2 +1\n2 3 +1\n'
            path.write_bytes(lf)
            first = source_fingerprints(path)
            crlf = lf.replace(b'\n', b'\r\n')
            path.write_bytes(crlf)
            second = source_fingerprints(path)
            self.assertNotEqual(first['source_sha256'], second['source_sha256'])
            self.assertEqual(first['source_lf_sha256'], second['source_lf_sha256'])
            self.assertEqual(path.read_bytes(), crlf)

    def test_all_published_size_anchors(self):
        for name, file, selector, n, m, _ in REFERENCES:
            with self.subTest(reference=name):
                graph = read_reference(ROOT/'real_networks'/file, selector)
                self.assertEqual((len(graph), graph.number_of_edges()), (n, m))

    def test_positive_union_not_negative_or_self_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ratings'
            path.write_text('% signed\n1 2 +1\n2 1 -1\n1 1 2\n3 4 0\n')
            graph = read_reference(path, None)
            self.assertEqual(set(graph.edges), {('1', '2')})
            self.assertEqual(ks_distance([0, 1], [0, 1]), 0)
            self.assertEqual(ks_distance([0, 0], [1, 1]), 1)

    def test_primary_only_outputs(self):
        _, data, _ = load_inputs(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_empirical_evaluation(ROOT, data, Path(tmp))
            self.assertEqual(result['reference_networks'], 8)
            self.assertFalse(result['human_validation'])
            data.loc[0, 'cohort'] = 'supplemental'
            with self.assertRaises(ValueError):
                run_empirical_evaluation(ROOT, data, Path(tmp))


if __name__ == '__main__':
    unittest.main()
