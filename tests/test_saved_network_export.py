"""Final-gallery adjacency parsing retains isolates and rejects invalid evidence."""
import unittest
from scripts.export_saved_networks import read_edges


class SavedNetworkExportTests(unittest.TestCase):
    def test_edges_and_isolates(self):
        self.assertEqual(read_edges(b"# header\n0 1\n1\n2\n", {"0", "1", "2"}), [["0", "1"]])
        self.assertEqual(read_edges(b"0\n1\n2\n", {"0", "1", "2"}), [])

    def test_rejects_missing_duplicate_unknown_and_self_edges(self):
        for raw in [b"0 1\n1\n", b"0 1\n1 0\n2\n", b"0 0\n1\n2\n",
                    b"0 3\n1\n2\n", b"0\n0\n1\n2\n"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                read_edges(raw, {"0", "1", "2"})
