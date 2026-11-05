import importlib.util
from pathlib import Path
import unittest
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('network_catalog',ROOT/'scripts/build_network_catalog.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CatalogTests(unittest.TestCase):
    def test_paths_cannot_escape(self):
        for name in ('../secret','..\\secret','C:\\secret','/secret'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.asset(ROOT,name)

    def test_published_inventory_counts(self):
        import csv
        with (ROOT/'outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv').open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),896)
        self.assertEqual(sum(r['cohort']=='primary' for r in rows),890)
        self.assertEqual(sum(r['cohort']!='primary' for r in rows),6)

    def test_refuse_different_destination_before_writing_index(self):
        import csv
        with (ROOT/'outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv').open(newline='') as stream:
            first=next(csv.DictReader(stream))
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)
            path=destination/first['adj_path']
            path.parent.mkdir(parents=True)
            path.write_bytes(b'changed destination')
            with self.assertRaisesRegex(ValueError,'Refuse differing destination'):
                module.catalog(ROOT,destination)
            self.assertEqual(path.read_bytes(),b'changed destination')
            self.assertFalse((destination/'NETWORKS.md').exists())


if __name__=='__main__': unittest.main()
