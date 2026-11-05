import hashlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).resolve().parents[1] / 'scripts/prepare_study_data.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Preparation(unittest.TestCase):
    def package(self, folder, name, entries, *, manifest_data=None, duplicate=False, symlink=False):
        archive = folder / name
        manifest = {'asset':name, 'entries':{}}
        with zipfile.ZipFile(archive, 'w') as z:
            for path, data in entries.items():
                info = zipfile.ZipInfo(path)
                if symlink:
                    info.create_system, info.external_attr = 3, 0o120777 << 16
                z.writestr(info, data)
                expected = data if manifest_data is None else manifest_data
                manifest['entries'][path] = {'bytes':len(expected),'sha256':hashlib.sha256(expected).hexdigest()}
                if duplicate:
                    z.writestr(info, data)
        manifest['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest['archive_bytes'] = archive.stat().st_size
        return archive, manifest

    def test_unsafe_zip_names_are_rejected(self):
        self.assertEqual(m.checked_name('outputs/supplement_20261105_v1/runs/a.json'), 'outputs/supplement_20261105_v1/runs/a.json')
        for name in ('../x.json','/x.json','C:/x.json','a\\x.json','outputs//x.json',
                     'outputs/supplement_20261105_v1/transport/a.json',
                     'outputs/supplement_20261105_v1/authorization.json',
                     'outputs/supplement_20261105_v1/runs/CON.json',
                     'outputs/supplement_20261105_v1/runs/a.json.',
                     'unrelated/a.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.checked_name(name)

    def test_extract_missing_and_leave_matching_files_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); root=folder/'root'; root.mkdir()
            name='outputs/supplement_20261105_v1/runs/a.json'
            package=self.package(folder,'ok.zip',{name:b'{}'})
            report=m.install_packages(root,[package])
            self.assertEqual(report['installed'],1)
            target=root/name; before=target.stat().st_mtime_ns
            report=m.install_packages(root,[package])
            self.assertEqual(report['installed'],0)
            self.assertEqual(target.stat().st_mtime_ns,before)

    def test_conflict_in_later_package_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); root=folder/'root'; root.mkdir()
            a='outputs/supplement_20261105_v1/runs/a.json'
            b='outputs/supplement_20261105_v1/runs/b.json'
            first=self.package(folder,'a.zip',{a:b'{}'})
            second=self.package(folder,'b.zip',{b:b'{}'})
            (root/b).parent.mkdir(parents=True); (root/b).write_bytes(b'old')
            with self.assertRaises(ValueError): m.install_packages(root,[first,second])
            self.assertFalse((root/a).exists())
            self.assertEqual((root/b).read_bytes(),b'old')

    def test_bad_member_hash_duplicate_and_symlink_fail_before_write(self):
        for options in ({'manifest_data':b'xx'}, {'duplicate':True}, {'symlink':True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temp:
                folder=Path(temp); root=folder/'root'; root.mkdir()
                package=self.package(folder,'bad.zip',{'outputs/supplement_20261105_v1/runs/a.json':b'{}'},**options)
                with self.assertRaises(ValueError): m.install_packages(root,[package])
                self.assertEqual(list(root.iterdir()),[])

    def test_case_collisions_fail_before_write(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); root=folder/'root'; root.mkdir()
            prefix='outputs/supplement_20261105_v1/runs/'
            package=self.package(folder,'bad.zip',{prefix+'a.json':b'{}',prefix+'A.json':b'{}'})
            with self.assertRaises(ValueError): m.install_packages(root,[package])

    def test_primary_keeps_failures_and_supplements_never_become_primary(self):
        rows=[{'run_id':'p1','cohort':'primary','valid_graph':True,'selected_for_896_view':True},
              {'run_id':'p2','cohort':'primary','valid_graph':False,'selected_for_896_view':False},
              {'run_id':'s1','cohort':'supplement_initial','valid_graph':True,'selected_for_896_view':True},
              {'run_id':'s2','cohort':'supplement_completion','valid_graph':False,'selected_for_896_view':False}]
        views=m.analysis_views(rows)
        self.assertEqual([r['run_id'] for r in views['primary_attempts.csv']],['p1','p2'])
        self.assertEqual([r['run_id'] for r in views['all_valid_graphs_descriptive_only.csv']],['p1','s1'])
        self.assertEqual([r['run_id'] for r in views['failures.csv']],['p2','s2'])
        self.assertEqual(len(views['supplemental_attempts.csv']),2)

    def test_binary_graph_loader_accepts_valid_empty_fifty_node_graph(self):
        data=('# empty graph\n'+'\n'.join(str(i) for i in range(50))+'\n').encode()
        graph=m.graph_from_bytes(data)
        self.assertEqual(set(graph),{str(i) for i in range(50)})
        self.assertEqual(graph.number_of_edges(),0)

    def test_undefined_metric_is_blank_not_zero(self):
        data=m.csv_bytes([{'run_id':'failed','density':None},{'run_id':'empty','density':0.0}])
        rows=list(csv.DictReader(io.StringIO(data.decode())))
        self.assertEqual(rows[0]['density'],'')
        self.assertEqual(rows[1]['density'],'0.0')

    def test_existing_redirected_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaisesRegex(ValueError,'Redirected'):
                m.destination(Path(temp).resolve(),'outputs/supplement_20261105_v1/runs/a.json')

if __name__=='__main__': unittest.main()
