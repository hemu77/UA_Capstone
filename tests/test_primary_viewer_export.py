"""Offline privacy and replay boundary checks; no provider or database access."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import networkx as nx
from scripts.export_primary_viewer import checked_record,clean_events

class PrimaryViewerExportTests(unittest.TestCase):
    def event(self,**changes):
        e=dict(method='global',persona=None,added=[['0','1']],removed=[],attempts=1)
        return dict(e,**changes)
    def graph(self):
        g=nx.Graph();g.add_nodes_from(['0','1','2']);g.add_edge('0','1');return g
    def test_whitelist_reconstructs_and_discards_private_fields(self):
        e=self.event();e['raw_response']='private fixture'
        result=clean_events({'events':[e]},self.graph())
        self.assertEqual(set(result[0]),{'step','method','persona','added','removed','attempts'})
    def test_invalid_replay_is_refused(self):
        for changes in [dict(persona='unknown'),dict(added=[['0','9']]),dict(added=[['0','0']]),
                        dict(added=[['0','1'],['1','0']]),dict(removed=[['0','1']]),dict(added=[])]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                clean_events({'events':[self.event(**changes)]},self.graph())
        with self.assertRaises(ValueError):clean_events({'events':[]},self.graph())
    def test_valid_empty_is_not_absent_history(self):
        g=nx.empty_graph(['0','1','2'])
        self.assertEqual(len(clean_events({'events':[self.event(added=[])]},g)),1)
    def test_record_escape_hash_and_identity_checks(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();p=base/'record.json'
            raw=json.dumps({'cell':{'run_id':'a'}}).encode();p.write_bytes(raw)
            row=SimpleNamespace(record_path='record.json',record_sha256=hashlib.sha256(raw).hexdigest(),run_id='a')
            self.assertEqual(checked_record(base,row)['cell']['run_id'],'a')
            row.run_id='b'
            with self.assertRaises(ValueError):checked_record(base,row)
            row.record_sha256='bad'
            with self.assertRaises(ValueError):checked_record(base,row)
            row.record_path='../escape.json'
            with self.assertRaises(ValueError):checked_record(base,row)
