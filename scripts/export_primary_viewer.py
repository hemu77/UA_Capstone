"""Export verified final-study networks and whitelisted recorded events. No provider calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_figures.deep_review import PRIMARY, PINS, ROSTER, v2
from analysis_primary.graph_evaluation import _read_graph

def checked_record(base,row):
    path=(base/row.record_path).resolve()
    if not path.is_relative_to(base): raise ValueError('Record path escaped evidence root')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row.record_sha256: raise ValueError('Record hash mismatch: '+row.run_id)
    record=json.loads(raw)
    if record['cell']['run_id']!=row.run_id: raise ValueError('Record identity mismatch')
    return record

def clean_events(record,graph):
    known=set(graph); current=set(); events=[]
    key=lambda e:tuple(sorted(e))
    for e in record['events']:
        actor=e['persona']
        if actor is not None and actor not in known: raise ValueError('Unknown actor')
        event={k:e[k] for k in ('method','persona','added','removed','attempts')}
        event['step']=len(events)+1
        for name in ('added','removed'):
            edges=event[name]
            if any(len(x)!=2 or x[0]==x[1] or not set(x)<=known for x in edges): raise ValueError('Invalid event edge')
            if len({key(x) for x in edges})!=len(edges): raise ValueError('Duplicate event edge')
        for edge in event['removed']:
            if key(edge) not in current: raise ValueError('Removing absent edge')
            current.remove(key(edge))
        for edge in event['added']:
            if key(edge) in current: raise ValueError('Adding duplicate edge')
            current.add(key(edge))
        events.append(event)
    if not events or current!={key(e) for e in graph.edges}: raise ValueError('Replay/final mismatch')
    return events

def export(base,dest):
    base=base.resolve()
    v2.verify_hashes({ROOT/PRIMARY:PINS[PRIMARY],ROOT/PRIMARY.with_name('primary_attempts.csv'):'ec24e7867a22d687d40d99253aa823a76bd506a734b6afb022ca359b64aac073'})
    all_rows=pd.read_csv(ROOT/PRIMARY.with_name('primary_attempts.csv'))
    catalog=pd.read_csv(ROOT/PRIMARY).set_index('run_id',drop=False,verify_integrity=True)
    roster=json.loads((ROOT/ROSTER).read_bytes())
    rid=hashlib.sha256(json.dumps(roster,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if not catalog.roster_sha256.eq(rid).all() or len(all_rows)!=896 or not all_rows.run_id.is_unique: raise ValueError('Catalog mismatch')
    runs=[]; failures=[]
    dest.mkdir(parents=True,exist_ok=True)
    event_dir=dest/'primary-events';event_dir.mkdir(exist_ok=True)
    for row in all_rows.itertuples():
        record=checked_record(base,row)
        meta={k:getattr(row,k) for k in ('run_id','model','method','culture','language','seed')}
        if not row.valid_graph:
            failures.append(dict(**meta,failure=row.failure))
            continue
        r=catalog.loc[row.run_id];g=_read_graph(ROOT,r)
        events=clean_events(record,g)
        event_name=row.run_id+'.json'
        raw=json.dumps(events,separators=(',',':'),allow_nan=False).encode()
        (event_dir/event_name).write_bytes(raw)
        metric_keys=['density','avg_clustering_coef','prop_nodes_lcc','global_efficiency','num_components','modularity','radius_lcc','diameter_lcc','avg_shortest_path_lcc','degree_one_share','isolate_share','degree_gini','coleman_gender','coleman_religion','coleman_political orientation','age_assortativity']
        metrics={k:float(r[k]) if pd.notna(r[k]) else None for k in metric_keys}
        runs.append(dict(**meta,study='primary',status='valid',roster_id=rid,prompt_variant='frozen060d_reference',
                         source=row.adj_path,source_sha256=r.adj_sha256,record_sha256=row.record_sha256,
                         protocol_sha256='060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f',
                         metrics=metrics,homophily={k:metrics.get('coleman_'+k) for k in ('gender','religion','political orientation')},
                         analysis_version='frozen primary; descriptive run metrics',edges=sorted([list(sorted(e)) for e in g.edges]),
                         events=[],events_url='./data/primary-events/'+event_name,events_sha256=hashlib.sha256(raw).hexdigest(),
                         replay_status='actual_recorded_graph_changes'))
    if len(runs)!=890 or len(failures)!=6 or sum(not r['edges'] for r in runs)!=12: raise ValueError('Primary totals differ')
    payload=dict(schema_version=1,dataset='primary',roster_id=rid,
                 personas=[dict(id=i,attributes=roster[i],position=[75*np.cos(2*np.pi*int(i)/50),0,75*np.sin(2*np.pi*int(i)/50)]) for i in sorted(roster,key=int)],
                 runs=runs,failures=failures,models=sorted(all_rows.model.unique()),methods=['global','local','sequential','iterative'],
                 settings=[['us','english'],['india','english'],['japan','english'],['brazil','english'],['us','hindi'],['us','japanese'],['us','portuguese']],
                 planned_networks=896,planned_repetitions=8,analysis_version='Final primary study',
                 evidence_notice='Completed primary study: 896 attempts; 890 valid networks, including 12 empty; six failures retained separately. Six selectively completed supplements are excluded.',
                 calibration_note='Eight planned seeds per condition. Individual layers are descriptive; only the separate 24-test family supplies adjusted inference.',
                 limitations='One designed fictional adult roster. Country framing is a prompt condition, not a representative national population. Marginal attribute mixing does not isolate causal preferences.',
                 verification='All primary adjacency and local record hashes verified; recorded added/removed edges reconstruct all 890 final graphs. Replay loads on demand. No raw prompts, responses or credentials exported.')
    (dest/'primary-viewer.json').write_bytes(json.dumps(payload,separators=(',',':'),allow_nan=False).encode())
    print('Exported 890 verified graphs, 890 lazy event logs, six failures; zero supplements or provider calls.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path,required=True)
    args=parser.parse_args()
    export(args.evidence_root,ROOT/'viewer/public/data')
