"""Export only anonymous node IDs, final edges and study condition labels for the public demo."""
import json
from pathlib import Path
import sys

import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_figures.deep_review import PRIMARY, PINS, v2
from analysis_primary.graph_evaluation import _read_graph

def export():
    attempts=PRIMARY.with_name('primary_attempts.csv')
    v2.verify_hashes({ROOT/PRIMARY:PINS[PRIMARY],ROOT/attempts:'ec24e7867a22d687d40d99253aa823a76bd506a734b6afb022ca359b64aac073'})
    catalog=pd.read_csv(ROOT/PRIMARY).set_index('run_id',drop=False,verify_integrity=True)
    all_rows=pd.read_csv(ROOT/attempts)
    assert len(all_rows)==896 and all_rows.run_id.is_unique and all_rows.cohort.eq('primary').all()
    assert set(all_rows[all_rows.valid_graph].run_id)==set(catalog.index)
    records=[]
    for r in all_rows.itertuples():
        edges=None
        if r.valid_graph:
            g=_read_graph(ROOT,catalog.loc[r.run_id])
            edges=sorted(min(int(a),int(b))*50+max(int(a),int(b)) for a,b in g.edges)
        records.append(dict(id=r.run_id,model=r.model,method=r.method,culture=r.culture,language=r.language,
                            seed=r.seed,edges=edges,failure=None if r.valid_graph else r.failure))
    assert sum(r['edges'] is None for r in records)==6 and sum(r['edges']==[] for r in records)==12
    out=ROOT/'viewer/public/data/primary-study.json'
    out.write_bytes(json.dumps(records,separators=(',',':'),allow_nan=False).encode())
    print(f'{len(records)} primary attempts, {len(catalog)} verified graphs; {out.stat().st_size:,} public bytes')

if __name__=='__main__':
    export()
