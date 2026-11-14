import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validatePlotCatalog} from './plot-gallery.js';
import {validateNetworkCatalog} from './network-catalog.js';
const read=name=>JSON.parse(readFileSync(new URL('./public/data/'+name,import.meta.url)));
const catalog=read('saved-networks.json'),primary=read('primary-viewer.json'),plots=validatePlotCatalog(read('saved-plots.json'));
test('896 final graphs preserve primary parity, empty nodes and separate supplements',()=>{
  const graphs=validateNetworkCatalog(catalog,plots,primary);
  assert.equal(graphs.size,896);
  for(const run of primary.runs)assert.deepEqual(graphs.get(run.run_id).edges,run.edges);
  assert.equal(plots.filter(p=>!primary.runs.some(r=>r.run_id===p.run_id)).length,6);
  for(const method of ['global','local','sequential','iterative'])assert.equal(plots.filter(p=>p.method===method).length,224);
  assert.equal(plots.filter(p=>p.cohort==='primary'&&!graphs.get(p.run_id).edges.length).length,12);
  assert.equal(plots.filter(p=>p.cohort!=='primary'&&!graphs.get(p.run_id).edges.length).length,1);
});
test('reject missing, duplicate, private, wrong-roster and invalid-edge gallery data',()=>{
  for(const mutate of [
    d=>d.graphs.pop(),d=>{d.graphs[1]=d.graphs[0];},d=>{d.raw_response='not public';},
    d=>{d.roster_id='other';},d=>{d.graphs[0].source='../private';},
    d=>{d.graphs[0].source_sha256='bad';},d=>{d.graphs[0].edges[0]=['0','0'];},
    d=>{d.graphs[0].edges[0]=['0','999'];},d=>{d.graphs[0].edges[1]=d.graphs[0].edges[0];}
  ]){const data=structuredClone(catalog);mutate(data);assert.throws(()=>validateNetworkCatalog(data,plots,primary));}
});
