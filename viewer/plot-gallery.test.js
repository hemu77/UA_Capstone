import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validatePlotCatalog,filterPlots,validateSupplements,supplementContext,SUPPLEMENT_COHORTS} from './plot-gallery.js';
import {loadPrimaryDataset} from './layers-data-all.js';
import {verifiedPlotProjection} from './plot-catalog-build.mjs';
const bytes=await readFile(new URL('./public/data/saved-plots.json',import.meta.url));
const catalog=JSON.parse(bytes);
test('Packaging pins the entire audited projection, not a self-reported CSV label',()=>{
  assert.equal(verifiedPlotProjection(bytes).length,896);
  for(const mutate of [
    d=>{d.plots[0].model='different-model';},
    d=>{d.plots[895].run_id+='a2';},
    d=>{d.plots[0].png_sha256='a'.repeat(64);}
  ]){
    const changed=structuredClone(catalog);mutate(changed);
    assert.throws(()=>verifiedPlotProjection(Buffer.from(JSON.stringify(changed))));
  }
});
test('All 896 plots preserve cohorts and twelve empty primary graphs',()=>{
  const plots=validatePlotCatalog(catalog);
  assert.equal(filterPlots(plots,'all','','').length,896);
  assert.equal(filterPlots(plots,'primary','','').length,890);
  assert.equal(filterPlots(plots,'supplement','','').length,6);
  assert.equal(filterPlots(plots,'primary','global','').length,219);
  assert.equal(filterPlots(plots,'supplement','local','').length,0);
  assert.ok(filterPlots(plots,'primary','','GPT-4.1 english 44000').every(p=>p.model==='gpt-4.1'&&p.language==='english'&&p.seed===44000));
  assert.equal(filterPlots(plots,'all','','no-such-run').length,0);
});
test('Reject duplicate identities, cohort mixing, paths, hashes and inconsistent metadata',()=>{
  for(const mutate of [
    d=>d.plots.pop(),
    d=>{d.plots[1]=d.plots[0];},
    d=>{d.plots[0].cohort='supplement_initial';},
    d=>{d.plots[0].png_url='https://example.com/tracker.png';},
    d=>{d.plots[0].source_path='../private.png';},
    d=>{d.plots[0].png_sha256='bad';},
    d=>{d.plots[0].run_id='../../private';},
    d=>{d.plots[0].empty_graph=!d.plots[0].empty_graph;},
    d=>{d.catalog_sha256='bad';},
    d=>{d.raw_response='private fixture';},
    d=>{d.plots[0].raw_response='private fixture';}
  ]){const changed=structuredClone(catalog);mutate(changed);assert.throws(()=>validatePlotCatalog(changed));}
});

// Supplement-only panel: the six excluded graphs, checked so they can never pass as primary graphs.
const fetchImpl=async url=>{const name=String(url).replace(/^\.\/data\//,'').split('?')[0];try{const raw=await readFile(new URL('./public/data/'+name,import.meta.url));return {ok:true,status:200,json:async()=>JSON.parse(raw),arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)};}catch{return {ok:false,status:404};}};
const dataset=await loadPrimaryDataset({fetchImpl});
// Fixture in the supplements.json shape (SPEC.md C.3), derived from the pinned saved-networks / saved-plots fixtures.
const networks=JSON.parse(await readFile(new URL('./public/data/saved-networks.json',import.meta.url)));
const supplementFixture={schema_version:1,note:'Excluded from the primary study',graphs:catalog.plots.filter(p=>p.cohort!=='primary').map(p=>{
  const g=networks.graphs.find(n=>n.run_id===p.run_id);
  return {run_id:p.run_id,cohort:p.cohort,model:p.model,method:p.method,culture:p.culture,language:p.language,seed:p.seed,edges:g.edges,source:g.source,source_sha256:g.source_sha256};
})};
test('six supplements validate, keep their ties exactly and never collide with a primary run',()=>{
  const graphs=validateSupplements(supplementFixture,dataset);
  assert.equal(graphs.length,6);
  assert.deepEqual(graphs.map(g=>g.edges.length),supplementFixture.graphs.map(g=>g.edges.length));
  assert.equal(graphs.filter(g=>!g.edges.length).length,1);
  assert.deepEqual(new Set(graphs.map(g=>g.cohort)),new Set(Object.keys(SUPPLEMENT_COHORTS)));
  const primary=new Set(dataset.runs.map(r=>r.run_id));
  assert.ok(graphs.every(g=>!primary.has(g.run_id)));
  assert.ok(graphs.every(g=>g.edges.every(([a,b])=>Number(a)<Number(b))));
  assert.deepEqual(validateSupplements(supplementFixture.graphs,dataset).map(g=>g.run_id),graphs.map(g=>g.run_id));
});
test('supplement validation rejects leaks, foreign personas and malformed ties',()=>{
  const primaryRun=dataset.runs[0];
  for(const mutate of [
    d=>d.graphs.pop(),
    d=>d.graphs.push(structuredClone(d.graphs[0])),
    d=>{d.graphs[1].run_id=d.graphs[0].run_id;},
    d=>{d.graphs[0].run_id=primaryRun.run_id;},
    d=>{d.graphs[0].run_id=dataset.failures[0].run_id;},
    d=>{d.graphs[0].cohort='primary';},
    d=>{d.graphs[0].source='outputs/final_study/060dcd3a86d5/main/runs/'+d.graphs[0].run_id+'.adj';},
    d=>{d.graphs[0].edges.push(['0','99']);},
    d=>{d.graphs[0].edges.push(['3','3']);},
    d=>{d.graphs[0].edges.push([...d.graphs[0].edges[0]].reverse());},
    d=>{d.graphs[0].source_sha256='bad';},
    d=>{d.graphs[0].run_id='../../private';},
    d=>{d.graphs=null;}
  ]){const changed=structuredClone(supplementFixture);mutate(changed);assert.throws(()=>validateSupplements(changed,dataset));}
});
test('supplement context comes from the loaded slots and never fills a failed slot',()=>{
  const graphs=validateSupplements(supplementFixture,dataset);
  const contexts=graphs.map(g=>supplementContext(g,dataset));
  for(const [i,context] of contexts.entries()){
    assert.ok(Number.isInteger(context.slot));
    assert.equal(context.status,dataset.slots[context.slot].status);
    assert.notEqual(dataset.slots[context.slot].run_id,graphs[i].run_id);
  }
  const failed=contexts.filter(c=>c.status==='failed');
  assert.ok(failed.length>=1);
  assert.ok(failed.every(c=>/does not fill that slot/.test(c.text)));
  assert.equal(supplementContext({...graphs[0],model:'unknown'},dataset).status,'outside');
});
