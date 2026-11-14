import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validatePlotCatalog,filterPlots} from './plot-gallery.js';
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
