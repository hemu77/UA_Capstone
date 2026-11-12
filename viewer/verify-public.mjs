// Check the exact deployment output, not just a development manifest.
import {readFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {validateData,replayFrames} from './graph-data.js';
const root=new URL('./dist/',import.meta.url);
const data=validateData(JSON.parse(await readFile(new URL('data/revised-calibration.json',root),'utf8')));
assert.equal(data.runs.length,104);assert.equal(data.main_authorized,false);
const primary=validateData(JSON.parse(await readFile(new URL('data/primary-viewer.json',root),'utf8')));
assert.equal(primary.runs.length,890);
assert.equal(primary.failures.length,6);
assert.equal(new Set([...primary.runs,...primary.failures].map(r=>r.run_id)).size,896);
assert.ok(primary.failures.every(r=>!('edges' in r)));
assert.equal(primary.runs.filter(r=>r.edges.length===0).length,12);
for(const r of primary.runs){
  assert.match(r.run_id,/^final_060dcd3a86d5_main_[a-zA-Z0-9._-]+$/);
  assert.equal(r.events_url,`./data/primary-events/${r.run_id}.json`);
  const bytes=await readFile(new URL(r.events_url,root));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),r.events_sha256);
  const events=JSON.parse(bytes);
  assert.ok(Array.isArray(events)&&events.length>0,'A valid empty graph still requires its recorded no-change event');
  for(const e of events)assert.deepEqual(Object.keys(e).sort(),['added','attempts','method','persona','removed','step']);
  assert.ok(replayFrames({...r,events},primary.personas.map(p=>p.id)),'Missing replay');
}
const files=(await readdir(root,{recursive:true})).map(f=>f.replaceAll('\\\\','/'));
const expected=primary.runs.map(r=>r.events_url.slice(2)).sort();
assert.deepEqual(files.filter(f=>f.startsWith('data/primary-events/')&&f.endsWith('.json')).sort(),expected,'Orphan event exports');
for(const path of ['index.html','layers.html','study.html','layers.js','layers.css','graph-data.js','formation-charts.js','navigation.js','research-plots.js','vendor/three.module.js','vendor/three.core.js','vendor/addons/controls/OrbitControls.js','data/study-figures/14_demographic_mixing.png','data/study-figures/15_efficiency_density.png'])assert.ok((await readFile(new URL(path,root))).length>0,path);
assert.equal(await readFile(new URL('index.html',root),'utf8'),await readFile(new URL('layers.html',root),'utf8'));
assert.equal(await readFile(new URL('study.html',root),'utf8'),await readFile(new URL('layers.html',root),'utf8'));
const blocked=/(^|[/\\])(?:\.env[^/\\]*|api-key\.txt|authorization\.json|requests|ledger|billing)([/\\.]|$)/i;
for(const file of files)assert.ok(!blocked.test(file),`Private path in public build: ${file}`);
for(const file of files.filter(name=>/\.(json|js|html|css|adj|svg|txt)$/.test(name))){
  const text=await readFile(new URL(file.replaceAll('\\','/'),root),'utf8');
  assert.ok(!/sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}/.test(text),`Credential pattern in ${file}`);
}
for(const run of data.runs){
  assert.equal(run.study,'revised_calibration');
  for(const key of ['requests','decisions','responses','raw_response','billing','authorization'])assert.ok(!(key in run));
  replayFrames(run,data.personas.map(person=>person.id));
  for(const [path,digest] of [[run.png_url,run.png_sha256],[run.adjacency_url,run.source_sha256]]){
    assert.match(path,/^\.\/data\/revised-calibration\/[a-zA-Z0-9_.-]+\.(png|adj)$/);
    assert.equal(createHash('sha256').update(await readFile(new URL(path,root))).digest('hex'),digest);
  }
}
console.log(JSON.stringify({status:'PASS',graphs:data.runs.length,verified_artifacts:data.runs.length*2,credential_patterns:0,paid_calls:0}));
