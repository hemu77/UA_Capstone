// Check the exact deployment output, not just a development manifest.
import {readFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {validateData,replayFrames} from './graph-data.js';
const root=new URL('./dist/',import.meta.url);
const data=validateData(JSON.parse(await readFile(new URL('data/revised-calibration.json',root),'utf8')));
assert.equal(data.runs.length,104);assert.equal(data.main_authorized,false);
const files=await readdir(root,{recursive:true});
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
