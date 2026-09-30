import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateData, differences, replayEdges, neighbors, comparisonLayout, layerComparison, layerCandidates, applyLayerSelection, replayFrames, personaSteps} from './graph-data.js';
import {filterRuns, conditionRows, coverage, measurement, topologySvg, homophilySvg, selectionCsv} from './research-plots.js';
import {trajectorySvg, metricBars, researchEvidence, homophilyMatrix, runLabel} from './formation-charts.js';
const data = JSON.parse(readFileSync(new URL('./public/data/networks.json', import.meta.url)));
import {personaView, personaTag, edgeFrequencies, personaComparison} from './graph-data.js';
import {loadPresentation,frequencySvg} from './formation-charts.js';
import {comparisonModels} from './graph-data.js';
import {togglePersona,tapTracker} from './graph-data.js';
test('persona clicks toggle while drags, pinches and cancelled gestures never select',()=>{
  assert.equal(togglePersona('13','13'),'');assert.equal(togglePersona('13','9'),'9');
  const tap=tapTracker(),e={pointerId:1,button:0,isPrimary:true,clientX:10,clientY:10};
  tap.down(e);assert.equal(tap.up(e),true);assert.equal(tap.up(e),false);
  tap.down(e);tap.move({...e,clientX:40});assert.equal(tap.up(e),false);
  tap.down(e);tap.cancel();assert.equal(tap.up(e),false);
  tap.down(e);tap.down({...e,pointerId:2,isPrimary:false});assert.equal(tap.up(e),false);
  tap.down({...e,button:2});assert.equal(tap.up(e),false);
});

test('revised comparison never relabels or substitutes missing model results',()=>{
  assert.deepEqual(comparisonModels,['gpt-5.6-luna','gpt-6-luna','gpt-6-sol']);
  const candidates=layerCandidates(data.runs,{collection:'revised'});
  assert.ok(candidates.length>0);
  assert.ok(candidates.every(run=>comparisonModels.includes(run.model)));
  assert.ok(candidates.every(run=>data.runs.includes(run)));
  assert.equal(layerCandidates(data.runs,{collection:'revised',model:'gpt-4.1-mini'}).length,0);
  assert.ok(layerCandidates(data.runs,{collection:'engineering_pilot',model:'gpt-4.1-mini'}).length>0);
});

test('new Luna 5.6 exports cover four real full-roster method runs',()=>{
  const runs=layerCandidates(data.runs,{collection:'revised',model:'gpt-5.6-luna'});
  assert.equal(runs.length,4);
  assert.deepEqual(runs.map(r=>r.method).sort(),['global','iterative','local','sequential']);
  assert.ok(runs.every(r=>r.culture==='us'&&r.language==='english'&&r.seed===1000));
  assert.ok(runs.every(r=>r.source.includes('luna56-pilot')&&r.edges.length>0&&r.events.length>0));
  assert.equal(data.personas.length,50);
});
test('optional gallery failure cannot prevent network initialization',async()=>{
  for(const fetcher of [async()=>{throw new Error('offline');},async()=>({ok:false}),async()=>({ok:true,json:async()=>{throw new Error('invalid JSON');}}),async()=>({ok:true,json:async()=>({runs:[]})})]){
    const result=await loadPresentation(fetcher);assert.deepEqual(result.runs,{});assert.match(result.error,/remain usable/);
  }
  assert.deepEqual(await loadPresentation(async()=>({ok:true,json:async()=>({runs:{a:{}}})})),{runs:{a:{}},error:null});
});
test('persona isolation, actor labels and selected-run frequencies preserve evidence',()=>{
  const edges=[['1','2'],['2','3'],['1','4']];
  const view=personaView(edges,'1',true);
  assert.deepEqual(view.edges,[['1','2'],['1','4']]);
  assert.deepEqual([...view.visibleIds].sort(),['1','2','4']);
  assert.deepEqual(personaView(edges,'5',true).edges,[]);
  assert.deepEqual([...personaView(edges,'5',true).visibleIds],['5']);
  assert.equal(personaView(edges,'',true).edges,edges);
  assert.equal(personaTag('3','3','3',8),'Persona 3 / selected / actor at event 8');
  assert.ok(!personaTag('3','','3',8).includes('acting'));
  const frequencies=edgeFrequencies([{run_id:'a',roster_id:'r',edges:[['1','2']]},{run_id:'b',roster_id:'r',edges:[['2','1'],['2','3']]}]);
  assert.deepEqual(frequencies.map(r=>r.frequency),[1,.5]);
  assert.deepEqual(personaComparison([{run_id:'a',roster_id:'r',edges:[['1','2'],['1','3']]},{run_id:'b',roster_id:'r',edges:[['1','2'],['1','4']]}],'1'),{union:3,shared:1,jaccard:1/3});
  const svg=frequencySvg([{id:'1'},{id:'2'},{id:'3'}],[{run_id:'a',roster_id:'r',edges:[['1','2']]}],new Map([['1',[0,0]],['2',[1,1]],['3',[2,1]]]),'3');
  assert.match(svg,/0 distinct persona 3 incident/);assert.ok(!svg.includes('NaN'));
  const matrix=frequencySvg([{id:'1'},{id:'2'}],[{run_id:'a',roster_id:'r',edges:[['1','2']]}],new Map());
  assert.match(matrix,/Tie frequency matrix/);
  assert.equal((matrix.match(/tabindex="0"/g)||[]).length,2);
  assert.match(matrix,/Personas 1 and 2: 1\/1 selected runs/);
  assert.match(matrix,/<td>1 \/ 1<\/td>/);assert.match(matrix,/<td>Present<\/td>/);
  assert.ok(!matrix.includes('<line'));
  assert.throws(()=>edgeFrequencies([{run_id:'a',roster_id:'a',edges:[]},{run_id:'b',roster_id:'b',edges:[]}]),/same roster/);
});

test('research questions require matched conditions and never turn source drift into model effects',()=>{
  const a={model:'a',method:'local',culture:'us',language:'english',seed:1,roster_id:'r',study:'pilot',prompt_variant:'v',edges:[['1','2']],homophily:{gender:0},age_assortativity:null};
  const b={...a,model:'b',edges:[['2','1'],['2','3']]};
  assert.equal(researchEvidence([a,b],'rq3').matched,true);
  assert.equal(researchEvidence([a,b],'rq3').pairs[0].jaccard,.5);
  assert.equal(researchEvidence([a,b],'rq1').matched,false);
  assert.equal(researchEvidence([a,{...b,prompt_variant:'other'}],'rq3').matched,false);
  assert.equal(researchEvidence([a,{...a,language:'hindi',language_confound:true}],'rq4').matched,false);
  assert.equal(researchEvidence([a,{...a,culture:'india'}],'rq1').matched,true);
  assert.match(homophilyMatrix([a]),/0.000/);
  assert.match(homophilyMatrix([a]),/NA/);
  assert.match(runLabel(a),/United States/);
  assert.ok(!runLabel(a).includes('roster_id'));
});
test('formation charts distinguish missing histories, negative metrics and escaped labels', () => {
  assert.match(trajectorySvg(null,'',0),/No recorded history/);
  const svg=metricBars([{model:'<script>',method:'local',culture:'us',language:'english',seed:0,metrics:{modularity:-.2}},{model:'unknown',metrics:{modularity:null}}],'modularity','Modularity');
  assert.ok(!svg.includes('<script>'));
  assert.match(svg,/-0.200/);
  assert.match(svg,/>NA</);
  assert.ok(!/width="-/.test(svg));
});
test('formation frames and persona journeys preserve real additions, removals and no-op decisions', () => {
  const events=[{persona:'a',added:[['a','b']],removed:[]},{persona:'c',added:[['c','b']],removed:[]},{persona:'b',added:[],removed:[]},{persona:'a',added:[],removed:[['a','b']]}];
  const run={events,edges:[['c','b']]};
  const frames=replayFrames(run,['a','b','c']);
  assert.deepEqual(frames.map(f=>f.edges.length),[0,1,2,2,1]);
  assert.deepEqual(personaSteps(events,'b'),[1,2,3,4]);
  assert.deepEqual(personaSteps(events,'c'),[2]);
  assert.deepEqual(personaSteps(events,'missing'),[]);
  assert.equal(replayFrames({events:null,edges:[]},[]),null);
  assert.throws(()=>replayFrames({...run,edges:[]},['a','b','c']),/final network/);
  assert.throws(()=>replayFrames(run,['a','b']),/Unknown replay/);
  for(const pilot of data.runs.filter(r=>r.study==='engineering_pilot')){
    const recorded=replayFrames(pilot,data.personas.map(p=>p.id));
    assert.equal(recorded.length,pilot.events.length+1);
    assert.equal(recorded.at(-1).edges.length,pilot.edges.length);
  }
  const global=data.runs.find(r=>r.study==='engineering_pilot'&&r.method==='global');
  assert.equal(replayFrames(global,data.personas.map(p=>p.id)).length,2);
});
test('applying filters selects exact saved runs without truncation or mutating current layers', () => {
  const filters={collection:'revised',method:'sequential',culture:'us',language:'english',model:'',seed:'1000'};
  const found=layerCandidates(data.runs,filters);
  assert.equal(found.length,3);
  const applied=applyLayerSelection(found);
  assert.notEqual(applied,found);
  const removed=applied.splice(1,1)[0];
  const recoverable=layerCandidates(data.runs,filters).find(r=>r.run_id===removed.run_id);
  assert.equal(recoverable,removed);
  assert.equal(applyLayerSelection([...applied,recoverable]).length,3);
  assert.throws(()=>applyLayerSelection(layerCandidates(data.runs,{collection:'engineering_pilot'})),/one and six/);
  assert.throws(()=>applyLayerSelection(layerCandidates(data.runs,{...filters,culture:'japan'})),/one and six/);
  assert.equal(applied.length,2);
  assert.equal(layerCandidates(data.runs,{...filters,seed:'0'}).length,0);
  assert.equal(layerCandidates(data.runs,{collection:'historical',model:'gpt-6-sol'}).length,0);
});
test('layer comparisons count undirected ties and reject incompatible identities', () => {
  const a={run_id:'a',roster_id:'same',edges:[['1','2'],['2','3']]};
  const b={run_id:'b',roster_id:'same',edges:[['2','1'],['3','4']]};
  assert.equal(layerComparison([a,b]).shared,1);
  assert.equal(layerComparison([a,b]).union,3);
  assert.equal(layerComparison([a]).shared,2);
  assert.throws(()=>layerComparison([]),/one and six/);
  assert.throws(()=>layerComparison([a,a]),/Duplicate/);
  assert.throws(()=>layerComparison([a,{...b,roster_id:'different'}]),/same roster/);
  assert.throws(()=>layerComparison(Array.from({length:7},(_,i)=>({...a,run_id:String(i)}))),/one and six/);
});

test('multi-select filters use real intersections and never silently change conditions', () => {
  const choice = {study: 'engineering_pilot', model: ['gpt-6-luna'], method: ['local', 'sequential'], culture: ['us'], language: ['english', 'japanese'], seed: [1000]};
  const rows = filterRuns(data.runs, choice);
  assert.equal(rows.length, 4);
  assert.deepEqual(coverage(data.runs, choice), {expected: 4, observed: 4, missing: 0});
  choice.culture = ['japan'];
  assert.equal(filterRuns(data.runs, choice).length, 0);
  assert.deepEqual(coverage(data.runs, choice), {expected: 4, observed: 0, missing: 4});
  choice.model = [];
  assert.equal(coverage(data.runs, choice).expected, 0);
});

test('charts preserve prompt/roster/study boundaries and undefined values', () => {
  const first = structuredClone(data.runs.find(run => run.model === 'gpt-6-luna'));
  const second = {...first, run_id: 'second', seed: 1001};
  const variant = {...first, prompt_variant: 'other'};
  assert.equal(conditionRows([first, second]).length, 1);
  assert.equal(conditionRows([first, variant]).length, 2);
  assert.equal(conditionRows([first, {...first, roster_id: 'other'}]).length, 2);
  assert.equal(conditionRows([first, {...first, study: 'language'}]).length, 2);
  first.homophily.gender = null;
  first.age_assortativity = 0;
  assert.equal(measurement(first, 'gender'), null);
  assert.equal(measurement(first, 'age'), 0);
  first.culture = '<script>bad</script>';
  const rows = conditionRows([first]);
  assert.ok(topologySvg(rows, 'density', 'Pilot').includes('n=1/1'));
  assert.ok(homophilySvg(rows, 'Pilot').includes('NA (0)'));
  assert.ok(!topologySvg(rows, 'density', 'Pilot').includes('<script>'));
  assert.ok(selectionCsv([first]).includes(first.source_sha256));
  assert.ok(topologySvg([], 'density', '').includes('No saved runs'));
});
test('union layout is deterministic, finite and independent of pane order', () => {
  const a=data.runs[0].edges,b=data.runs[1].edges;
  const first=comparisonLayout(data.personas,a,b);
  const second=comparisonLayout(data.personas,b,a);
  for(const person of data.personas){
    assert.ok(first.get(person.id).every(Number.isFinite));
    first.get(person.id).forEach((v,k)=>assert.ok(Math.abs(v-second.get(person.id)[k])<1e-8));
  }
  assert.throws(()=>comparisonLayout(data.personas,[['invalid','other']],[]),/Invalid layout edge/);
});
test('all exported historical graphs satisfy the display contract', () => {
  validateData(data);
  assert.equal(data.personas.length, 50);
  const historical = data.runs.filter(run => run.status === 'historical_reanalyzed');
  assert.equal(historical.length + data.quarantined_runs.length, 192);
  assert.equal(historical.length, 176);
  for (const run of historical) {
    assert.equal(run.events, null);
    assert.equal(run.metrics.density, run.edges.length / 1225);
  }
});
test('engineering pilots are distinct and their actual events reproduce the graph', () => {
  const pilots = data.runs.filter(run => run.study === 'engineering_pilot');
  assert.equal(pilots.length, data.engineering_pilots || 0);
  for (const run of pilots) {
    assert.equal(run.status, 'ENGINEERING_PILOT_NOT_CONFIRMATORY');
    const canonical = edges => edges.map(edge => JSON.stringify([...edge].sort())).sort();
    assert.deepEqual(canonical(replayEdges(run.events, run.events.length)), canonical(run.edges));
    assert.equal(run.metrics.density, run.edges.length / 1225);
  }
});

test('Sol coverage contains four real US-English pilot methods, not renamed Luna graphs', () => {
  const sol = data.runs.filter(run => run.model === 'gpt-6-sol');
  assert.equal(sol.length, 4);
  assert.deepEqual(new Set(sol.map(run => run.method)), new Set(['global', 'local', 'sequential', 'iterative']));
  assert.ok(sol.every(run => run.study === 'engineering_pilot' && run.language === 'english' && run.culture === 'us'));
  assert.equal(new Set(sol.map(run => run.source_sha256)).size, 4);
});
test('undirected comparisons are orientation independent and require a matching roster', () => {
  const a = {roster_id: 'same', edges: [['a', 'b']]};
  assert.equal(differences(a, {...a, edges: [['b', 'a']]}).shared.size, 1);
  assert.equal(differences(a, {...a, roster_id: 'other'}), null);
  assert.deepEqual([...neighbors(a.edges, 'a')], ['b']);
});
test('invalid edges are rejected instead of silently rendered', () => {
  const invalid = structuredClone(data);
  invalid.runs[0].edges.push([data.personas[0].id, data.personas[0].id]);
  assert.throws(() => validateData(invalid), /Invalid edge/);
});
test('replay follows recorded additions/removals and rejects invented history', () => {
  const events = [{added: [['a', 'b']], removed: []}, {added: [['b', 'c']], removed: [['b', 'a']]}];
  assert.deepEqual(replayEdges(events, 2), [['b', 'c']]);
  assert.throws(() => replayEdges(null, 1), /recorded events/);
  assert.throws(() => replayEdges([{added: [], removed: [['a', 'b']]}], 1), /missing edge/);
});
