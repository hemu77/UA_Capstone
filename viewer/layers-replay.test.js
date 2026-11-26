// Unit tests for Replay (docs/qa/viewer-polish-2): what each method's record holds, where replay opens, that
// stepping follows the record exactly, and that a one-response record is stated instead of being split into steps.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {REPLAY_FINAL,hasStepRecord,replayStart,replayPick,replayAlternatives,busiestPersona,eventSummary,eventTicks,tickStripSvg,phaseLine,phaseLabel,singleRecordState,SINGLE_RECORD_TEXT} from './layers-replay.js';
import {replayFrames,edgeKey} from './graph-data.js';
import {loadPrimaryDataset} from './layers-data-all.js';
import {createStore} from './layers-app.js';
import {replayRun} from './layers-filters.js';

const fetchImpl=async url=>{const name=String(url).replace(/^\.\/data\//,'').split('?')[0];const bytes=readFileSync(new URL('./public/data/'+name,import.meta.url));
  return {ok:true,status:200,json:async()=>JSON.parse(bytes.toString('utf8')),text:async()=>bytes.toString('utf8'),arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};};
const data=await loadPrimaryDataset({fetchImpl});
const ids=data.personas.map(p=>p.id);
const eventsOf=run=>JSON.parse(readFileSync(new URL(`./public/data/primary-events/${run.run_id}.json`,import.meta.url)));
const pick=(method,model='gpt-4.1',seed=44000)=>data.runs.find(r=>r.method===method&&r.model===model&&r.culture==='us'&&r.language==='english'&&r.seed===seed);

test('event counts per method: Global is one response, person-level methods are one event per persona or more',()=>{
  const counts={global:[],local:[],sequential:[],iterative:[]};
  for(const run of data.runs){
    const events=eventsOf(run);counts[run.method].push(events.length);
    // The method alone predicts whether there are steps, before and after the log is loaded.
    assert.equal(hasStepRecord(run),events.length>1,run.run_id);
    assert.equal(hasStepRecord({...run,events}),events.length>1,run.run_id);
    if(run.method==='global'){assert.equal(events[0].persona,null);assert.equal(events[0].added.length,run.edges.length);assert.equal(events[0].removed.length,0);}
    else{
      assert.ok(events.every(event=>ids.includes(event.persona)),'every person-level event names its acting persona');
      // The first 50 events are one per persona, each persona exactly once.
      assert.deepEqual(events.slice(0,50).map(e=>Number(e.persona)).sort((a,b)=>a-b),ids.map(Number).sort((a,b)=>a-b),run.run_id);
    }
    if(run.method==='iterative')assert.ok(events.slice(50).every(e=>/^iterative-(add|drop)$/.test(e.method))&&events.slice(0,50).every(e=>e.method==='local'),run.run_id);
  }
  assert.deepEqual(Object.fromEntries(Object.entries(counts).map(([k,v])=>[k,v.length])),{global:219,local:224,sequential:224,iterative:223});
  assert.equal(Object.values(counts).flat().length,890);
  assert.ok(counts.global.every(n=>n===1),'every Global log is a single event');
  assert.ok(counts.local.every(n=>n===50)&&counts.sequential.every(n=>n===50),'Local and Sequential: exactly one event per persona');
  assert.ok(counts.iterative.every(n=>n>50&&n<=350),'Iterative: 50 first choices, then recorded add / drop events');
});

test('replay opens at event 0 when there are steps, and on the saved graph when the record is one response',()=>{
  for(const method of ['local','sequential','iterative']){const run=pick(method);assert.equal(replayStart(run),0);assert.equal(replayStart({...run,events:eventsOf(run)}),0);}
  assert.equal(replayStart(pick('global')),REPLAY_FINAL);
  assert.equal(replayStart(undefined),REPLAY_FINAL);
  // The filter list and the 1-8 keys go through replayRun: one update, cursor at the start (or the saved graph).
  const stack=['global','local','sequential','iterative'].map(m=>pick(m).run_id),app=createStore(),seen=[];
  app.init(data,{layers:stack,active:stack[0],view:'stack',frame:REPLAY_FINAL});
  app.subscribe((state,changed)=>seen.push([...changed].sort()));
  replayRun(app,stack[1]);assert.deepEqual(seen,[['active','frame','view']]);assert.equal(app.getState().frame,0);
  replayRun(app,stack[3]);assert.equal(app.getState().frame,0);assert.equal(app.getState().active,stack[3]);
  app.update({frame:17});replayRun(app,stack[2]);assert.equal(app.getState().frame,0,'another run never inherits the previous cursor');
  replayRun(app,stack[0]);assert.equal(app.getState().frame,REPLAY_FINAL);
});

test('stepping changes the tie count exactly as recorded, for every method and model',()=>{
  let checked=0;
  for(const model of data.models)for(const method of data.methods){
    const run=pick(method,model)||data.runs.find(r=>r.model===model&&r.method===method),events=eventsOf(run),frames=replayFrames({...run,events},ids);
    assert.equal(frames.length,events.length+1);assert.equal(frames[0].edges.length,0,'event 0 has no ties');assert.equal(frames[0].event,null);
    let present=new Set();
    events.forEach((event,i)=>{
      for(const edge of event.removed){assert.ok(present.has(edgeKey(edge)),'a removed tie was present');present.delete(edgeKey(edge));}
      for(const edge of event.added){assert.ok(!present.has(edgeKey(edge)),'an added tie was absent');present.add(edgeKey(edge));}
      assert.equal(frames[i+1].edges.length,frames[i].edges.length+event.added.length-event.removed.length,`${run.run_id} step ${i+1}`);
      assert.deepEqual(new Set(frames[i+1].edges.map(edgeKey)),present);
      assert.equal(frames[i+1].event,event,'frame k is the state after recorded event k');
    });
    assert.equal(frames.at(-1).edges.length,run.edges.length);checked++;
  }
  assert.equal(checked,16);
});

test('a mixed stack replays a run with a multi-event record by default, and says which',()=>{
  const stack=['global','local','sequential','iterative'].map(m=>pick(m));
  let choice=replayPick(stack,stack[0].run_id);
  assert.equal(choice.run,stack[1]);assert.equal(choice.switched,true);assert.equal(choice.from,stack[0]);
  choice=replayPick(stack,stack[3].run_id);assert.equal(choice.run,stack[3]);assert.equal(choice.switched,false,'an active run with steps is kept');
  const globals=data.runs.filter(r=>r.method==='global').slice(0,3);
  choice=replayPick(globals,globals[1].run_id);assert.equal(choice.run,globals[1]);assert.equal(choice.switched,false,'nothing to prefer in an all-Global stack');
  assert.equal(replayPick([],'x').run,null);
  assert.equal(replayPick([stack[0],stack[2]],'absent').run,stack[2],'an unknown active id falls back to the stack');
});

test('single-event runs get the explanatory state and real alternatives, never invented steps',()=>{
  const run=pick('global'),state=singleRecordState(data.runs,run);
  assert.equal(state.text,'Global generation returned the whole network in one response — there is no step-by-step record to replay.');
  assert.equal(state.text,SINGLE_RECORD_TEXT);
  assert.deepEqual(state.alternatives.map(a=>a.method),['local','sequential','iterative']);
  assert.deepEqual(state.alternatives.map(a=>a.label),['Replay the Local run','Replay the Sequential run','Replay the Iterative run']);
  for(const other of replayAlternatives(data.runs,run)){
    for(const key of ['model','culture','language','seed','roster_id'])assert.equal(other[key],run[key]);
    assert.ok(hasStepRecord(other));assert.ok(data.runs.includes(other),'alternatives are saved primary runs');
  }
  // Every Global run: alternatives are only valid saved graphs of the same condition and seed (failed attempts are absent).
  for(const g of data.runs.filter(r=>r.method==='global')){
    const list=replayAlternatives(data.runs,g);assert.ok(list.length<=3);
    assert.ok(list.every(o=>o.model===g.model&&o.culture===g.culture&&o.language===g.language&&o.seed===g.seed&&o.method!=='global'));
  }
  assert.match(singleRecordState([run],run).hint,/No person-level run/);assert.deepEqual(singleRecordState([run],run).alternatives,[]);
  // One event stays one event: no tick strip, one tick, no phases beyond the response itself.
  const events=eventsOf(run);
  assert.equal(tickStripSvg(events),'');assert.equal(eventTicks(events).ticks.length,1);assert.equal(phaseLine(events),'1 Global response event');
  assert.equal(replayFrames({...run,events},ids).length,2,'the record offers the start and the whole network, nothing between');
  assert.equal(busiestPersona([['1','2'],['2','3'],['4','3']]),'2');assert.equal(busiestPersona([]),'');
});

test('event card text and scrubber ticks describe the recorded event only',()=>{
  const run=pick('iterative'),events=eventsOf(run),frames=replayFrames({...run,events},ids);
  const start=eventSummary(null,{frame:0,total:events.length});
  assert.equal(start.position,`Event 0 of ${events.length}`);assert.equal(start.actor,'Start');assert.equal(start.added+start.removed,0);
  events.forEach((event,i)=>{
    const info=eventSummary(event,{frame:i+1,total:events.length,ties:frames[i+1].edges.length,limit:4});
    assert.equal(info.position,`Event ${i+1} of ${events.length}`);assert.equal(info.actor,`Persona ${event.persona}`);assert.equal(info.phase,phaseLabel(event.method));
    assert.equal(info.added,event.added.length);assert.equal(info.removed,event.removed.length);
    assert.equal(info.list.length+info.more,event.added.length+event.removed.length,'every changed tie is listed or counted');
    assert.ok(info.sentence.includes(`${frames[i+1].edges.length} tie`));
  });
  const {ticks,phases}=eventTicks(events);
  assert.equal(ticks.length,events.length);assert.deepEqual(ticks.map(t=>t.step),events.map((e,i)=>i+1));
  assert.deepEqual(ticks.map(t=>[t.added,t.removed]),events.map(e=>[e.added.length,e.removed.length]));
  assert.deepEqual(phases,[{label:'Local choice',from:1,to:50},{label:'Iterative add / drop',from:51,to:events.length}]);
  assert.equal(phaseLine(events),`50 Local choice, then ${events.length-50} Iterative add / drop events`);
  const svg=tickStripSvg(events);
  assert.match(svg,new RegExp(`viewBox="0 0 ${events.length} 20"`));
  assert.equal((svg.match(/class="tick-add/g)||[]).length,events.filter(e=>e.added.length).length);
  assert.equal((svg.match(/class="tick-remove/g)||[]).length,events.filter(e=>e.removed.length).length);
  // A selected persona dims the events that do not involve it; nothing is dropped.
  const person=events[0].persona,dimmed=eventTicks(events,person).ticks;
  assert.equal(dimmed.length,events.length);
  assert.deepEqual(dimmed.filter(t=>t.touches).map(t=>t.step),events.flatMap((e,i)=>e.persona===person||[...e.added,...e.removed].some(edge=>edge.includes(person))?[i+1]:[]));
  assert.equal(phaseLine(eventsOf(pick('sequential'))),'50 Sequential choice events');
});

test('the page carries the whole transport: start, back, play/pause, next, end, speed, scrubber with ticks',()=>{
  const html=readFileSync(new URL('./layers.html',import.meta.url),'utf8');
  for(const id of ['rewind','previous','play','next','show-final','speed','timeline','timeline-ticks','step-label','playback-scope','replay-single','event-delta'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1,id);
  const order=['rewind','previous','play','next','show-final'].map(id=>html.indexOf(`id="${id}"`));
  assert.deepEqual([...order].sort((a,b)=>a-b),order,'transport buttons read start, back, play, next, end');
  for(const [id,key] of [['rewind','Home'],['previous','ArrowLeft'],['play','Space'],['next','ArrowRight'],['show-final','End']])assert.match(html,new RegExp(`id="${id}"[^>]*aria-keyshortcuts="${key}"`));
  assert.ok((html.match(/<option value="\d+"/g)||[]).length>=4,'four playback speeds');
});
