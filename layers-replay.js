// Replay helpers: pure functions shared by the page controller, the filter list and the tests.
// Replay is the recorded generation order of ONE saved run. Nothing here invents an order: a run whose record is
// a single response (Global) is reported as such and is never split into steps.
export const REPLAY_FINAL=Number.MAX_SAFE_INTEGER; // "final recorded state"; clamped to the real index once verified
const PERSON_LEVEL=Object.freeze(['local','sequential','iterative']);
const METHOD_NAMES={global:'Global',local:'Local',sequential:'Sequential',iterative:'Iterative'};
const PHASES={local:'Local choice','iterative-add':'Iterative add','iterative-drop':'Iterative drop',sequential:'Sequential choice',global:'Global response',iterative:'Iterative'};
const count=(n,word)=>`${n} ${word}${n===1?'':'s'}`;
const pair=edge=>[...edge].sort((a,b)=>Number(a)-Number(b)).join('–');

// Does this run carry more than one recorded event? Known exactly once its log is loaded; before that the method
// decides (every Global log is one response, every person-level log has one event per persona or more:
// asserted against all 890 logs in layers-replay.test.js).
export function hasStepRecord(run){
  if(!run)return false;
  if(Array.isArray(run.events)&&run.events.length)return run.events.length>1;
  return run.method!=='global';
}
// Where replay opens: at event 0 when there are steps to play, at the saved graph when the record is one response.
export const replayStart=run=>hasStepRecord(run)?0:REPLAY_FINAL;

// Which stacked run "Replay" opens by default: the active run when it has steps, otherwise the first stacked run
// that does. `switched` tells the caller to say so. A stack of single-response runs keeps the active run.
export function replayPick(runs,activeId){
  const active=runs.find(run=>run.run_id===activeId)||runs[0]||null;
  if(!active||hasStepRecord(active))return {run:active,switched:false,from:null};
  const other=runs.find(hasStepRecord);
  return other?{run:other,switched:true,from:active}:{run:active,switched:false,from:null};
}

// Person-level runs of the same condition and seed as a single-response run (same model, framing, language, seed,
// roster and prompt version), in method order. Only runs that exist as valid graphs are returned.
export function replayAlternatives(all,run){
  if(!run)return [];
  const keys=['study','roster_id','prompt_variant','model','culture','language','seed'];
  return PERSON_LEVEL.map(method=>all.find(other=>other.method===method&&keys.every(key=>other[key]===run[key]))).filter(Boolean);
}

// The persona with the most ties in a saved graph (lowest id on a tie); '' for an empty graph.
export function busiestPersona(edges){
  const degree=new Map();
  for(const edge of edges||[])for(const id of edge)degree.set(String(id),(degree.get(String(id))||0)+1);
  let best='',top=0;
  for(const [id,n] of degree)if(n>top||n===top&&Number(id)<Number(best)){best=id;top=n;}
  return best;
}

export const phaseLabel=method=>PHASES[method]||String(method||'Recorded update');

// What one recorded event did, as text for the card beside the stage. `frame` 0 is the state before any event.
// `ties` lists the changed ties exactly as recorded (order within an event was not recorded).
export function eventSummary(event,{frame=0,total=0,ties=0,limit=6}={}){
  if(!event)return {position:`Event 0 of ${total}`,actor:'Start',phase:'Before the first recorded response',added:0,removed:0,
    change:'No ties yet',list:[],more:0,sentence:`Start: no ties yet. ${count(total,'recorded event')} to replay.`};
  const added=event.added||[],removed=event.removed||[],single=event.persona==null;
  const actor=single?'Whole network':`Persona ${event.persona}`,phase=phaseLabel(event.method);
  const change=!added.length&&!removed.length?'No change to the network':`+${count(added.length,'tie')} · −${removed.length} removed`;
  const all=[...added.map(edge=>({kind:'add',text:pair(edge)})),...removed.map(edge=>({kind:'remove',text:pair(edge)}))];
  return {position:`Event ${frame} of ${total}`,actor,phase,added:added.length,removed:removed.length,change,list:all.slice(0,limit),more:Math.max(0,all.length-limit),
    sentence:`Event ${frame} of ${total}: ${actor}, ${phase}. ${added.length} added, ${removed.length} removed; ${count(ties,'tie')} now.`};
}

// One tick per recorded event for the scrubber, plus the recorded phases as contiguous segments.
// Consecutive iterative add/drop events form one "Iterative add / drop" segment.
export function eventTicks(events,persona=''){
  const person=persona?String(persona):'';
  const ticks=(events||[]).map((event,i)=>{
    const touches=!person||String(event.persona)===person||[...(event.added||[]),...(event.removed||[])].some(edge=>edge.map(String).includes(person));
    return {step:i+1,added:(event.added||[]).length,removed:(event.removed||[]).length,touches};
  });
  const group=method=>/^iterative/.test(method||'')?'Iterative add / drop':phaseLabel(method);
  const phases=[];
  (events||[]).forEach((event,i)=>{const label=group(event.method),last=phases.at(-1);if(last&&last.label===label)last.to=i+1;else phases.push({label,from:i+1,to:i+1});});
  return {ticks,phases,max:Math.max(1,...ticks.map(t=>Math.max(t.added,t.removed)))};
}
// SVG tick strip aligned with a range input of `total` steps (viewBox x = step). Bar height: ties changed in that
// event (square-root scale so single-tie events stay visible); teal up = added, orange down = removed.
export function tickStripSvg(events,{persona='',frame=0}={}){
  const {ticks,phases,max}=eventTicks(events,persona),total=ticks.length;
  if(total<2)return '';
  const h=value=>value?Math.max(2.5,8*Math.sqrt(value/max)):0;
  const bars=ticks.map(t=>{
    const cls=`${t.touches?'':' off'}${t.step<=frame?' done':''}`;
    return (t.added?`<line class="tick-add${cls}" x1="${t.step}" x2="${t.step}" y1="9" y2="${(9-h(t.added)).toFixed(1)}"/>`:'')+
      (t.removed?`<line class="tick-remove${cls}" x1="${t.step}" x2="${t.step}" y1="11" y2="${(11+h(t.removed)).toFixed(1)}"/>`:'')+
      (!t.added&&!t.removed?`<line class="tick-none${cls}" x1="${t.step}" x2="${t.step}" y1="9" y2="11"/>`:'');
  }).join('');
  const bounds=phases.slice(1).map(p=>`<line class="tick-phase" x1="${p.from-.5}" x2="${p.from-.5}" y1="0" y2="20"/>`).join('');
  return `<svg viewBox="0 0 ${total} 20" preserveAspectRatio="none" aria-hidden="true" focusable="false">${bounds}${bars}</svg>`;
}
// "50 Local choice events" / "50 Local choice, then 290 Iterative add / drop events".
export function phaseLine(events){
  const {phases}=eventTicks(events);
  if(!phases.length)return '';
  return phases.map(p=>`${p.to-p.from+1} ${p.label}`).join(', then ')+(phases.length===1&&phases[0].to===1?' event':' events');
}

// The plain statement for a run whose whole record is one response, and what to do instead.
export const SINGLE_RECORD_TEXT='Global generation returned the whole network in one response — there is no step-by-step record to replay.';
export function singleRecordState(all,run){
  const alternatives=replayAlternatives(all,run);
  return {text:SINGLE_RECORD_TEXT,alternatives:alternatives.map(other=>({run_id:other.run_id,method:other.method,label:`Replay the ${METHOD_NAMES[other.method]||other.method} run`})),
    hint:alternatives.length?'Person-level runs of the same model, setting and seed were recorded one response at a time:':'No person-level run of this condition and seed was saved as a valid graph.'};
}
