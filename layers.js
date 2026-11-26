// Layers page controller. WORKSTREAM 2 owns this file (docs/qa/viewer-3d-2026-11-25/SPEC.md C.4).
// The shared store (layers-app.js) is the single source of truth for view, layers, active run, replay frame,
// persona, colour, lens, groups, filters and focus. Every control here writes the store through commit();
// one subscriber derives everything on screen from the store, whoever changed it. Only presentation-local
// controls (tie opacity, plane spacing, tie filter, label density, "only this person's ties", persona journey,
// playback timer) live outside the store.
import * as THREE from 'three';
import {comparisonLayout, layerComparison, neighbors, replayFrames, personaSteps, personaView, personaTag, personaComparison, togglePersona, tapTracker, matchedRuns, evidenceDataset, conditionCoverage, MAX_LAYERS, edgeKey} from './graph-data.js?v=d068babb8c4c';
import {topology as topologyMetrics} from './research-plots.js?v=d068babb8c4c';
import {runLabel, runParts, layerLabel, homophilyMatrix, frequencyMatrix, modelName, methodName, settingName, plural} from './formation-charts.js?v=d068babb8c4c';
import {configureNavigation, panCamera} from './navigation.js?v=d068babb8c4c';
import {decodeVerifiedEvents} from './verified-events.js?v=d068babb8c4c';
import {createStore, EMPTY_FILTERS, LAYER_COLORS, MAX_STACK_LAYERS, VIEWS} from './layers-app.js?v=d068babb8c4c';
import {loadPrimaryDataset, loadMetrics, settingKey} from './layers-data-all.js?v=d068babb8c4c';
import {mountFilters, applyFilters, decodeState, slotMask} from './layers-filters.js?v=d068babb8c4c';
import {mountInsight, nodeEncoding, edgeEncoding, consensus as consensusOf, comparisonDesign, comparisonReading, overlapLine, metricChartHtml, tieTableHtml, ledgerRows, trajectoryChartHtml} from './layers-insight.js?v=d068babb8c4c';
import {mountShell} from './layers-shell.js?v=d068babb8c4c';
import {createStage3D, STAGE_COLORS, boxCorners} from './layers-3d-scene.js?v=d068babb8c4c';
import {createAtlas} from './layers-3d-atlas.js?v=d068babb8c4c';
import {createStack, spreadLayout} from './layers-3d-stack.js?v=d068babb8c4c';
import {createLabels} from './layers-3d-labels.js?v=d068babb8c4c';
import {REPLAY_FINAL,hasStepRecord,replayStart,replayPick,busiestPersona,eventSummary,tickStripSvg,phaseLine,singleRecordState} from './layers-replay.js?v=d068babb8c4c';
export const app=createStore();
// Diagnostics for checks: the drawable objects currently submitted to the GPU (name, type, instance/vertex count).
export function drawables(){
  const out=[];if(!stage)return out;
  stage.scene.traverseVisible(o=>{if(o.isMesh||o.isPoints||o.isLine||o.isLineSegments)out.push({name:o.parent?.name||'',type:o.type,count:o.isInstancedMesh?o.count:o.geometry?.instanceCount??o.geometry?.drawRange?.count});});
  return out;
}

const $=id=>document.getElementById(id);
// The address bar belongs to layers-filters.js (short grammar, SPEC.md B.4 / C.5); this file only reads links.
const STACK_LIMIT=Math.min(MAX_STACK_LAYERS,MAX_LAYERS);
const FINAL=REPLAY_FINAL; // "final recorded state"; clamped to the real index once the history is verified
const SOURCE='layers.js';
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const url=new URL(location.href);
const dataset='primary';
const label=runLabel;
const fmt=v=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(3):'NA';
const option=(select,value,text)=>select?.add(new Option(text,value));
const S=()=>app.getState();
const commit=patch=>app.update(patch,SOURCE);

let data=null,runById=new Map();
let stage=null,atlas=null,stack=null,labels=null,raycaster=null;
let runs=[],stackLayout=null,stackLayoutKey='',trace=null;
let journey=false,playing=false,timer=null,removed=[],startingRuns=[],animateNext=false,keepUndo=false;
let consensusLive=false;
let ticksKey='',singleKey='',replayNotice='',replayNoticeFor='';
let ledgerKey='',aggregateKey='',tieShowAll=false,questionKey='',lastView=null,lastPlaneKey='',entering=null,loadNotice='';
const view3d={lift:false,ribbons:false}; // ribbons are opt-in: with many shared ties they cover the planes
const traceCache=new Map(),historyLoads=new Map();

// Saved measurements beyond density and clustering (and the saved homophily scores) arrive with metrics.json.
// Anything that shows them asks here first: false means "not loaded yet" and a repaint follows when they are.
let metricsState='idle';
const BASIC_METRICS=new Set(['density','avg_clustering_coef']);
function metricsReady(){
  if(metricsState==='ready')return true;
  if(metricsState==='idle'){
    metricsState='loading';const owner=data;
    loadMetrics(data).then(()=>{metricsState='ready';if(data===owner)paintSelection();}).catch(error=>{metricsState='failed';announce(`Saved measurements could not be loaded: ${error.message}`);if(data===owner)paintSelection();});
  }
  return false;
}
const metricsNote=()=>`<p class="hint">${metricsState==='failed'?'Saved measurements could not be loaded. Density and mean clustering remain available; reload to retry.':'Loading saved measurements…'}</p>`;
function announce(message){const status=$('action-status');if(status)status.textContent=message;}
function activeRun(){return runById.get(S().active)&&runs.find(run=>run.run_id===S().active)||runs[0];}
function stop(){clearTimeout(timer);timer=null;playing=false;}
const isReplay=()=>S().view==='replay';
const selectedPersona=()=>S().persona||'';
const incidentOnly=()=>!!$('incident')?.checked;

// ---------------------------------------------------------------------------------------------
// Validation of a requested stack. Returns '' when acceptable, otherwise a message; never truncates.
function stackProblem(list) {
  if(!list.length)return 'Choose at least one saved run.';
  if(list.some(run=>!run))return 'The selection contains a run absent from this dataset.';
  if(list.length>STACK_LIMIT)return `A stack holds at most ${STACK_LIMIT} runs (one condition's ${STACK_LIMIT} seeds). Remove a layer first.`;
  try{layerComparison(list);}catch(error){return error.message;}
  return '';
}
function setStack(ids,patch={},message='') {
  const list=ids.map(id=>runById.get(id)),problem=stackProblem(list);
  if(problem){announce(problem);return false;}
  const active=list.some(run=>run.run_id===(patch.active??S().active))?(patch.active??S().active):list[0].run_id;
  commit({...patch,layers:ids,active});
  if(message)announce(message);
  return true;
}

// ---------------------------------------------------------------------------------------------
// Recorded replay histories: fetched lazily, checksum-verified, cached (unchanged semantics).
async function verifiedHistory(run) {
  if(traceCache.has(run.run_id)){
    const frames=traceCache.get(run.run_id);
    if(!frames)throw Error('Recorded history failed verification; reload to retry');
    return frames;
  }
  if(historyLoads.has(run.run_id))return historyLoads.get(run.run_id);
  const id=run.run_id,owner=data;
  const task=(async()=>{
    let events=run.events;
    if(!events?.length){
      if(![`./data/primary-events/${id}.json`,`./data/primary-events/${id}.json.gz`].includes(run.events_url)||!/^[a-zA-Z0-9_.-]+$/.test(id))throw Error('Invalid event path');
      const response=await fetch(run.events_url);
      if(!response.ok)throw Error('Recorded replay could not be loaded');
      events=await decodeVerifiedEvents(await response.arrayBuffer(),run.events_sha256);
    }
    const frames=replayFrames({...run,events},owner.personas.map(p=>p.id));
    if(!frames)throw Error('Recorded history is empty');
    // Retain verified raw events for offline revisits; bound expanded replay prefixes.
    run.events=events;traceCache.set(id,frames);
    if(traceCache.size>12){const oldest=traceCache.keys().next().value;traceCache.delete(oldest);historyLoads.delete(oldest);}
    return frames;
  })();
  historyLoads.set(id,task);
  try{return await task;}catch(error){historyLoads.delete(id);traceCache.set(id,null);throw error;}
}
function clampFrame(){if(trace&&S().frame>trace.length-1)queueMicrotask(()=>{if(trace&&S().frame>trace.length-1)commit({frame:trace.length-1});});}
function loadTrace() {
  const run=activeRun();if(!run){trace=null;return;}
  const id=run.run_id;
  if(run.events_url&&!traceCache.has(id)){
    trace=null;const owner=data;
    verifiedHistory(run).then(frames=>{
      if(data===owner&&activeRun()?.run_id===id){trace=frames;if(S().frame>frames.length-1)commit({frame:frames.length-1});else render(new Set(['trace']));}
    }).catch(error=>{
      if(data===owner&&activeRun()?.run_id===id){announce(error.message+' Final graph is still available. Reload to retry.');render(new Set(['trace']));}
    });
    return;
  }
  if(!traceCache.has(id))traceCache.set(id,run.replay_status==='actual_recorded_graph_changes'?replayFrames(run,data.personas.map(p=>p.id)):null);
  trace=traceCache.get(id);clampFrame();
}
// A one-response record has nothing between "nothing" and "everything": its replay position is always the saved graph.
const currentFrame=()=>!trace?0:hasStepRecord(activeRun())?Math.min(S().frame,trace.length-1):trace.length-1;

// ---------------------------------------------------------------------------------------------
// Actions. Each one writes the store; render() reacts.
function selectPersona(id,toggle=true){
  stop();journey=false;
  const next=toggle?togglePersona(selectedPersona(),id):id;
  if($('incident'))$('incident').checked=!!next;
  if(!commit({persona:next}).size)render(new Set(['local']));
}
// Replay always opens at the start of the record (event 0); a single-response run opens on its saved graph.
function activate(run,notice=''){stop();journey=false;replayNotice=notice;replayNoticeFor=run.run_id;commit({active:run.run_id,view:'replay',frame:replayStart(run)});}
// "Replay" without naming a run: the active run when it has a step-by-step record, otherwise the first stacked run
// that has one (and the page says which run that is).
function openReplay(){
  stop();journey=false;if($('incident'))$('incident').checked=false;
  const pick=replayPick(runs,S().active),run=pick.run,index=runs.indexOf(run)+1;
  replayNoticeFor=run.run_id;replayNotice=pick.switched?`Replaying run ${index} (${methodName(run.method)}): run ${runs.indexOf(pick.from)+1} (${methodName(pick.from.method)}) was returned in one response and has no steps.`:'';
  if(!commit({view:'replay',active:run.run_id,frame:replayStart(run),persona:''}).size)render(new Set(['local']));
  announce(`Replaying run ${index} of ${runs.length}: ${label(run)}.`);
  stage?.pose('orbit');
}
function seek(step,preserveScope=false){
  stop();replayNotice='';if(!preserveScope){journey=false;if($('incident'))$('incident').checked=false;}
  if(!commit({frame:Math.max(0,Math.min(trace?.length-1||0,step))}).size)render(new Set(['local']));
}
function steps(){return journey?personaSteps(activeRun().events,selectedPersona()):Array.from({length:trace?.length-1||0},(_,i)=>i+1);}
function advance(){
  const frame=currentFrame(),next=steps().find(step=>step>frame);
  if(next===undefined){stop();render(new Set(['local']));return;}
  animateNext=true;replayNotice='';commit({frame:next});
  timer=setTimeout(advance,Number($('speed')?.value||800));
}
function play(){
  if(playing){stop();render(new Set(['local']));return;}
  if(!isReplay())openReplay();
  if(!trace||!hasStepRecord(activeRun()))return;
  stop();replayNotice='';
  const patch={view:'replay'};
  commit(patch);
  if(!steps().some(step=>step>currentFrame()))commit({frame:0});
  playing=true;render(new Set(['local']));timer=setTimeout(advance,Number($('speed')?.value||800));
}
function removeLayer(index){
  const run=runs[index];if(!run||runs.length<2)return;
  removed.push({run,index});keepUndo=true;
  // Move focus to Undo before the list re-renders, so focus never falls back to the page when the button goes.
  const undo=$('undo');if(undo){undo.disabled=false;undo.focus({preventScroll:true});}
  setStack(runs.filter((_,i)=>i!==index).map(r=>r.run_id),{},`Removed ${modelName(run.model)} / ${methodName(run.method)}. Undo remove restores it even if filters change.`);
  keepUndo=false;
}
// Condition = one tower = one coverage cell: filter to it and stack its valid seeds (at most 8).
function openCondition(model,method,culture,language,{slot=null,fly=false}={}) {
  const ids=data.runs.filter(run=>run.model===model&&run.method===method&&run.culture===culture&&run.language===language).sort((a,b)=>a.seed-b.seed).map(run=>run.run_id);
  if(!ids.length){announce('No valid graph exists for this condition; its attempts failed.');return;}
  const failed=(data.failures||[]).filter(run=>run.model===model&&run.method===method&&run.culture===culture&&run.language===language).length;
  const active=slot!==null&&data.slots[slot]?.status!=='failed'?data.slots[slot].run_id:ids[0];
  const filters={...EMPTY_FILTERS,model:[model],method:[method],setting:[settingKey(culture,language)]};
  const apply=()=>{
    entering=fly?'tower':null;
    const message=`Stacked ${plural(ids.length,'valid seed')} of ${modelName(model)} · ${methodName(method)} · ${settingName(culture,language)}${failed?`; ${plural(failed,'failed attempt')} not shown`:''}.`;
    if(setStack(ids,{filters,active,view:'stack',frame:FINAL},message)&&$('match-note'))$('match-note').textContent='Inspecting repetitions of one condition, not a cross-model or cross-country contrast.';
  };
  if(fly&&stage&&atlas?.visible){const tower=atlas.towerOf(slot??slotOfCondition(model,method,culture,language));stage.flyTo(atlas.towerFraming(tower.index),{duration:420}).then(apply);}
  else apply();
}
function slotOfCondition(model,method,culture,language){const d=data.design;return ((d.models.indexOf(model)*d.methods.length+d.methods.indexOf(method))*d.settings.length+d.settings.findIndex(([c,l])=>c===culture&&l===language))*d.seeds.length;}

// Filters saved in a legacy ?view=<json> link (single values per field) as store filters.
function legacyFilters(saved={}){
  const pick=key=>typeof saved[key]==='string'&&saved[key]?saved[key]:'';
  const culture=pick('culture'),language=pick('language');
  const settings=data.design.settings.filter(([c,l])=>(!culture||c===culture)&&(!language||l===language)).map(([c,l])=>settingKey(c,l));
  return {...EMPTY_FILTERS,model:pick('model')&&data.design.models.includes(pick('model'))?[pick('model')]:[],method:pick('method')&&data.design.methods.includes(pick('method'))?[pick('method')]:[],
    setting:culture||language?settings:[],seed:pick('seed')&&data.design.seeds.includes(Number(pick('seed')))?[Number(pick('seed'))]:[]};
}

// ---------------------------------------------------------------------------------------------
// Colour encoding comes from layers-insight.js (node colours, group isolation, same/mixed tie lens); recomputed
// from the store on every relevant change and applied in place (instance and line colours only).
function encodings(){
  const s=S(),node=nodeEncoding(data,s),edge=edgeEncoding(data,s),lens=s.lens!=='none';
  return {node:{colorOf:(id,l)=>node.colorOf(id,l),dimmed:id=>!!node.dimmed?.(id)},
    edge:lens||s.groups.length?{classOf:e=>lens?edge.classOf(e):null,colorOf:e=>edge.colorOf?.(e)||null,dimmed:e=>!!edge.dimmed?.(e)}:null};
}

// ---------------------------------------------------------------------------------------------
// 3D stage.
function setupStage(){
  labels=createLabels($('plane-labels'),{onRemeasure:()=>stage?.requestRender()});
  try{stage=createStage3D($('viewport'),{labels,reducedMotion:reducedMotion.matches});}
  catch{stage=null;labels.dispose();labels=null;document.body.dataset.renderer='unavailable';if($('status'))$('status').textContent='WebGL unavailable. Measurements and condition plots remain usable.';return;}
  reducedMotion.addEventListener?.('change',event=>{stage.setReducedMotion(event.matches);paintTools();});
  const canvas=stage.canvas;
  canvas.tabIndex=0;canvas.dataset.slots=String(data.counts?.attempts??data.slots.length);
  canvas.setAttribute('aria-label','Network camera. Drag to rotate, Shift-drag to pan, scroll or pinch to zoom toward pointer. Arrow keys pan; Shift-arrow rotates.');
  stage.controls.listenToKeyEvents(canvas);
  stack=createStack(stage,data,{colors:{layers:LAYER_COLORS,same:STAGE_COLORS.accent}});
  // The Atlas framing depends on the canvas shape (phone portrait vs desktop); refresh it before the stage re-poses.
  stage.onResize(()=>{if(atlas&&S().view==='atlas'){environment('atlas');labels.set(atlasLabels());}else if(stack&&lastView)environment(S().view);});
  raycaster=new THREE.Raycaster();
  // Keep the label layer aligned with the canvas even if the shell positions the two boxes differently.
  // Chrome that the shell draws over the canvas (heading and tools at the top, key and legend at the bottom) is
  // measured here, on resize only, so the camera frames content in the area people can actually see.
  const align=()=>{
    const view=$('viewport').getBoundingClientRect(),host=$('plane-labels').getBoundingClientRect();
    stage.setOffset(view.left-host.left,view.top-host.top);
    let top=0,bottom=0;
    // The 3D key and encoding toggles (#stage-overlay) sit above the camera bar and count as bottom chrome too.
    for(const element of document.querySelectorAll('.stage-heading,.stage-tools,.stage-key,.stage-footer,#stage-overlay .g3-hud')){
      const r=element.getBoundingClientRect();if(!r.height||r.bottom<view.top||r.top>view.bottom)continue;
      if(r.top-view.top<view.height*.35&&r.bottom-view.top<view.height*.5)top=Math.max(top,r.bottom-view.top);
      else if(view.bottom-r.bottom<view.height*.35)bottom=Math.max(bottom,view.bottom-r.top);
    }
    // The heading is quiet text and may overlap the far edge of the scene a little; the bottom band (legend, key,
    // toggles, camera bar) never covers it.
    stage.setSafeArea({top:top*.85,bottom});
  };
  const observer=new ResizeObserver(align);observer.observe($('viewport'));observer.observe($('plane-labels'));
  stage.observeChrome=element=>observer.observe(element);
  for(const element of document.querySelectorAll('.stage-heading,.stage-tools,.stage-key,.stage-footer'))observer.observe(element);
  const interaction=$('graph-interaction'),dragMode=$('drag-mode');
  const configureInteraction=()=>{const enabled=interaction?interaction.checked:true;configureNavigation(stage.controls,enabled,dragMode?.value||'orbit');stage.controls.enableDamping=!stage.reducedMotion;canvas.style.setProperty('touch-action',enabled?'none':'pan-y','important');};
  if(interaction)interaction.onchange=configureInteraction;if(dragMode)dragMode.onchange=configureInteraction;configureInteraction();
  const tap=tapTracker(),pointer=new THREE.Vector2();
  const rayAt=event=>{const rect=canvas.getBoundingClientRect();pointer.set(2*(event.clientX-rect.left)/rect.width-1,1-2*(event.clientY-rect.top)/rect.height);raycaster.setFromCamera(pointer,stage.camera);return raycaster.ray;};
  canvas.addEventListener('pointerdown',event=>{canvas.focus({preventScroll:true});if(stage.controls.autoRotate){stage.setAutoRotate(false);paintTools();}tap.down(event);},true);
  canvas.addEventListener('wheel',event=>{
    if(!stage.controls.enabled||!event.shiftKey)return;
    event.preventDefault();event.stopImmediatePropagation();tap.cancel();
    const unit=event.deltaMode===1?16:event.deltaMode===2?canvas.clientHeight:1;
    panCamera(stage.camera,stage.controls,event.deltaX*unit,event.deltaY*unit,stage.size.height);stage.markUserMoved();stage.requestRender();
  },{capture:true,passive:false});
  canvas.addEventListener('wheel',()=>tap.cancel(),{passive:true});
  canvas.addEventListener('pointercancel',()=>tap.cancel());
  canvas.addEventListener('pointermove',event=>{
    tap.move(event);
    if(event.buttons){hideTooltip();return;}
    hover(rayAt(event));
  });
  canvas.addEventListener('pointerleave',()=>{hideTooltip();if(S().view==='atlas'&&S().focusSlot!==null)commit({focusSlot:null});});
  canvas.addEventListener('pointerup',event=>{
    if(!tap.up(event))return;
    const ray=rayAt(event);
    if(S().view==='atlas'){
      const hit=atlas?.pick(ray);
      if(!hit)return;
      const tower=atlas.layout.towers[hit.tower];
      openCondition(tower.model,tower.method,tower.setting[0],tower.setting[1],{slot:hit.slot,fly:true});
      return;
    }
    const hit=stack.pick(ray);
    selectPersona(hit?.person||'',!!hit?.person);
  });
  stage.addHook(()=>{const s=stage.stats();canvas.dataset.geometries=String(s.geometries);});
}
function hideTooltip(){const tip=$('node-tooltip');if(tip)tip.hidden=true;stage&&(stage.canvas.style.cursor='');}
function hover(ray){
  const tip=$('node-tooltip');if(!tip)return;
  if(S().view==='atlas'&&atlas){
    const hit=atlas.pick(ray);
    if(S().focusSlot!==(hit?.slot??null))commit({focusSlot:hit?.slot??null});
    tip.hidden=!hit;stage.canvas.style.cursor=hit?'pointer':'';
    if(hit)tip.replaceChildren(towerCard(hit));
    return;
  }
  const hit=stack?.pick(ray),person=hit?.person&&data.personas.find(p=>p.id===hit.person);
  tip.hidden=!person;stage.canvas.style.cursor=person?'pointer':'';
  if(person){const run=runById.get(hit.run_id);tip.replaceChildren(personaCard(person,`${modelName(run.model)} / ${methodName(run.method)} · seed ${run.seed}`));}
}
function towerCard(hit){
  const tower=atlas.layout.towers[hit.tower],slots=tower.slots.map(slot=>data.slots[slot]);
  const valid=slots.filter(s=>s.status!=='failed'),empty=slots.filter(s=>s.status==='empty').length,failed=slots.length-valid.length;
  const card=document.createElement('div'),title=document.createElement('strong'),counts=document.createElement('span'),slot=document.createElement('span'),hint=document.createElement('span');
  card.className='persona-card g3-card';
  title.textContent=`${modelName(tower.model)} · ${methodName(tower.method)} · ${settingName(...tower.setting)}`;
  counts.textContent=`${valid.length} of ${slots.length} valid${empty?` · ${plural(empty,'empty graph')}`:''}${failed?` · ${plural(failed,'failed attempt')}`:''}`;
  const target=data.slots[hit.slot],seed=data.design.seeds[hit.slot%data.design.seeds.length],run=runById.get(target.run_id);
  slot.textContent=target.status==='failed'?`Seed ${seed}: failed attempt${target.failure?` (${String(target.failure).replace(/_/g,' ')})`:''}, not a graph`:`Seed ${seed}: ${plural(run?.edges.length??0,'tie')}${target.status==='empty'?' (valid, empty)':''}`;
  hint.className='g3-card-hint';hint.textContent=valid.length?`Click to stack ${valid.length===1?'its valid seed':`its ${valid.length} valid seeds`}`:'No valid graph to stack';
  for(const part of [counts,slot,hint])part.classList.add('persona-context');
  card.append(title,counts,slot,hint);return card;
}
function ensureAtlas(){
  if(atlas||!stage)return atlas;
  const started=performance.now();
  let layout=data.layout;
  if(!(layout?.xy?.length>=data.personas.length*2)&&!(layout instanceof Map)){
    // The compact loader carries a precomputed layout; this fallback derives the same kind of layout here.
    const counts=new Map();for(const run of data.runs)for(const edge of run.edges){const key=edgeKey(edge);counts.set(key,(counts.get(key)||0)+1);}
    const max=Math.max(1,...counts.values()),weights=new Map([...counts].map(([key,n])=>[key,n/max]));
    layout=comparisonLayout(data.personas,[...counts.keys()].map(key=>JSON.parse(key)),[],{weights,planar:true});
  }
  atlas=createAtlas(stage,data,{layout,colors:{valid:STAGE_COLORS.accent,empty:STAGE_COLORS.empty,failed:STAGE_COLORS.failed,same:STAGE_COLORS.accent}});
  atlas.setVisible(false);
  performance.measure?.('layers-3d:atlas-build',{start:started,end:performance.now()});
  return atlas;
}
// Slot mask for the active filters (layers-filters.js): 1 matching graph, 2 failed attempt in scope, 0 outside.
const filterMask=filters=>slotMask(data,filters);
function environment(view){
  if(!stage)return;
  if(view==='atlas'){
    // Room on the left for the setting labels (full names on wide canvases, codes on narrow ones) and a little on the
    // right for the seed axis. The camera looks down at about 57 degrees (62 on phones): steep enough that every row
    // clears the row in front of it, shallow enough that tower height (seed order) still reads.
    // Label room is reserved in screen pixels (a view offset), so the camera still looks straight down the middle.
    const b=atlas.layout.bounds,narrow=stage.size.width<700;
    const padPx=narrow?[54,8]:[Math.min(190,stage.size.width*.17),96];
    const direction=narrow?[0,1.9,1]:[0,1.55,1],front=narrow?14:21; // model and method labels in front of the first row
    const center=[0,b.height*.3,front/2],points=boxCorners([-b.width/2,0,-b.depth/2],[b.width/2,b.height,b.depth/2+front],center);
    stage.setBounds({center,radius:b.width*.5,points,margin:1.02,fov:22,padPx,fogScale:.36,poses:{orbit:direction,front:[0,.4,1],top:[0,1,.02]}});
    stage.setEnvironment({groundY:0,groundRadius:Math.max(b.width,b.depth)*.85,cell:5.3,center:[0,0]});return;
  }
  // Stack and Replay fit the exact projected box of the planes (perspective included). The camera looks down more
  // steeply when there are few planes (graphs read best from above) and flatter for tall stacks (planes separate).
  // Plane labels get a reserved column on the left (a view offset, so perspective stays symmetric).
  const b=stack.bounds(),plane=isReplay()?150:110,narrow=stage.size.width<700,L=Math.max(1,stack.planes);
  const elevation=isReplay()?.78:L<=2?.8:L<=4?.6:.38;
  const direction=isReplay()?[.22,elevation,1]:[.42,elevation,.9],padPx=isReplay()?[0,0]:narrow?[100,4]:[Math.min(150,stage.size.width*.13),16];
  stage.setBounds({center:[0,(b.box.min[1]+b.box.max[1])/2,0],radius:b.radius,points:b.points,margin:1.02,fov:34,padPx,fogScale:L>4?.5:.42,poses:{orbit:direction,front:[.02,.32,1],top:[0,1,.02]}});
  stage.setEnvironment({groundY:b.floor,groundRadius:plane*2.7,cell:plane/5.5,center:[0,0],shadow:{x:0,z:0,halfX:plane*.9,halfZ:plane*.9,soft:plane*.35,strength:.5}});
}
function stackLayoutFor(list){
  const key=list.map(r=>r.run_id).join('|');
  if(key!==stackLayoutKey){stackLayoutKey=key;stackLayout=comparisonLayout(data.personas,list.flatMap(r=>r.edges),[]);stackSpread=null;}
  return stackLayout;
}
// The 3D planes use the same layout spread evenly over the plane (direction and distance order kept).
let stackSpread=null;
function stackPlaneLayout(list){const layout=stackLayoutFor(list);if(!stackSpread)stackSpread=spreadLayout(layout);return stackSpread;}
function sync3D(changed){
  const s=S(),canvas=stage?.canvas;
  if(!stage)return;
  if(s.view!=='atlas'&&[...changed].every(key=>key==='focusSlot'))return;
  canvas.dataset.view=s.view;canvas.dataset.layers=String(runs.length);
  const viewChanged=s.view!==lastView,enc=encodings();
  if(viewChanged)hideTooltip();
  if(s.view==='atlas'){
    ensureAtlas();
    stack.setVisible(false);atlas.setVisible(true);
    if(viewChanged||changed.has('filters')||changed.has('init'))atlas.setMatch(filterMask(s.filters));
    atlas.setFocus(s.focusSlot);
    atlas.setStacked(runs.map(run=>run.slot));
    atlas.setOpacity(Number($('opacity')?.value||50)/100*.6);
    if(viewChanged||['colorBy','persona','groups','lens','init'].some(k=>changed.has(k))){
      const s2=S();
      atlas.setEncoding({colorOf:id=>s2.colorBy==='layer'?STAGE_COLORS.accent:enc.node.colorOf(id,0),dimmed:enc.node.dimmed,persona:s2.persona,classOf:s2.lens!=='none'?enc.edge.classOf:null});
    }
    canvas.dataset.planes=String(atlas.slots);canvas.dataset.atlas=JSON.stringify(atlas.drawables);
    if(viewChanged){
      environment('atlas');labels.set(atlasLabels());
      if(lastView&&!stage.reducedMotion&&activeRun()){const tower=atlas.towerOf(activeRun().slot);const close=atlas.towerFraming(tower.index);stage.flyTo(close,{instant:true});}
      stage.pose('orbit');
    }
    lastView=s.view;return;
  }
  atlas?.setVisible(false);
  const replay=s.view==='replay',run=activeRun(),list=replay?[run]:runs;
  const frame=replay&&trace?trace[currentFrame()]:null;
  // Tall stacks get a little more room between planes; the camera also looks flatter at them (environment()).
  const spacing=Number($('separation')?.value||55)*(replay||runs.length<=4?1:1.12);
  stack.configure({runs:list,activeId:run.run_id,layout:stackPlaneLayout(runs),
    options:{replay,spacing,scale:replay?150*.88:110*.88,plane:replay?150:110,lift:view3d.lift,opacity:Number($('opacity')?.value||50)/100},
    frame:replay?(frame?{edges:frame.edges,event:hasStepRecord(run)?frame.event:null}:null):null,animate:animateNext,duration:Math.min(550,Number($('speed')?.value||800)*.7),
    persona:s.persona,isolate:incidentOnly(),ties:replay?'all':$('ties')?.value||'all',ribbons:view3d.ribbons,encoding:enc,
    consensus:!replay&&s.consensus&&consensusLive?consensusPlane():null});
  stack.setVisible(true);
  canvas.dataset.planes=String(stack.planes);
  const planeKey=`${s.view}|${list.map(r=>r.run_id).join(',')}|${spacing}|${view3d.lift}|${stack.consensusPlane}`;
  if(planeKey!==lastPlaneKey||viewChanged){
    lastPlaneKey=planeKey;environment(s.view);
    if(viewChanged||changed.has('layers')||changed.has('active')&&replay||changed.has('init')){
      if(entering==='tower'&&!stage.reducedMotion){const target=stage.poseTarget('orbit');stage.flyTo({position:new THREE.Vector3(...target.position).multiplyScalar(1.9).add(new THREE.Vector3(0,120,0)),target:target.target},{instant:true});}
      entering=null;stage.pose('orbit');
    }
  }
  labels.set(stackLabels(list,frame));
  lastView=s.view;
}
// Aggregate plane (store key `consensus`): ties present in at least `min` of the stacked or the filtered graphs.
function consensusScope(){const setting=S().consensus;return !setting?[]:setting.scope==='stack'?runs:applyFilters(data.runs,S().filters);}
function consensusPlane(){
  const setting=S().consensus,scope=consensusScope();if(!setting||!scope.length)return null;
  const floor=Math.max(1,Math.ceil(setting.min*scope.length-1e-9)),list=consensusOf(scope).filter(row=>row.count>=floor);
  return {list,total:scope.length};
}
const consensusCaption=()=>{const setting=S().consensus,n=consensusScope().length;return `Aggregate of ${plural(n,setting?.scope==='stack'?'stacked graph':'filtered graph')} · ties in ≥ ${Math.round((setting?.min||0)*100)}% · not a run`;};
function atlasLabels(){
  const compact=stage.size.width<700,codes={us:'US',india:'IN',japan:'JP',brazil:'BR',english:'EN',hindi:'HI',japanese:'JA',portuguese:'PT'};
  // Full method names need about 45 px per column; below that each column gets its initial (key in the legend).
  const letters={global:'G',local:'L',sequential:'S',iterative:'I'},short=methodLetters();
  const text={model:modelName,method:key=>short?letters[key]||methodName(key):methodName(key),setting:([c,l])=>compact?`${codes[c]||c} / ${codes[l]||l}`:settingName(c,l),seed:seed=>String(seed),'seed-axis':()=>'Seed order ↑\nnot time'};
  const priority={model:6,setting:5,'seed-axis':5.5,seed:2,method:4},anchor={model:'center',method:'center',setting:'left',seed:'right','seed-axis':'right'};
  const seeds=data.design.seeds,ends=new Set([seeds[0],seeds.at(-1)]);
  // Every method name is shown under its own column (two staggered lines on wide canvases, initials otherwise);
  // seeds show only the first and last plane so the axis never crowds; setting names sit at the end of their own row.
  return atlas.layout.axes.filter(axis=>!((axis.kind==='seed'||axis.kind==='seed-axis')&&compact)&&!(axis.kind==='seed'&&!ends.has(axis.key))).map(axis=>({id:`axis-${axis.id}`,text:text[axis.kind](axis.key),
    title:axis.kind==='setting'?settingName(...axis.key):axis.kind==='method'?methodName(axis.key):undefined,position:short&&axis.kind==='method'?[axis.position[0],0,axis.position[2]-(axis.stagger?6.8:0)]:axis.position,
    priority:priority[axis.kind],anchor:anchor[axis.kind],className:`g3-axis g3-axis-${axis.kind}${compact||short&&axis.kind==='method'?' g3-compact':''}`}));
}
const methodLetters=()=>!!stage&&stage.size.width<1000;
function stackLabels(list,frame){
  const s=S(),items=[],replay=s.view==='replay',selected=s.persona,mode=$('node-labels')?.value||'none';
  const event=replay?frame?.event:null;
  if(!replay)list.forEach((run,index)=>{const anchor=stack.planeAnchor(index);if(anchor)items.push({id:`plane-${run.run_id}`,text:layerLabel(runs,index),position:anchor,priority:4,cull:false,clamp:true,anchor:['left','right'],column:'planes',leader:true,color:LAYER_COLORS[index],title:label(run),ariaLabel:`Replay layer ${index+1}: ${label(run)}`,className:'plane-label run-label g3-plane',pressed:run.run_id===s.active,onClick:()=>activate(run)});});
  if(!replay&&stack.consensusPlane>=0){items.push({id:'plane-consensus',text:consensusCaption(),position:stack.planeAnchor(stack.consensusPlane),priority:4,cull:false,clamp:true,anchor:['left','right'],color:STAGE_COLORS.gold,className:'plane-label run-label g3-plane g3-consensus'});}
  list.forEach((run,index)=>{
    const edges=replay&&run.run_id===s.active&&frame?frame.edges:run.edges;
    const ego=personaView(edges,selected,incidentOnly()),adjacent=ego.neighbors;
    const removedNeighbors=new Set((event?.removed||[]).filter(edge=>selected&&edge.includes(selected)).flat());
    for(const person of data.personas){
      if(!stack.nodeVisible(index,person.id))continue;
      const actor=event?.persona===person.id,focus=person.id===selected||actor;
      if(!(person.id===selected||mode==='all'||mode==='auto'&&(!selected||focus||adjacent.includes(person.id)||removedNeighbors.has(person.id))))continue;
      const description=personaTag(person.id,selected,actor?person.id:null,currentFrame());
      items.push({id:`p-${run.run_id}-${person.id}`,text:person.id===selected?`Persona ${person.id} / ${adjacent.length} neighbors`:focus?description:person.id,position:stack.nodePosition(index,person.id),
        priority:focus?3:0,cull:mode==='auto'&&!focus,ariaLabel:description,color:actor?'#ffbd82':'#80e2c4',className:'plane-label persona-label',data:{person:person.id,run:run.run_id}});
    }
  });
  if(!replay&&selected&&list.length>1){const top=stack.nodePosition(list.length-1,selected);if(top)items.push({id:'thread',text:`Persona ${selected}: same persona, not a tie`,position:[top[0],top[1]+8,top[2]],priority:3,cull:false,clamp:true,anchor:'above',className:'g3-thread'});}
  return items;
}

// ---------------------------------------------------------------------------------------------
// Stage HUD (#stage-overlay): view caption, encoding toggles, atlas key. No population counts here (the ledger).
function buildHud(){
  const root=$('stage-overlay');if(!root||!stage)return;
  root.replaceChildren();
  const hud=document.createElement('div');hud.className='g3-hud';
  const caption=document.createElement('p');caption.className='g3-caption';caption.id='g3-caption';
  const key=document.createElement('div');key.className='g3-key';key.id='g3-key';
  const tools=document.createElement('div');tools.className='g3-tools';tools.setAttribute('role','group');tools.setAttribute('aria-label','3D encodings');
  const toggle=(id,text,title,onClick)=>{const button=document.createElement('button');button.type='button';button.id=id;button.textContent=text;button.title=title;button.setAttribute('aria-pressed','false');button.onclick=onClick;tools.append(button);return button;};
  toggle('g3-lift','Lift by degree','Raise each node on a stem by its number of ties in that graph. Off: nodes sit on their plane.',()=>{view3d.lift=!view3d.lift;render(new Set(['local']));});
  toggle('g3-ribbons','Shared-tie ribbons','Draw a translucent ribbon between adjacent planes for every tie present in both.',()=>{view3d.ribbons=!view3d.ribbons;render(new Set(['local']));});
  toggle('g3-consensus','Aggregate plane','Add an aggregate plane above the stack: ties present in at least half of the filtered graphs, brighter when in more of them. Descriptive; not a run. Scope and share are set under Compare.',()=>{const on=!S().consensus;commit({consensus:on?{scope:'filters',min:.5}:null});announce(on?'Aggregate plane added above the stack: ties present in at least half of the filtered graphs. It is not a run.':'Aggregate plane removed.');});
  hud.append(caption,key,tools);root.append(hud);
  stage.observeChrome?.(hud);
}
function paintHud(){
  const s=S(),caption=$('g3-caption'),key=$('g3-key');if(!caption){paintTools();return;}
  caption.textContent='';caption.hidden=true;
  key.replaceChildren();
  const swatch=(cls,text)=>{const span=document.createElement('span');span.className=`g3-swatch ${cls}`;span.textContent=text;key.append(span);};
  if(s.view==='atlas'){swatch('g3-valid','Graph · brighter = more ties');swatch('g3-empty','Valid graph, no ties');swatch('g3-failed','Failed attempt, no graph');swatch('g3-stacked','In your stack');swatch('g3-zoom','Zoom into a tower to see nodes and ties');}
  else if(s.view==='stack'){const aggregate=!!s.consensus&&stack?.consensusPlane>=0;if(view3d.ribbons&&runs.length>1&&!aggregate)swatch('g3-ribbon','Ribbon: tie shared with next plane');if(aggregate)swatch('g3-consensus-key',`Top plane: aggregate, brighter = in more of the ${s.consensus.scope==='stack'?'stacked':'filtered'} graphs`);if(view3d.lift)swatch('g3-lift','Height above plane: ties of that node');}
  else if(view3d.lift)swatch('g3-lift','Height above plane: ties of that node');
  key.hidden=!key.childElementCount;
  paintTools();
}
function paintTools(){
  const s=S(),set=(id,pressed,hidden,disabled=false)=>{const b=$(id);if(!b)return;b.setAttribute('aria-pressed',String(pressed));b.hidden=hidden;b.disabled=disabled;};
  set('g3-lift',view3d.lift,s.view==='atlas'||!stage);
  set('g3-ribbons',view3d.ribbons,s.view!=='stack'||!stage,runs.length<2);
  set('g3-consensus',!!s.consensus,s.view!=='stack'||!stage||!consensusLive);
  const rotate=$('rotate');
  if(rotate){rotate.disabled=!stage||stage.reducedMotion;rotate.setAttribute('aria-pressed',String(!!stage?.controls.autoRotate));rotate.title=stage?.reducedMotion?'Auto-orbit is off because reduced motion is requested':'Slowly orbit the camera, not the recorded data';}
}

// ---------------------------------------------------------------------------------------------
// DOM panels (legacy ids). Painted from the store on every change.
function personaCard(person,context){
  const card=document.createElement('div'),title=document.createElement('strong'),list=document.createElement('dl');card.className='persona-card';title.textContent=`Persona ${person.id}`;card.append(title);
  if(context){const source=document.createElement('span');source.className='persona-context';source.textContent=context;card.append(source);}
  for(const [key,value] of Object.entries(person.attributes)){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;if(key===S().colorBy)dd.classList.add('colored-by');list.append(dt,dd);}
  card.append(list);return card;
}
function paintModes(){
  const s=S(),selected=s.persona;
  $('atlas-mode')?.setAttribute('aria-pressed',String(s.view==='atlas'));
  $('formation-mode')?.setAttribute('aria-pressed',String(s.view==='replay'&&!journey&&!(incidentOnly()&&selected)));
  $('compare-mode')?.setAttribute('aria-pressed',String(s.view==='stack'));
  document.body.dataset.stageView=s.view;
}
function paintLayers(){
  const s=S(),list=$('layers');if(!list)return;
  const focused=document.activeElement?.classList.contains('select-run')?document.activeElement.getAttribute('aria-label'):null;
  list.replaceChildren();
  runs.forEach((run,index)=>{
    const li=document.createElement('li');li.style.setProperty('--layer',LAYER_COLORS[index]);li.classList.toggle('active',run.run_id===s.active);
    const title=document.createElement('button');title.className='select-run';title.setAttribute('aria-pressed',String(run.run_id===s.active));title.title=label(run);
    const parts=runParts(run);
    title.textContent=`${index+1}. ${parts.model}\n${parts.method} / ${parts.setting}\n${parts.seed}\n${run.events?.length?plural(run.events.length,'recorded update'):run.events_url?'Recorded replay available':'Final graph only'}`;
    title.onclick=()=>activate(run);
    title.setAttribute('aria-label',`Inspect run: ${title.textContent.replace(/\n/g,', ')}`);li.append(title);
    const remove=document.createElement('button');remove.textContent='×';remove.className='remove-run';remove.setAttribute('aria-label',`Remove layer ${index+1}`);remove.disabled=runs.length===1;remove.onclick=()=>removeLayer(index);li.append(remove);
    list.append(li);
  });
  if($('layer-count'))$('layer-count').textContent=`${runs.length}/${STACK_LIMIT}`;
  if($('undo'))$('undo').disabled=!removed.length||runs.length>=STACK_LIMIT;
  if(focused)[...list.querySelectorAll('.select-run')].find(button=>button.getAttribute('aria-label')===focused)?.focus({preventScroll:true});
}
function paintSelection(){
  const s=S(),selected=s.persona,comparison=layerComparison(runs),replay=s.view==='replay';
  if($('person')&&$('person').value!==selected)$('person').value=selected;
  if($('color')&&$('color').value!==s.colorBy&&[...$('color').options].some(o=>o.value===s.colorBy))$('color').value=s.colorBy;
  const rows=$('rows'),degrees=$('person-degrees'),design=comparisonDesign(runs);rows?.replaceChildren();degrees?.replaceChildren();
  if($('rows-axis'))$('rows-axis').textContent=design.axis;
  runs.forEach((run,index)=>{
    const renderedEdges=replay&&run.run_id===s.active&&trace?trace[currentFrame()].edges:run.edges,adjacent=personaView(renderedEdges,selected,incidentOnly()).neighbors;
    if(degrees&&selected&&(!replay||run.run_id===s.active)){const item=document.createElement('p');item.textContent=`Run ${index+1} / ${modelName(run.model)}: ${adjacent.length} ${replay?'current':'final'} neighbors${adjacent.length?'':' (isolated)'}`;item.style.borderLeft=`3px solid ${LAYER_COLORS[index]}`;for(const id of [...adjacent].sort((a,b)=>Number(a)-Number(b))){const button=document.createElement('button');button.className='neighbor';button.textContent=id;button.title=`Inspect persona ${id} across selected runs`;button.onclick=()=>{selectPersona(id,false);$('person')?.focus({preventScroll:true});};item.append(button);}degrees.append(item);}
    const finalAdjacent=selected?[...neighbors(run.edges,selected)].sort():[];
    if(rows){const row=document.createElement('tr');for(const value of [design.labels[index]+(run.run_id===s.active?' (active)':''),data.personas.length,run.edges.length,fmt(run.metrics.density),fmt(run.metrics.avg_clustering_coef),selected?`${finalAdjacent.length}: ${finalAdjacent.sort((a,b)=>a-b).join(', ')||'none'}`:'—']){const td=document.createElement('td');td.textContent=value;row.append(td);}rows.append(row);}
  });
  const person=data.personas.find(p=>p.id===selected);
  if($('person-detail')){if(person)$('person-detail').replaceChildren(personaCard(person));else $('person-detail').textContent='Choose a node or persona to compare their neighbors.';}
  if(stage&&$('status'))$('status').textContent=s.view==='atlas'?'':`${plural(runs.length,'layer')} · ${data.personas.length} people per layer · ${plural(comparison.shared,'tie')} shared across all layers`;
  if($('comparison'))$('comparison').textContent=overlapLine(comparison.shared,comparison.union,runs.length);
  const summary=$('quick-summary');
  if(summary){
    summary.replaceChildren();
    const pc=selected?personaComparison(runs,selected):null;
    const stats=pc?[['Selected persona',selected],['Distinct neighbors',pc.union],['Neighbors in every run',pc.shared],['Neighbor overlap',runs.length<2?'Add 2nd run':pc.jaccard===null?'No neighbors':`${(100*pc.jaccard).toFixed(1)}%`]]:[['Layers',runs.length],['People / layer',data.personas.length],['Shared ties',runs.length>1?comparison.shared:'Add 2nd run'],['Intersection / union',runs.length<2?'Add 2nd run':comparison.union?`${(100*comparison.shared/comparison.union).toFixed(1)}%`:'No ties']];
    for(const [name,value] of stats){const item=document.createElement('div'),number=document.createElement('strong'),caption=document.createElement('span');number.textContent=value;caption.textContent=name;item.append(number,caption);summary.append(item);}
  }
  const varies=['model','method','culture','language','seed'].filter(key=>new Set(runs.map(r=>r[key])).size>1);
  if($('comparison-scope')){
    $('comparison-scope').textContent=runs.length===1?'Add a second run to compare edge overlap.':`Varies: ${varies.join(', ')||'study or source variant'}. Overlap is descriptive, not a significance test.`;
    if(new Set(runs.map(r=>r.prompt_variant??'historical')).size>1)$('comparison-scope').textContent+=' Source variants also differ; this is not a controlled model-only test.';
    if(runs.some(r=>r.language_confound))$('comparison-scope').textContent+=' Historical language confounding is present.';
  }
  if($('warning')){
    $('warning').textContent=runs.some(r=>r.language_confound)?'Historical language confounding is present: instructions and persona language were not cleanly separated.':'';
    if(new Set(runs.map(r=>r.prompt_variant??'historical')).size>1)$('warning').textContent+=' Selected runs use different prompt/source variants; this comparison is descriptive.';
  }
  const metric=$('metric')?.value||'density';
  const measured=BASIC_METRICS.has(metric)||metricsReady();
  const reference=activeRun();
  if($('plot'))$('plot').innerHTML=measured?metricChartHtml(runs,metric,{activeId:reference.run_id,title:topologyMetrics[metric]}):metricsNote();
  if($('metric-deltas'))$('metric-deltas').textContent=!measured||runs.length<2?'':`“vs active” is each run minus the active run (${design.labels[runs.indexOf(reference)]}). Descriptive, not a significance test.`;
  const definitions={density:'Ties present out of all possible persona pairs. Higher means denser, not better.',avg_clustering_coef:'How often two neighbors of a persona are also tied. Higher means more closed triangles, not greater accuracy.',prop_nodes_lcc:'Share of personas in the largest connected group.',modularity:'Strength of community structure against the measure’s null model. Not a validation score.',global_efficiency:'Mean inverse shortest-path length; disconnected pairs count as zero.',num_components:'Number of separate groups with no path between them.'};
  if($('metric-definition'))$('metric-definition').textContent=definitions[metric]||'';
  if($('ties'))$('ties').disabled=replay||s.view==='atlas';
  if($('separation')){$('separation').disabled=s.view!=='stack'||runs.length<2;if($('spacing-value'))$('spacing-value').textContent=$('separation').disabled?'(comparison only)':$('separation').value;}
  if($('clear-person'))$('clear-person').disabled=!selected;if($('incident'))$('incident').disabled=!selected;
  if($('focus-person'))$('focus-person').disabled=!selected||!stage||s.view==='atlas';
  for(const button of document.querySelectorAll('[data-match]')){const matches=matchedRuns(data.runs,reference,button.dataset.match);button.disabled=matches.length<2||matches.length>STACK_LIMIT||(button.dataset.match==='culture'&&reference.language!=='english');}
  renderEvidence();
}
function renderPlayback(){
  const s=S(),run=activeRun(),selected=s.persona,replay=s.view==='replay',atlasView=s.view==='atlas',frame=currentFrame(),available=!!trace&&replay;
  const pending=!!run.events_url&&!traceCache.has(run.run_id),failed=!!run.events_url&&traceCache.has(run.run_id)&&!trace;
  const current=trace?.[frame],event=replay?current?.event:null,edges=current?.edges||run.edges;
  const set=(id,text)=>{const el=$(id);if(el)el.textContent=text;};
  set('stage-title',replay?label(run):atlasView?`${data.counts.cells??112} conditions × ${data.design.seeds.length} seed planes`:`${plural(runs.length,'final network')} · shared persona positions`);
  const single=replay&&!hasStepRecord(run),total=trace?trace.length-1:0,more=steps().some(step=>step>frame);
  document.body.dataset.replayRecord=!replay?'':single?'single':'steps';
  set('stage-state',atlasView?'Every primary attempt slot':!replay?'Final snapshots':single?'One response':trace?`Event ${frame} of ${total}`:pending?'Loading verified replay…':'History unavailable');
  const playButton=$('play');
  if(playButton){
    const text=playing?'Pause':!replay?'Play active run':journey?`Play persona ${selected}`:available&&!more&&frame>0?'Play again':'Play';
    if($('play-text'))$('play-text').textContent=text;else playButton.textContent=text;
    playButton.dataset.state=playing?'playing':'paused';playButton.disabled=!trace||single;
    playButton.title=playing?'Pause (Space)':available&&!more&&frame>0?'Replay the recorded events from the start (Space)':'Play the recorded events in generation order (Space)';
  }
  for(const id of ['rewind','previous','next','timeline','show-final','speed'])if($(id))$(id).disabled=!available||single;
  if($('previous'))$('previous').disabled=!available||single||frame===0;if($('next'))$('next').disabled=!available||single||!more;
  if($('rewind'))$('rewind').disabled=!available||single||frame===0;if($('show-final'))$('show-final').disabled=!available||single||frame>=total;
  paintTicks(run,selected,replay&&!single?frame:-1);
  paintSingle(run,single);
  const play=$('play-persona');
  if(play){play.disabled=!trace||!selected||!personaSteps(run.events,selected).length;play.textContent=journey?`Following persona ${selected}`:selected?`Follow persona ${selected}`:'Select a persona to follow';play.setAttribute('aria-pressed',String(journey));}
  if($('timeline')){$('timeline').max=trace?trace.length-1:0;$('timeline').value=replay?frame:trace?trace.length-1:0;}
  set('step-label',!replay?'Final networks':trace?`Event ${frame} of ${total}`:pending?'Loading recorded events…':'No recorded history');
  if($('timeline')&&trace)$('timeline').setAttribute('aria-valuetext',`Event ${frame} of ${total}`);
  const relevantSteps=personaSteps(run.events,selected);
  set('playback-scope',!replay?'Saved outputs / no shared timeline':journey?`Following persona ${selected}: ${relevantSteps.filter(step=>step<=frame).length} of ${relevantSteps.length} of its events`:incidentOnly()&&selected?`Only persona ${selected}'s ties visible`:trace&&run.events?`${phaseLine(run.events)} · generation order, not time`:'Generation order, not time');
  const legend=document.querySelector('.canvas-legend');
  if(legend){
    legend.innerHTML=atlasView?`<span>Columns: model × method${methodLetters()?' (G global · L local · S sequential · I iterative)':''}</span>`+'<span>Rows: country framing / language</span><span>Tower height: seed order, not time</span>':!replay?'<span>Each plane: one saved run, not a time slice</span>':single?'<span>Saved graph of one response; no recorded order within it</span>':'<span class="new-key">Added in this event</span><span class="removed-key">Removed: dashed</span><span>Ring: acting persona</span>';
  }
  const note=$('replay-note');
  if(note){
    note.textContent=!replay?'':!trace?'Playback unavailable: only the final graph exists. Its history cannot be reconstructed honestly.':single?'':`${run.method==='sequential'?'Sequential: one persona is asked per recorded update.':run.method==='local'?'Local: each recorded update holds one persona\'s choices.':'Iterative: each recorded update adds and removes ties.'} Ties within one update appear together because their internal order was not recorded. Not simultaneous agents.`;
    if(replay&&pending)note.textContent='Loading and verifying this run’s recorded events. Final graph remains visible; playback will become available after the checks pass.';
    if(replay&&failed)note.textContent='Recorded history could not be verified. Playback is disabled; the saved final graph remains available. Reload the page to retry.';
    if(journey&&available)note.textContent+=' Following a persona skips unrelated updates on screen; the graph still includes them.';
    if(replay&&['primary','revised_calibration'].includes(run.study)&&run.method==='global'&&!run.edges.length)note.textContent+=' This valid run has no ties: all 50 personas remain. No connection history exists to animate.';
  }
  set('event-heading',!replay?'Final comparison':pending?'Verifying replay':failed?'Replay unavailable':!trace?'No event log':!event?'Before the first decision':event.persona==null?'Global batch':`Event actor: persona ${event.persona}`);
  const incidentChange=event&&selected&&[...event.added,...event.removed].some(edge=>edge.includes(selected));
  const relevance=!event||!selected?'':event.persona===selected?' The selected persona made this recorded decision.':incidentChange?` ${event.persona==null?'The batch':'Another persona'} changed a tie involving persona ${selected}.`:' This event does not change the selected persona\'s ties.';
  set('event-explanation',!replay?'Select Replay to inspect the active run.':!trace?'Only the final edges are available.':!event?'Press Play or choose a decision below.':`${methodName(event.method)}: ${plural(event.added.length,'new tie')}; ${event.removed.length} removed.${!event.added.length&&!event.removed.length?' No net change.':''}${relevance}${event.attempts?` Recorded attempts: ${event.attempts}.`:''} Ties are undirected.`);
  if(replay)paintEventCard(run,{event,frame,total,ties:edges.length,single,pending,failed});
  else{const card=$('event-delta');if(card){card.className='';card.textContent=atlasView?(matchMedia('(hover: none)').matches?'Tap a tower to stack its seeds':'Hover a tower · click to stack its seeds'):'';}}
  set('active-badge',atlasView?'Atlas · every primary attempt slot':!replay?'Stack · saved final graphs':single?`Run ${runs.indexOf(run)+1} of ${runs.length} · one recorded response`:`Replaying run ${runs.indexOf(run)+1} of ${runs.length} · generation order`);
  const changes=$('event-changes');
  if(changes){changes.replaceChildren();if(event)for(const [kind,list] of [['add',event.added],['remove',event.removed]]){
    const relevant=selected?list.filter(edge=>edge.includes(selected)):list;
    if(relevant.length){const p=document.createElement('p');p.className=`change-${kind}`;p.textContent=`${kind==='add'?'Added':'Removed'}${selected?' involving selected persona':''}: ${relevant.map(edge=>edge.join(' - ')).join(', ')}`;changes.append(p);}
  }}
  if(replay){
    const degree=selected?neighbors(edges,selected).size:null,summary=$('quick-summary');
    if(summary){summary.replaceChildren();for(const [name,value] of [['Current ties',edges.length],['Current density',fmt(2*edges.length/(data.personas.length*(data.personas.length-1)))],['Persona neighbors',degree??'--'],['Recorded event',trace?frame:'--']]){const box=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=name;box.append(strong,span);summary.append(box);}}
    set('status',`${data.personas.length} roster members / ${plural(edges.length,'current tie')}${selected?` / persona ${selected}: ${degree} neighbors`:''}`);
  }
  if(!stage)set('status','WebGL unavailable. Playback counts, the ledger and tables remain available.');
  set('trajectory-title',selected?`Persona ${selected}’s ties over the run`:'Network ties over the run');
  if($('trajectory-purpose'))$('trajectory-purpose').textContent=`Active run: ${label(run)}. The cursor marks the ${replay?'replay position':'final state'}.`;
  if($('trajectory'))$('trajectory').innerHTML=trajectoryChartHtml(trace,selected,!replay&&trace?trace.length-1:frame,{width:$('trajectory').clientWidth||680});
  const events=$('events'),key=`${run.run_id}|${selected}|${run.events?.length||0}|${pending}`;
  if(events&&ledgerKey!==key){ledgerKey=key;events.replaceChildren();
    const rowsLedger=trace||run.events?ledgerRows(run.events||[],selected):[],total=run.events?.length||0;
    set('ledger-count',selected?`${rowsLedger.length} of ${plural(total,'event')}`:plural(total,'event'));
    if($('ledger-purpose'))$('ledger-purpose').textContent=selected?`Recorded events that changed persona ${selected}’s ties; counts are ties involving persona ${selected}. Select a row to replay to it.`:'Every recorded event of the active run, in generation order. Select a row to replay to it.';
    if(!rowsLedger.length){const p=document.createElement('p');p.className='empty-note';p.textContent=pending?'Verifying recorded events…':failed?'Recorded events unavailable; playback is disabled.':trace?`No recorded event involves persona ${selected}.`:'No event log was saved for this run.';events.append(p);}
    else{const head=document.createElement('div');head.className='event-row event-head';head.setAttribute('aria-hidden','true');for(const text of ['Step','Persona','Phase','Added','Removed']){const span=document.createElement('span');span.textContent=text;head.append(span);}events.append(head);}
    for(const row of rowsLedger){const button=document.createElement('button');button.type='button';button.className='event-row'+(row.pair?` pair-${row.pair}`:'');button.dataset.step=row.step;
      const cells=[String(row.step),row.persona??'All',row.phase,row.added?`+${row.added}`:'0',row.removed?`−${row.removed}`:'0'];
      cells.forEach((text,i)=>{const span=document.createElement('span');span.textContent=text;if(i===1&&row.persona===selected&&selected)span.className='is-selected';if(i===3&&row.added)span.className='add';if(i===4&&row.removed)span.className='remove';button.append(span);});
      button.dataset.label=`step ${row.step}, ${row.persona==null?'global batch':'persona '+row.persona}, ${row.phase}, ${row.added} added, ${row.removed} removed${selected?` involving persona ${selected}`:''}`;
      button.title=`Whole event: ${plural(row.batchAdded,'tie')} added, ${row.batchRemoved} removed. Order within an event was not recorded.`;
      button.onclick=()=>{if(S().view!=='replay')commit({view:'replay'});seek(row.step);};events.append(button);}
  }
  if(events)for(const button of events.querySelectorAll('button')){button.setAttribute('aria-current',String(Number(button.dataset.step)===(!replay&&trace?trace.length-1:frame)));button.setAttribute('aria-label',`Replay to ${button.dataset.label}`);}
}
// The card beside the stage: which event the cursor is on and what it did (persona, phase, ties added and removed).
function paintEventCard(run,{event,frame,total,ties,single,pending,failed}){
  const card=$('event-delta');if(!card)return;
  const el=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
  card.className='event-card';card.replaceChildren();
  if(replayNotice&&replayNoticeFor===run.run_id)card.append(el('span','ev-notice',replayNotice));
  if(single){
    card.append(el('span','ev-pos','One response'),el('strong','ev-actor','Whole network'),el('span','ev-change',`${plural(run.edges.length,'tie')} returned together${run.edges.length?'':' (a valid, empty graph)'}`));
    return;
  }
  if(!trace){card.append(el('span','ev-pos',pending?'Loading recorded events…':failed?'Replay unavailable':'No event log'));return;}
  const info=eventSummary(event,{frame,total,ties});
  card.append(el('span','ev-pos',info.position));
  const who=el('span','ev-who');who.append(el('strong','ev-actor',info.actor),el('span','ev-phase',info.phase));card.append(who);
  const change=el('span','ev-change');
  if(!event)change.append(el('span','',playing?'No ties yet':'No ties yet · press Play'));
  else{change.append(el('b','add',`+${info.added} added`),el('b',info.removed?'remove':'none',`−${info.removed} removed`),el('span','',`${plural(ties,'tie')} now`));}
  card.append(change);
  if(info.list.length){const list=el('span','ev-ties');for(const tie of info.list)list.append(el('i',tie.kind,tie.text));if(info.more)list.append(el('i','more',`+${info.more} more`));card.append(list);}
}
// Scrubber ticks: one per recorded event (height = ties changed). Rebuilt only when the run or persona changes.
function paintTicks(run,selected,frame){
  const host=$('timeline-ticks');if(!host)return;
  const events=frame>=0&&trace?run.events:null,key=events?`${run.run_id}|${selected}|${events.length}`:'';
  if(key!==ticksKey){ticksKey=key;const svg=events?tickStripSvg(events,{persona:selected}):'';host.innerHTML=svg+svg;}
  // Events up to the cursor are drawn at full strength.
  host.style.setProperty('--done',events?`${100*frame/Math.max(1,events.length)}%`:'0%');
}
// A run recorded as one response has nothing to step through: say so and offer what does work.
function paintSingle(run,single){
  const host=$('replay-single');if(!host)return;
  host.hidden=!single;
  const key=single?run.run_id:'';if(key===singleKey)return;singleKey=key;host.replaceChildren();
  if(!single)return;
  const info=singleRecordState(data.runs,run),text=document.createElement('p'),hint=document.createElement('p'),actions=document.createElement('div');
  text.className='single-text';text.id='replay-single-text';text.textContent=info.text;
  hint.className='hint';hint.textContent=info.hint;actions.className='single-actions';
  for(const option of info.alternatives){
    const other=runById.get(option.run_id),button=document.createElement('button');button.type='button';button.className='primary';button.dataset.replayAlt=option.method;button.textContent=option.label;
    button.title=`${label(other)} · same model, setting and seed`;
    button.onclick=()=>{const replaced=!runs.includes(other)&&runs.length>=STACK_LIMIT;activate(other,`Replaying the ${methodName(other.method)} run of the same model, setting and seed.`);announce(`Replaying ${label(other)} from event 0.${replaced?' The stack was full, so it now holds this run only.':''}`);$('play')?.focus({preventScroll:true});};
    actions.append(button);
  }
  const byPersona=document.createElement('button');byPersona.type='button';byPersona.id='single-by-persona';byPersona.textContent='View ties by persona';
  byPersona.title='Select the persona with the most ties in this graph and show only its ties; step through personas with [ and ]';
  byPersona.disabled=!run.edges.length;
  byPersona.onclick=()=>{const id=busiestPersona(run.edges);if(!id)return;selectPersona(id,false);if($('incident')){$('incident').checked=true;render(new Set(['local']));}$('tab-persona')?.click();announce(`Showing persona ${id}, which has the most ties in this graph. Use [ and ] to step through personas.`);};
  actions.append(byPersona);
  host.append(text,hint,actions);
}
function renderEvidence(){
  const run=activeRun(),selected=S().persona,select=$('research-question');
  const selectionKey=runs.map(r=>r.run_id).join('|'),summaryKey=selectionKey+'|'+selected;
  // A new stack selects the one question it can answer; the reader may still pick another (RQ2 fits any stack).
  if(select&&selectionKey!==questionKey){questionKey=selectionKey;const fit=comparisonReading(runs).suggested;if(fit&&[...select.options].some(o=>o.value===fit))select.value=fit;}
  const question=select?.value||'method',reading=comparisonReading(runs,question==='rq2'?'':question);
  if(summaryKey!==aggregateKey){aggregateKey=summaryKey;tieShowAll=false;}
  if($('aggregate-graph')){
    $('aggregate-graph').innerHTML=tieTableHtml(runs,{selected,showAll:tieShowAll})+(runs.length?`<details class="tie-matrix"><summary>Matrix view</summary>${frequencyMatrix(data.personas,runs,selected)}</details>`:'');
    const button=$('aggregate-graph').querySelector('[data-action="toggle-ties"]');if(button)button.onclick=()=>{tieShowAll=!tieShowAll;renderEvidence();$('aggregate-graph').querySelector('[data-action="toggle-ties"]')?.focus({preventScroll:true});};
  }
  if($('aggregate-note'))$('aggregate-note').textContent=`${selected?`Persona ${selected}’s ties`:'Every tie'} in the saved final graphs of ${plural(runs.length,'stacked run')}${reading.design.shared&&runs.length>1?` (${reading.design.shared})`:''}, sorted by how many runs contain it. Final graphs, not the replay position.`;
  const descriptions={seed:'Run-to-run variation: the same condition repeated with different seeds.',method:'Generation method: the procedure changes while model, setting and seed stay fixed.',rq1:'RQ1 · Country framing: does the framing change the graph with English instructions held fixed?',rq2:'RQ2 · Attribute mixing: which recorded attributes show within-group ties? Scores describe mixing, not causes.',rq3:'RQ3 · Model consistency: how similar are different models under matched conditions?',rq4:'RQ4 · Instruction language: does the language change the graph at a fixed country framing?'};
  if($('rq-explanation'))$('rq-explanation').textContent=question===reading.suggested?'':descriptions[question]||'';
  if($('rq-condition')){const parts=reading.design;$('rq-condition').replaceChildren();if(runs.length){for(const [k,v] of [['Held fixed',parts.shared||'nothing'],['Varies',runs.length<2?'nothing (one run)':`${parts.fields.map(f=>({setting:'country and language',culture:'country framing',language:'instruction language',method:'generation method'}[f]||f)).join(', ')||'nothing'} · ${plural(runs.length,'run')}`]]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;$('rq-condition').append(dt,dd);}}}
  if($('rq-status')){$('rq-status').textContent=reading.sentence;$('rq-status').className=reading.caveat?'selection-unmatched':'selection-matched';}
  const issues=$('rq-issues');
  if(issues){issues.replaceChildren();if(reading.caveat){const li=document.createElement('li');li.textContent=reading.caveat;issues.append(li);}issues.hidden=!reading.caveat;}
  if($('rq-homophily')){$('rq-homophily').hidden=question!=='rq2';$('rq-homophily').innerHTML=question!=='rq2'?'':metricsReady()?homophilyMatrix(runs):metricsNote();}
  const pairs=$('rq-pairs');
  if(pairs){pairs.replaceChildren();}
  if($('provenance-summary'))$('provenance-summary').textContent=data.verification;
  const fields=$('provenance-fields');
  if(fields){fields.replaceChildren();
    for(const [key,value] of [['Experiment',label(run)],['Collection','Final primary collection (supplements excluded)'],['Graph source',run.source],['Replay',trace?'Recorded deltas reconstruct the final saved graph':run.events_url?'History loads on demand and must pass its checksum':'Unavailable; final graph only'],['PNG',run.png_provenance||'Historical PNG-to-graph correspondence unknown; no verified image is offered here.'],['Analysis',run.analysis_version||data.analysis_version]]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;fields.append(dt,dd);}}
  const png=$('source-png');if(png){png.hidden=false;png.href='https://github.com/hemu77/UA_Capstone/blob/main/'+run.source.replace(/\.adj$/,'.png');}
  if($('provenance-hashes'))$('provenance-hashes').textContent=JSON.stringify({run_id:run.run_id,roster_sha256:run.roster_id,adjacency_sha256:run.source_sha256,png_sha256:run.png_sha256||null,receipt_sha256:run.receipt_sha256||null,protocol_sha256:run.protocol_sha256||null,generation_source_variant:run.prompt_variant||'unrecorded'},null,2);
}
function renderCoverage(){
  if($('primary-findings'))$('primary-findings').hidden=false;
  const host=$('coverage-table');if(!host)return;
  const table=document.createElement('table'),caption=document.createElement('caption'),head=document.createElement('thead'),body=document.createElement('tbody'),header=document.createElement('tr');
  caption.className='vh';caption.textContent='Valid graphs / planned attempts per condition. Select a cell to stack its valid seeds.';
  for(const text of ['Model / method',...data.settings.map(([country,language])=>settingName(country,language))]){const th=document.createElement('th');th.scope='col';th.textContent=text;header.append(th);}head.append(header);
  for(const model of data.models)for(const method of data.methods){
    const row=document.createElement('tr'),name=document.createElement('th');name.scope='row';name.textContent=`${modelName(model)} / ${methodName(method)}`;row.append(name);
    for(const [culture,language] of data.settings){
      const {saved:selected,planned,failed}=conditionCoverage(data,model,method,culture,language),td=document.createElement('td'),button=document.createElement('button');
      button.textContent=planned?`${selected.length} / ${planned}${failed.length?' · '+failed.length+' failed':''}`:'Not in scope';button.disabled=!selected.length;
      button.setAttribute('aria-label',`${modelName(model)}, ${methodName(method)}, ${settingName(culture,language)}: ${selected.length} saved of ${planned} allocated. Stack them.`);
      button.title=selected.length?'Stack every saved repetition in this condition':'Not collected; no generated graph exists';
      button.onclick=()=>openCondition(model,method,culture,language);
      td.append(button);row.append(td);
    }body.append(row);
  }table.append(caption,head,body);host.replaceChildren(table);
}

// ---------------------------------------------------------------------------------------------
// The one subscriber: derive local caches from the store, then repaint.
function render(changed,source=SOURCE){
  if(!data)return;
  if(source==='init')changed=new Set([...changed,'init']);
  const s=S();
  if(changed.has('layers')||changed.has('init')){
    const next=s.layers.map(id=>runById.get(id)),problem=stackProblem(next);
    if(problem){announce(problem);const previous=runs.map(r=>r.run_id);if(previous.length)queueMicrotask(()=>commit({layers:previous,active:runs.some(r=>r.run_id===s.active)?s.active:previous[0]}));return;}
    if(!keepUndo)removed=[]; // only Remove / Undo keep the undo history; any other replacement clears it
    runs=next;
  }
  if(!runs.some(run=>run.run_id===s.active)){
    const run=runById.get(s.active);
    // A run opened from elsewhere (result row "Replay") joins the stack, or replaces it when the stack is full.
    if(run){queueMicrotask(()=>commit({layers:runs.length<STACK_LIMIT?[...runs.map(r=>r.run_id),run.run_id]:[run.run_id],active:run.run_id}));}
    else if(runs.length)queueMicrotask(()=>commit({active:runs[0].run_id}));
    return;
  }
  if(changed.has('active')||changed.has('layers')||changed.has('init'))loadTrace();
  try{sync3D(changed);}catch(error){console.error(error);announce('3D view could not update: '+error.message);}
  animateNext=false;
  paintModes();
  paintHud();
  if([...changed].every(key=>key==='focusSlot'))return; // hover emphasis only: nothing else on the page depends on it
  const frameOnly=[...changed].every(key=>key==='frame'||key==='focusSlot'||key==='trace');
  if(!frameOnly||changed.has('trace')){paintLayers();paintSelection();}
  renderPlayback();
}

// ---------------------------------------------------------------------------------------------
// Initial state: new short links (layers-filters decodeState), legacy ?view=<json> links, or the default Atlas.
function defaultLayers(){return data.methods.map(method=>data.runs.find(r=>r.model==='gpt-4.1'&&r.method===method&&r.culture==='us'&&r.language==='english'&&r.seed===44000)).filter(Boolean).map(r=>r.run_id);}
function initialState(saved,params){
  const base={view:'atlas',layers:defaultLayers(),frame:FINAL,colorBy:'age',persona:''};
  base.active=base.layers[0]||data.runs[0].run_id;
  if(!base.layers.length)base.layers=[data.runs[0].run_id];
  const shortKeys=['v','m','p','s','b','g','q','l','a','n','c','x','i','f','k'];
  if(!saved&&shortKeys.some(k=>params.has(k))){
    try{
      const decoded=decodeState(params,data)||{};
      if(Object.keys(decoded).length){
        const state={...base,...decoded};
        if(!VIEWS.includes(state.view))state.view='atlas';
        if(stackProblem((state.layers||[]).map(id=>runById.get(id))))throw new Error('This link names runs that cannot be stacked. The default view is shown instead.');
        if(!state.layers.includes(state.active))state.active=state.layers[0];
        // A replay link without a position opens at the start of the record, like the Replay button.
        if(state.view==='replay'&&!('frame' in decoded))state.frame=replayStart(runById.get(state.active));
        return state;
      }
    }catch(error){if(/archived/i.test(error.message))throw error;loadNotice=error.message;}
    return base;
  }
  if(!saved)return base;
  const list=saved.runs.map(id=>runById.get(id));
  if(list.some(run=>!run))throw Error('The saved selection contains a run absent from this dataset. Open the default comparison to choose again.');
  const problem=stackProblem(list);if(problem)throw Error(problem);
  const layers=list.map(r=>r.run_id);
  const state={...base,layers,view:saved.mode==='formation'?'replay':saved.mode==='atlas'?'atlas':'stack',active:layers.includes(saved.active)?saved.active:layers[0],frame:Number.isSafeInteger(saved.frame)&&saved.frame>=0?saved.frame:saved.mode==='formation'?replayStart(runById.get(layers.includes(saved.active)?saved.active:layers[0])):FINAL};
  if(saved.person&&data.personas.some(p=>p.id===saved.person))state.persona=saved.person;
  if(typeof saved.color==='string'&&(saved.color==='layer'||saved.color in data.personas[0].attributes))state.colorBy=saved.color;
  if(saved.filters&&typeof saved.filters==='object')state.filters=legacyFilters(saved.filters);
  for(const id of ['ties','metric','research-question','node-labels'])if($(id)&&[...$(id).options].some(o=>o.value===saved[id]))$(id).value=saved[id];
  for(const id of ['separation','opacity'])if($(id)&&Number.isFinite(Number(saved[id]))&&saved[id]!==null&&saved[id]!=='')$(id).value=saved[id];
  if($('incident'))$('incident').checked=saved.incident===true;
  journey=saved.journey===true&&!!state.persona;
  return state;
}
function wireControls(){
  const on=(id,event,fn)=>{const el=$(id);if(el)el[`on${event}`]=fn;};
  on('undo','click',()=>{const last=removed.at(-1);if(!last)return;const next=runs.map(r=>r.run_id);next.splice(Math.min(last.index,next.length),0,last.run.run_id);keepUndo=true;removed.pop();if(!setStack(next,{},`Restored ${modelName(last.run.model)} / ${methodName(last.run.method)}.`))removed.push(last);keepUndo=false;});
  on('restore','click',()=>{setStack(startingRuns,{},'Restored the comparison present when you opened this page.');});
  on('person','change',()=>selectPersona($('person').value,false));
  on('focus-person','click',()=>{
    if(!stage||!S().persona||S().view==='atlas')return;
    const index=S().view==='replay'?0:Math.max(0,runs.findIndex(r=>r.run_id===S().active));
    const p=stack.nodePosition(index,S().persona);if(!p)return;
    const target=new THREE.Vector3(...p),offset=stage.camera.position.clone().sub(stage.controls.target);
    stage.markUserMoved();stage.flyTo({position:target.clone().add(offset),target},{duration:450});
  });
  on('clear-person','click',()=>selectPersona('',false));
  on('color','input',()=>commit({colorBy:$('color').value}));
  for(const id of ['ties','incident','node-labels','separation'])on(id,'input',()=>render(new Set(['local'])));
  on('opacity','input',()=>{const value=Number($('opacity').value)/100;stack?.setOpacity(value);atlas?.setOpacity(value*.6);});
  for(const id of ['metric','research-question'])on(id,'input',()=>paintSelection());
  on('atlas-mode','click',()=>{stop();journey=false;commit({view:'atlas'});});
  on('formation-mode','click',openReplay);
  on('compare-mode','click',()=>{stop();if(!commit({view:'stack'}).size)stage?.pose('orbit');});
  on('play','click',()=>play());
  on('play-persona','click',()=>{stop();journey=!journey;if($('incident'))$('incident').checked=journey;const patch={view:'replay'};if(journey)patch.frame=0;if(!commit(patch).size)render(new Set(['local']));});
  on('rewind','click',()=>seek(0,true));on('show-final','click',()=>seek(trace?.length-1||0));
  on('previous','click',()=>seek([...steps()].reverse().find(step=>step<currentFrame())||0,true));
  on('next','click',()=>seek(steps().find(step=>step>currentFrame())??currentFrame(),true));
  on('timeline','input',()=>seek(Number($('timeline').value)));
  on('speed','change',()=>{if(playing){clearTimeout(timer);timer=setTimeout(advance,Number($('speed').value));}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){stop();render(new Set(['local']));}});
  for(const id of ['orbit','top','front'])on(id,'click',()=>stage?.pose(id));
  for(const [id,factor] of [['zoom-in',.8],['zoom-out',1.25]])on(id,'click',()=>{
    if(!stage)return;const {camera,controls}=stage,offset=camera.position.clone().sub(controls.target);
    const distance=Math.min(controls.maxDistance,Math.max(controls.minDistance,offset.length()*factor));
    stage.markUserMoved();stage.flyTo({position:controls.target.clone().add(offset.setLength(distance)),target:controls.target.clone()},{duration:220});
  });
  on('rotate','click',()=>{if(!stage)return;stage.setAutoRotate(!stage.controls.autoRotate);paintTools();});
  // Share view: clipboard API, then a hidden-textarea copy; the label always returns to "Share view" after 2 s.
  let shareTimer=0;
  on('share','click',async()=>{
    const button=$('share'),href=location.href;let copied=false;
    if(!button.dataset.label)button.dataset.label=button.textContent; // the width is fixed in layers.css, so the header never jumps
    try{await navigator.clipboard.writeText(href);copied=true;}
    catch{
      try{const area=document.createElement('textarea');area.value=href;area.setAttribute('readonly','');area.setAttribute('aria-hidden','true');area.style.cssText='position:fixed;top:0;left:-9999px;opacity:0';
        document.body.append(area);area.select();copied=document.execCommand('copy');area.remove();button.focus({preventScroll:true});}catch{copied=false;}
    }
    button.textContent=copied?'Link copied':'Not copied';button.dataset.state=copied?'copied':'failed';
    announce(copied?'Link to this view copied.':'The link could not be copied automatically. Copy the URL from your address bar.');
    clearTimeout(shareTimer);shareTimer=setTimeout(()=>{button.textContent=button.dataset.label;delete button.dataset.state;},2000);
  });
  on('download-selection','click',async()=>{
    const selected=[...runs],owner=data,exportView={...S(),runs:selected.map(r=>r.run_id)};
    $('download-selection').disabled=true;
    try{
      await loadMetrics(owner); // the export carries every saved measurement, not only the two that ship with the index
      for(const run of selected)if(run.events_url)await verifiedHistory(run);
      if(data!==owner)throw Error('Dataset changed during export');
      const value={schema_version:1,dataset,scope:'Selected valid saved final graphs; not a population estimate. Histories are included only after verified replay loads.',view:exportView,personas:owner.personas,runs:selected};
      const objectUrl=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
      const link=document.createElement('a');link.href=objectUrl;link.download=`network-selection-${dataset}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(objectUrl),1000);
      announce(`Download requested: ${plural(selected.length,'saved network')} with verified recorded events, measurements and source hashes.`);
    }catch(error){announce('Export stopped: '+error.message+'. No incomplete download was created.');}
    finally{$('download-selection').disabled=false;}
  });
  on('expand-stage','click',()=>{const lab=document.querySelector('.laboratory');if(!lab)return;const expanded=lab.classList.toggle('expanded');$('expand-stage').setAttribute('aria-pressed',String(expanded));$('expand-stage').textContent=expanded?'Collapse graph':'Expand graph';});
  for(const button of document.querySelectorAll('[data-match]'))button.onclick=()=>{
    const reference=activeRun(),dimension=button.dataset.match,matches=matchedRuns(data.runs,reference,dimension);
    if(setStack(matches.map(r=>r.run_id),{view:'stack',active:reference.run_id},'Matched comparison applied. No missing conditions were filled.')){
      if($('research-question')){$('research-question').value={model:'rq3',culture:'rq1',language:'rq4',method:'method'}[dimension];paintSelection();}
      if($('match-note'))$('match-note').textContent=`${plural(matches.length,'saved run')} matched to ${modelName(reference.model)} / ${methodName(reference.method)} / ${settingName(reference.culture,reference.language)} / seed ${reference.seed}. Only ${dimension} varies among the matched fields.`;
    }
  };
}
async function start(){
  let saved;try{saved=JSON.parse(url.searchParams.get('view')||'null');}catch{saved=null;}
  if(!saved||typeof saved!=='object'||Array.isArray(saved)||!Array.isArray(saved.runs))saved=null;
  const requested=url.searchParams.get('dataset');
  if((requested&&requested!=='primary')||evidenceDataset(null,saved?.runs)!=='primary')throw Error('This link selects archived research. The public demo now shows only the final primary study. Open the default comparison; historical files remain in the repository archive.');
  data=await loadPrimaryDataset();
  runById=new Map(data.runs.map(run=>[run.run_id,run]));
  // Population facts live in the study ledger (layers-insight.js); #dataset-summary only carries load errors.
  if($('dataset-summary'))$('dataset-summary').textContent='';
  if($('roster-limitations'))$('roster-limitations').textContent=data.limitations;
  for(const p of [...data.personas].sort((a,b)=>Number(a.id)-Number(b.id)))option($('person'),p.id,`Persona ${p.id}`);
  if($('color')){$('color').replaceChildren();option($('color'),'layer','Run identity');for(const key of Object.keys(data.personas[0].attributes))option($('color'),key,key);}
  if(!saved&&$('research-question'))$('research-question').value='method';
  if($('metric'))for(const [key,value] of Object.entries(topologyMetrics))option($('metric'),key,value);
  const initial=initialState(saved,url.searchParams);
  startingRuns=[...initial.layers];
  const atlasButton=$('atlas-mode');if(atlasButton){atlasButton.hidden=false;atlasButton.textContent='Atlas';atlasButton.title='Every primary attempt slot: 112 condition towers of 8 seed planes. Failed attempts are hollow frames.';}
  consensusLive=consensusOf([{run_id:'a',edges:[['0','1']]},{run_id:'b',edges:[['0','1']]}]).length===1;
  setupStage();buildHud();wireControls();renderCoverage();
  app.setAnnouncer(announce);
  app.subscribe((state,changed,source)=>{try{render(changed,source);}catch(error){console.error(error);announce(error.message);}});
  app.init(data,initial);
  if(loadNotice)announce(loadNotice);
  mountShell(app);
  mountFilters(app,{chips:$('active-filters'),root:$('filter-root'),results:$('run-results'),writeUrl:true});
  mountInsight(app,{root:$('insight-root'),ledger:$('study-ledger'),legend:$('legend')});
}
start().catch(error=>{document.body.dataset.state='error';const message=`Saved evidence could not be loaded: ${error.message}`;if($('status'))$('status').textContent=message;if($('dataset-summary'))$('dataset-summary').textContent=message;if($('evidence-notice'))$('evidence-notice').textContent='No substitute data has been loaded. Use Retry to request the saved files again.';for(const control of document.querySelectorAll('main button,main select,main input'))control.disabled=true;if($('retry-load')){$('retry-load').hidden=false;$('retry-load').disabled=false;$('retry-load').onclick=()=>location.reload();}if($('reset-view')){$('reset-view').hidden=false;$('reset-view').href='./layers.html?dataset='+encodeURIComponent(dataset);}}).finally(()=>{document.body.dataset.ready='true';});
