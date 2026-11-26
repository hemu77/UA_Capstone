// Layers page controller. WORKSTREAM 2 owns this file (docs/qa/viewer-3d-2026-11-25/SPEC.md C.4).
// The shared store (layers-app.js) is the single source of truth for view, layers, active run, replay frame,
// persona, colour, lens, groups, filters and focus. Every control here writes the store through commit();
// one subscriber derives everything on screen from the store, whoever changed it. Only presentation-local
// controls (tie opacity, plane spacing, tie filter, label density, "only this person's ties", persona journey,
// playback timer) live outside the store.
import * as THREE from 'three';
import {comparisonLayout, layerComparison, neighbors, replayFrames, personaSteps, personaView, personaTag, personaComparison, togglePersona, tapTracker, matchedRuns, evidenceDataset, conditionCoverage, MAX_LAYERS, edgeKey} from './graph-data.js?v=8fa1a1a66ea0';
import {topology as topologyMetrics} from './research-plots.js?v=8fa1a1a66ea0';
import {trajectorySvg, metricBars, runLabel, runParts, layerLabel, researchEvidence, homophilyMatrix, frequencySvg, modelName, methodName, settingName, plural} from './formation-charts.js?v=8fa1a1a66ea0';
import {configureNavigation, panCamera} from './navigation.js?v=8fa1a1a66ea0';
import {decodeVerifiedEvents} from './verified-events.js?v=8fa1a1a66ea0';
import {createStore, EMPTY_FILTERS, LAYER_COLORS, MAX_STACK_LAYERS, VIEWS} from './layers-app.js?v=8fa1a1a66ea0';
import {loadPrimaryDataset, loadMetrics, settingKey} from './layers-data-all.js?v=8fa1a1a66ea0';
import {mountFilters, applyFilters, decodeState, slotMask} from './layers-filters.js?v=8fa1a1a66ea0';
import {mountInsight, nodeEncoding, edgeEncoding, consensus as consensusOf} from './layers-insight.js?v=8fa1a1a66ea0';
import {mountShell} from './layers-shell.js?v=8fa1a1a66ea0';
import {createStage3D, STAGE_COLORS} from './layers-3d-scene.js?v=8fa1a1a66ea0';
import {createAtlas} from './layers-3d-atlas.js?v=8fa1a1a66ea0';
import {createStack} from './layers-3d-stack.js?v=8fa1a1a66ea0';
import {createLabels} from './layers-3d-labels.js?v=8fa1a1a66ea0';
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
const FINAL=Number.MAX_SAFE_INTEGER; // "final recorded state"; clamped to the real index once the history is verified
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
let ledgerKey='',aggregateKey='',lastView=null,lastPlaneKey='',entering=null,loadNotice='';
const view3d={lift:false,ribbons:true};
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
const currentFrame=()=>trace?Math.min(S().frame,trace.length-1):0;

// ---------------------------------------------------------------------------------------------
// Actions. Each one writes the store; render() reacts.
function selectPersona(id,toggle=true){
  stop();journey=false;
  const next=toggle?togglePersona(selectedPersona(),id):id;
  if($('incident'))$('incident').checked=!!next;
  if(!commit({persona:next}).size)render(new Set(['local']));
}
function activate(run){stop();journey=false;commit({active:run.run_id,view:'replay',frame:FINAL});}
function seek(step,preserveScope=false){
  stop();if(!preserveScope){journey=false;if($('incident'))$('incident').checked=false;}
  if(!commit({frame:Math.max(0,Math.min(trace?.length-1||0,step))}).size)render(new Set(['local']));
}
function steps(){return journey?personaSteps(activeRun().events,selectedPersona()):Array.from({length:trace?.length-1||0},(_,i)=>i+1);}
function advance(){
  const frame=currentFrame(),next=steps().find(step=>step>frame);
  if(next===undefined){stop();render(new Set(['local']));return;}
  animateNext=true;commit({frame:next});
  timer=setTimeout(advance,Number($('speed')?.value||800));
}
function play(){
  if(!trace)return;
  if(playing){stop();render(new Set(['local']));return;}
  stop();
  const patch={view:'replay'};
  if(!isReplay()){journey=false;patch.frame=0;if($('incident'))$('incident').checked=false;}
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
  labels=createLabels($('plane-labels'));
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
    // On narrow stages the encoding toggles sit in a row above the camera bar and count as bottom chrome too.
    for(const element of document.querySelectorAll('.stage-heading,.stage-tools,.stage-key,.stage-footer'+(view.width<700?',#stage-overlay .g3-tools':''))){
      const r=element.getBoundingClientRect();if(!r.height||r.bottom<view.top||r.top>view.bottom)continue;
      if(r.top-view.top<view.height*.35&&r.bottom-view.top<view.height*.5)top=Math.max(top,r.bottom-view.top);
      else if(view.bottom-r.bottom<view.height*.35)bottom=Math.max(bottom,view.bottom-r.top);
    }
    // Text overlays stay readable over the scene; give them most, not all, of their height.
    stage.setSafeArea({top:top*.8,bottom:bottom*.92});
  };
  const observer=new ResizeObserver(align);observer.observe($('viewport'));observer.observe($('plane-labels'));
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
    // Room on the left for the setting labels (full names on wide canvases, codes on narrow ones); the camera
    // sits slightly off-axis so tower sides and row depth read as volume.
    const b=atlas.layout.bounds,narrow=stage.size.width<700,pad=b.width*(narrow?.2:Math.min(.4,Math.max(.22,270/stage.size.width)));
    stage.setBounds({center:[b.center[0]-pad*.5,b.center[1]*.5,b.center[2]],radius:b.width*.5,extent:[b.width*.5+pad*.5,b.depth*.36+b.height*.5],margin:narrow?1.1:1.2,fov:26,poses:{orbit:narrow?[0,1.15,1]:[.14,.6,1],front:[0,.22,1],top:[0,1,.05]}});
    stage.setEnvironment({fogDensity:.00125,groundY:0,groundRadius:Math.max(b.width,b.depth)*.78,cell:6.5,center:[0,0]});return;
  }
  // Portrait canvases are width-limited: back off so the plane corners stay inside the frame.
  const b=stack.bounds(),plane=isReplay()?150:110,portrait=stage.size.width<stage.size.height*1.35;
  // Replay is one flat plane: fit its projected box (wide, shallow) rather than a sphere, so it fills the stage.
  if(isReplay())stage.setBounds({center:b.center,radius:b.radius,extent:[plane*1.4,plane*.62+(view3d.lift?30:6)],margin:1.04,fov:38,poses:null});
  else stage.setBounds({center:b.center,radius:b.radius,extent:null,margin:portrait?Math.max(1,plane*1.45/b.radius):1,fov:38,poses:null});
  stage.setEnvironment({fogDensity:isReplay()?.0009:.00105,groundY:b.floor,groundRadius:plane*2.7,cell:plane/5.5,center:[0,0],shadow:{x:0,z:0,halfX:plane*.9,halfZ:plane*.9,soft:plane*.35,strength:.5}});
}
function stackLayoutFor(list){
  const key=list.map(r=>r.run_id).join('|');
  if(key!==stackLayoutKey){stackLayoutKey=key;stackLayout=comparisonLayout(data.personas,list.flatMap(r=>r.edges),[]);}
  return stackLayout;
}
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
  const spacing=Number($('separation')?.value||55);
  stack.configure({runs:list,activeId:run.run_id,layout:stackLayoutFor(runs),
    options:{replay,spacing,scale:replay?1.8:1.25,plane:replay?150:110,lift:view3d.lift,opacity:Number($('opacity')?.value||50)/100},
    frame:replay?(frame?{edges:frame.edges,event:frame.event}:null):null,animate:animateNext,duration:Math.min(550,Number($('speed')?.value||800)*.7),
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
  const text={model:modelName,method:methodName,setting:([c,l])=>settingName(c,l),seed:seed=>String(seed),'seed-axis':()=>'Seed ↑'};
  const priority={model:6,setting:5,'seed-axis':5.5,seed:2,method:1},anchor={model:'center',method:'center',setting:'left',seed:'right','seed-axis':'right'};
  // Narrow canvases have no room for full setting names beside the towers: use country / language codes there.
  const compact=stage.size.width<700,codes={us:'US',india:'IN',japan:'JP',brazil:'BR',english:'EN',hindi:'HI',japanese:'JA',portuguese:'PT'};
  if(compact)text.setting=([c,l])=>`${codes[c]||c} / ${codes[l]||l}`;
  return atlas.layout.axes.filter(axis=>!(compact&&(axis.kind==='method'||axis.kind==='seed'))).map((axis,order)=>({id:`axis-${axis.id}`,text:text[axis.kind](axis.key),title:axis.kind==='setting'?settingName(...axis.key):undefined,position:axis.position,priority:priority[axis.kind],anchor:anchor[axis.kind],column:axis.kind==='setting'?'settings':undefined,className:`g3-axis g3-axis-${axis.kind}${compact?' g3-compact':''}`}));
}
function stackLabels(list,frame){
  const s=S(),items=[],replay=s.view==='replay',selected=s.persona,mode=$('node-labels')?.value||'none';
  const event=replay?frame?.event:null;
  if(!replay)list.forEach((run,index)=>{const anchor=stack.planeAnchor(index);if(anchor)items.push({id:`plane-${run.run_id}`,text:layerLabel(runs,index),position:anchor,priority:4,cull:false,clamp:true,anchor:['left','right'],column:stage.size.width<700?'planes':undefined,color:LAYER_COLORS[index],title:label(run),ariaLabel:`Replay layer ${index+1}: ${label(run)}`,className:'plane-label run-label g3-plane',pressed:run.run_id===s.active,onClick:()=>activate(run)});});
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
}
function paintHud(){
  const s=S(),caption=$('g3-caption'),key=$('g3-key');if(!caption){paintTools();return;}
  caption.textContent='';caption.hidden=true;
  key.replaceChildren();
  const swatch=(cls,text)=>{const span=document.createElement('span');span.className=`g3-swatch ${cls}`;span.textContent=text;key.append(span);};
  if(s.view==='atlas'){swatch('g3-valid','Graph');swatch('g3-empty','Valid, no ties');swatch('g3-failed','Failed attempt, no graph');swatch('g3-stacked','In your stack');}
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
  const rows=$('rows'),degrees=$('person-degrees');rows?.replaceChildren();degrees?.replaceChildren();
  runs.forEach((run,index)=>{
    const renderedEdges=replay&&run.run_id===s.active&&trace?trace[currentFrame()].edges:run.edges,adjacent=personaView(renderedEdges,selected,incidentOnly()).neighbors;
    if(degrees&&selected&&(!replay||run.run_id===s.active)){const item=document.createElement('p');item.textContent=`Run ${index+1} / ${modelName(run.model)}: ${adjacent.length} ${replay?'current':'final'} neighbors${adjacent.length?'':' (isolated)'}`;item.style.borderLeft=`3px solid ${LAYER_COLORS[index]}`;for(const id of [...adjacent].sort((a,b)=>Number(a)-Number(b))){const button=document.createElement('button');button.className='neighbor';button.textContent=id;button.title=`Inspect persona ${id} across selected runs`;button.onclick=()=>{selectPersona(id,false);$('person')?.focus({preventScroll:true});};item.append(button);}degrees.append(item);}
    const finalAdjacent=selected?[...neighbors(run.edges,selected)].sort():[];
    if(rows){const row=document.createElement('tr');for(const value of [`${index+1}. ${label(run)}`,data.personas.length,run.edges.length,fmt(run.metrics.density),fmt(run.metrics.avg_clustering_coef),selected?`${finalAdjacent.length}: ${finalAdjacent.join(', ')||'none'}`:'Select a persona']){const td=document.createElement('td');td.textContent=value;row.append(td);}rows.append(row);}
  });
  const person=data.personas.find(p=>p.id===selected);
  if($('person-detail')){if(person)$('person-detail').replaceChildren(personaCard(person));else $('person-detail').textContent='Choose a node or persona to compare their neighbors.';}
  if(stage&&$('status'))$('status').textContent=s.view==='atlas'?'':`${plural(runs.length,'layer')} · ${data.personas.length} people per layer · ${plural(comparison.shared,'tie')} shared across all layers`;
  if($('comparison'))$('comparison').textContent=`${comparison.shared} shared ties / ${comparison.union} distinct ties across selected final networks.`;
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
  if($('plot'))$('plot').innerHTML=measured?metricBars(runs,metric,topologyMetrics[metric]):metricsNote();
  const reference=activeRun(),baseline=reference.metrics[metric];
  const contrasts=runs.filter(run=>run!==reference).map(run=>{const value=run.metrics[metric];return `${layerLabel(runs,runs.indexOf(run))}: ${Number.isFinite(value)&&Number.isFinite(baseline)?`${value-baseline>=0?'+':''}${(value-baseline).toFixed(4)}`:'undefined'}`;});
  if($('metric-deltas'))$('metric-deltas').textContent=!measured?'':contrasts.length?`Signed difference from active run (${label(reference)}): ${contrasts.join('; ')}. Each value is that run minus the active run. Descriptive differences, not significance or accuracy.`:'Add another run for a metric difference. No comparison is defined for one run.';
  const definitions={density:'Density: observed edges divided by all possible undirected pairs. Higher means denser, not better.',avg_clustering_coef:'Mean clustering: how often a person\'s neighbors are connected to one another. Higher means more closed triangles, not greater accuracy.',prop_nodes_lcc:'Largest component share: fraction of people in the largest connected group.',modularity:'Modularity: community structure relative to the metric\'s null model. It is not a validation score.',global_efficiency:'Global efficiency: mean inverse shortest-path distance, with disconnected pairs contributing zero.',num_components:'Connected components: separate groups with no path between them.'};
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
  set('stage-state',atlasView?'Every primary attempt slot':!replay?'Final snapshots':trace?`Event ${frame} / ${trace.length-1}`:pending?'Loading verified replay…':'History unavailable');
  if($('play')){$('play').textContent=playing?'Pause':!replay?'Play active run':journey?'Play persona journey':'Play / resume';$('play').disabled=!trace;}
  for(const id of ['rewind','previous','next','timeline','show-final','speed'])if($(id))$(id).disabled=!available;
  if($('previous'))$('previous').disabled=!available||frame===0;if($('next'))$('next').disabled=!available||!steps().some(step=>step>frame);
  const play=$('play-persona');
  if(play){play.disabled=!trace||!selected||!personaSteps(run.events,selected).length;play.textContent=journey?`Following persona ${selected}`:selected?`Follow persona ${selected}`:'Select a persona to follow';play.setAttribute('aria-pressed',String(journey));}
  if($('timeline')){$('timeline').max=trace?trace.length-1:0;$('timeline').value=replay?frame:trace?trace.length-1:0;}
  set('step-label',!replay?'Final networks':trace?`Event ${frame} / ${trace.length-1}`:'No recorded history');
  const relevantSteps=personaSteps(run.events,selected);
  set('playback-scope',!replay?'Saved outputs / no shared timeline':journey?`Persona ${selected}: ${relevantSteps.filter(step=>step<=frame).length} / ${relevantSteps.length} relevant events`:incidentOnly()&&selected?`All events / only persona ${selected}'s ties visible`:'All events / whole network');
  const legend=document.querySelector('.canvas-legend');
  if(legend){
    legend.innerHTML=atlasView?'<span>Columns: model × method</span><span>Rows: country framing / language</span><span>Tower height: seed order, not time</span>':!replay?'<span>Each plane: one saved run, not a time slice</span>':'<span class="new-key">New ties</span><span class="removed-key">Removed: dashed</span><span>Outline: actor / larger: selected</span>';
  }
  const note=$('replay-note');
  if(note){
    note.textContent=!replay?'':!trace?'Playback unavailable: only the final graph exists. Its history cannot be reconstructed honestly.':run.method==='global'?'Global generation recorded the whole network in one batch; no per-person order exists.':`${run.method==='sequential'?'Sequential: one persona is asked per recorded update.':run.method==='local'?'Local: each recorded update holds one persona\'s choices.':'Iterative: each recorded update adds and removes ties.'} Ties within one update appear together because their internal order was not recorded. Not simultaneous agents.`;
    if(replay&&pending)note.textContent='Loading and verifying this run’s recorded events. Final graph remains visible; playback will become available after the checks pass.';
    if(replay&&failed)note.textContent='Recorded history could not be verified. Playback is disabled; the saved final graph remains available. Reload the page to retry.';
    if(journey&&available)note.textContent+=' Following a persona skips unrelated updates on screen; the graph still includes them.';
    if(replay&&['primary','revised_calibration'].includes(run.study)&&run.method==='global'&&!run.edges.length)note.textContent+=' This valid run has no ties: all 50 personas remain. No connection history exists to animate.';
  }
  set('event-heading',!replay?'Final comparison':pending?'Verifying replay':failed?'Replay unavailable':!trace?'No event log':!event?'Before the first decision':event.persona==null?'Global batch':`Event actor: persona ${event.persona}`);
  const incidentChange=event&&selected&&[...event.added,...event.removed].some(edge=>edge.includes(selected));
  const relevance=!event||!selected?'':event.persona===selected?' The selected persona made this recorded decision.':incidentChange?` ${event.persona==null?'The batch':'Another persona'} changed a tie involving persona ${selected}.`:' This event does not change the selected persona\'s ties.';
  set('event-explanation',!replay?'Select Replay to inspect the active run.':!trace?'Only the final edges are available.':!event?'Press Play or choose a decision below.':`${methodName(event.method)}: ${plural(event.added.length,'new tie')}; ${event.removed.length} removed.${!event.added.length&&!event.removed.length?' No net change.':''}${relevance}${event.attempts?` Recorded attempts: ${event.attempts}.`:''} Ties are undirected.`);
  set('event-delta',atlasView?(matchMedia('(hover: none)').matches?'Tap a tower to stack its seeds':'Hover a tower · click to stack its seeds'):!replay?'':!event?trace?'Ready to replay recorded decisions':'No event history':`EVENT ${frame} / +${event.added.length} added / -${event.removed.length} removed${event.persona==null?' / batch':` / actor ${event.persona}`}`);
  set('active-badge',atlasView?'Atlas · every primary attempt slot':!replay?'Stack · saved final graphs':'Replay · one saved run, generation order');
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
  set('trajectory-title',`${modelName(run.model)}: ${selected?`persona ${selected} connections`:'network growth'}`);
  if($('trajectory'))$('trajectory').innerHTML=trajectorySvg(trace,selected,!replay&&trace?trace.length-1:frame);
  const events=$('events'),key=`${run.run_id}|${selected}|${run.events?.length||0}|${pending}`;
  if(events&&ledgerKey!==key){ledgerKey=key;events.replaceChildren();const indices=selected?personaSteps(run.events,selected):Array.from({length:run.events?.length||0},(_,i)=>i+1);
    set('ledger-count',plural(indices.length,`${selected?'relevant':'recorded'} event`));
    if(!indices.length){const p=document.createElement('p');p.className='empty-note';p.textContent=pending?'Verifying recorded decisions…':failed?'Recorded decisions unavailable; playback is disabled.':trace?'No recorded decisions involve this persona.':'No event log was saved for this run.';events.append(p);}
    for(const index of indices){const e=run.events[index-1],button=document.createElement('button');button.className='event-row';button.dataset.step=index;const n=document.createElement('span'),who=document.createElement('span'),delta=document.createElement('span');n.textContent=String(index).padStart(3,'0');who.textContent=e.persona==null?'Global batch':`Persona ${e.persona} / ${methodName(e.method)}`;const added=selected?e.added.filter(edge=>edge.includes(selected)).length:e.added.length,removedCount=selected?e.removed.filter(edge=>edge.includes(selected)).length:e.removed.length;delta.textContent=`${selected?'Person':'Run'} +${added} / -${removedCount}`;button.title=`Whole recorded batch: ${e.added.length} added, ${e.removed.length} removed. No within-batch ordering exists.`;button.append(n,who,delta);button.onclick=()=>{if(S().view!=='replay')commit({view:'replay'});seek(index);};events.append(button);}
  }
  if(events)for(const button of events.querySelectorAll('button')){button.setAttribute('aria-current',String(Number(button.dataset.step)===(!replay&&trace?trace.length-1:frame)));button.setAttribute('aria-label',`Seek event ${button.dataset.step}: ${[...button.children].slice(1).map(child=>child.textContent).join('; ')}`);}
}
function renderEvidence(){
  const run=activeRun(),question=$('research-question')?.value||'method',result=researchEvidence(runs,question),selected=S().persona;
  const selectionKey=runs.map(r=>r.run_id).join('|'),summaryKey=selectionKey+'|'+selected;
  if(summaryKey!==aggregateKey&&$('aggregate-graph')){aggregateKey=summaryKey;$('aggregate-graph').innerHTML=frequencySvg(data.personas,runs,stackLayoutFor(runs),selected);if($('aggregate-note'))$('aggregate-note').textContent=`Based on ${plural(runs.length,'selected final graph')}${selected?`, showing persona ${selected}'s direct ties`:''}. It does not follow the playback cursor. Inputs: ${runs.map(label).join('; ')}.`;}
  const descriptions={method:'Method comparison: hold model, roster, seed and instructions fixed while the generation procedure changes. This is a descriptive view; the computational budget and interaction sequence differ by procedure.',rq1:'RQ1: Does changing country framing alter topology and homophily with English instructions held constant? Match model, method, seed and roster. This tests prompt framing, not entire national cultures.',rq2:'RQ2: Which recorded attributes show within-group mixing? Compare categorical Coleman homophily within each run. Age uses a separate assortativity measure. These descriptive scores cannot establish what caused a tie.',rq3:'RQ3: How similar are different models under otherwise matched conditions? Compare final topology and exact shared edges. Code/prompt changes prevent clean model-only attribution.',rq4:'RQ4: Does instruction language change the graph at a fixed country framing? Match model, method, seed and roster. Translation equivalence still needs independent review.'};
  if($('rq-explanation'))$('rq-explanation').textContent=descriptions[question];
  if($('rq-status')){$('rq-status').textContent=result.matched?'Selection matches the recorded comparison fields. Still exploratory, not a significance test.':'Comparison is not controlled for this research question.';$('rq-status').className=result.matched?'selection-matched':'selection-unmatched';}
  const issues=$('rq-issues');
  if(issues){issues.replaceChildren();for(const issue of result.issues){const li=document.createElement('li');li.textContent=issue;issues.append(li);}
    if(question==='rq3'){const config=document.createElement('li');config.textContent='Model configurations differ in supported decoding settings. Treat this as configuration comparison, not a causal model-architecture test.';issues.append(config);}}
  if($('rq-homophily')){$('rq-homophily').hidden=question!=='rq2';$('rq-homophily').innerHTML=question!=='rq2'?'':metricsReady()?homophilyMatrix(runs):metricsNote();}
  const pairs=$('rq-pairs');
  if(pairs){pairs.replaceChildren();if(question==='rq3')for(const pair of result.pairs){const p=document.createElement('p');p.textContent=`Runs ${pair.a} and ${pair.b}: ${pair.shared} shared / ${pair.union} distinct ties. Edge Jaccard ${pair.jaccard===null?'NA':(100*pair.jaccard).toFixed(1)+'%'} (100% means identical edges; not accuracy).`;pairs.append(p);}}
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
  caption.textContent='Valid graphs / 8 planned attempts per condition. Failed attempts remain in the repository audit, never rendered as graphs. Select a cell to stack its valid seeds (the same as clicking its tower in the Atlas).';
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
  const state={...base,layers,view:saved.mode==='formation'?'replay':saved.mode==='atlas'?'atlas':'stack',active:layers.includes(saved.active)?saved.active:layers[0],frame:Number.isSafeInteger(saved.frame)&&saved.frame>=0?saved.frame:FINAL};
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
  on('formation-mode','click',()=>{stop();journey=false;if($('incident'))$('incident').checked=false;if(!commit({view:'replay',frame:0,persona:''}).size)render(new Set(['local']));stage?.pose('orbit');});
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
  on('share','click',async()=>{try{await navigator.clipboard.writeText(location.href);$('share').textContent='Link copied';}catch{$('share').textContent='Copy the URL from your address bar';}});
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
