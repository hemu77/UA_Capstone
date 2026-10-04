import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
import {validateData, comparisonLayout, layerComparison, layerCandidates, applyLayerSelection, edgeKey, neighbors, replayFrames, personaSteps, personaView, personaTag, personaComparison} from './graph-data.js';
import {topology as topologyMetrics} from './research-plots.js';
import {comparisonModels,togglePersona,tapTracker,matchedRuns,filterOptions,evidenceDataset,conditionCoverage} from './graph-data.js';
import {trajectorySvg, metricBars, runLabel, layerLabel, countryNames, researchEvidence, homophilyMatrix, frequencySvg, loadPresentation} from './formation-charts.js';
import {configureNavigation,panCamera} from './navigation.js';

const $ = id => document.getElementById(id);
const colors = ['#b7c9c3','#d0baa0','#aebacd','#c7b9c7','#bac4aa','#c8c6be'];
let data, runs=[], layout, renderer, scene, camera, controls, group, meshes=[], scheduled=false;
let removed=[], startingRuns=[], planeLabels=[];
let mode='formation',activeId='',frame=0,trace=null,journey=false,playing=false,timer=null,ledgerKey='';
const traceCache=new Map();
let animation=null,animatedMaterials=[],animateNext=false;
let presentation={},galleryKey='',aggregateKey='';
let fitOnResize=false;
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const filterKeys=['collection','method','culture','language','model','seed'];
const url = new URL(location.href);
let dataset='revised_calibration';
const label = runLabel;
const fmt = v => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : 'NA';
function option(select, value, text) { select.add(new Option(text, value)); }
function saveView() {
  const state={runs:runs.map(r=>r.run_id),filters:readFilters(),mode,active:activeId,frame,journey,person:$('person').value,color:$('color').value,separation:$('separation').value,ties:$('ties').value,opacity:$('opacity').value,incident:$('incident').checked,metric:$('metric').value,'research-question':$('research-question').value,'node-labels':$('node-labels').value};
  url.searchParams.set('dataset',dataset);url.searchParams.set('view',JSON.stringify(state)); history.replaceState(null,'',url);
}
function readFilters() {return Object.fromEntries(filterKeys.map(key=>[key,$(key).value]));}
function announce(message) {$('action-status').textContent=message;}
function activeRun() {return runs.find(run=>run.run_id===activeId)||runs[0];}
function stop() {clearTimeout(timer);timer=null;playing=false;}
function selectPersona(id,toggle=true) {stop();journey=false;const next=toggle?togglePersona($('person').value,id):id;$('person').value=next;$('incident').checked=!!next;draw();}
function loadTrace() {
  const run=activeRun();activeId=run.run_id;
  if(!traceCache.has(activeId))traceCache.set(activeId,run.replay_status==='actual_recorded_graph_changes'?replayFrames(run,data.personas.map(p=>p.id)):null);
  trace=traceCache.get(activeId);frame=Math.min(frame,trace?trace.length-1:0);
}
function activate(run) {stop();activeId=run.run_id;frame=Number.MAX_SAFE_INTEGER;journey=false;loadTrace();mode='formation';draw();pose('orbit');}
function seek(step,preserveScope=false) {stop();if(!preserveScope){journey=false;$('incident').checked=false;}frame=Math.max(0,Math.min(trace?.length-1||0,step));draw();}
function steps() {return journey?personaSteps(activeRun().events,$('person').value):Array.from({length:trace?.length-1||0},(_,i)=>i+1);}
function advance() {
  const next=steps().find(step=>step>frame);
  if(next===undefined){stop();draw();return;}
  frame=next;animateNext=true;draw();
  timer=setTimeout(advance,Number($('speed').value));
}
function play() {
  if(!trace)return;
  if(playing){stop();draw();return;}
  stop();if(mode==='compare'){journey=false;frame=0;$('incident').checked=false;}mode='formation';
  if(!steps().some(step=>step>frame))frame=0;
  playing=true;draw();timer=setTimeout(advance,Number($('speed').value));
}
function renderPlayback() {
  const run=activeRun(),selected=$('person').value,available=!!trace&&mode==='formation';
  const current=trace?.[frame],event=current?.event,edges=current?.edges||run.edges;
  $('formation-mode').setAttribute('aria-pressed',String(mode==='formation'&&!journey&&!($('incident').checked&&selected)));
  $('compare-mode').setAttribute('aria-pressed',String(mode==='compare'));
  $('stage-title').textContent=mode==='formation'?label(run):`${runs.length} final networks / shared coordinates`;
  $('stage-state').textContent=mode==='compare'?'Final snapshots':trace?`Event ${frame} / ${trace.length-1}`:'History unavailable';
  $('play').textContent=playing?'Pause':mode==='compare'?'Play active run':journey?'Play persona journey':'Play / resume';
  $('play').disabled=!trace;
  for(const id of ['rewind','previous','next','timeline','show-final','speed'])$(id).disabled=!available;
  $('previous').disabled=!available||frame===0;$('next').disabled=!available||!steps().some(step=>step>frame);
  $('play-persona').disabled=!trace||!selected||!personaSteps(run.events,selected).length;
  $('play-persona').textContent=journey?`Following persona ${selected}`:selected?`Follow persona ${selected}`:'Select a persona to follow';
  $('play-persona').setAttribute('aria-pressed',String(journey));
  $('timeline').max=trace?trace.length-1:0;$('timeline').value=frame;
  $('step-label').textContent=mode==='compare'?'Final networks':trace?`Event ${frame} / ${trace.length-1}`:'No recorded history';
  const relevantSteps=personaSteps(run.events,selected);
  $('playback-scope').textContent=mode==='compare'?'Saved outputs / no shared timeline':journey?`Persona ${selected}: ${relevantSteps.filter(step=>step<=frame).length} / ${relevantSteps.length} relevant events`:$('incident').checked&&selected?`All events / only persona ${selected}'s ties visible`:'All events / whole network';
  document.querySelector('.canvas-legend').innerHTML=mode==='compare'?'<span>Each plane: one final run</span><span>Cross-plane line: same persona, not a tie</span>':'<span class="new-key">New ties</span><span class="removed-key">Removed: dashed</span><span>Outline: actor / larger: selected</span>';
  if(mode==='compare'&&selected){const note=document.createElement('span');note.className='identity-key';note.textContent=`Gold dashed line: persona ${selected} across ${runs.length} runs. Not a friendship.`;document.querySelector('.canvas-legend').replaceChildren(note);}
  $('replay-note').textContent=mode==='compare'?'Each plane is a separate saved experiment, not a time slice or country map. Whole network replays one run.':!trace?'Playback unavailable: only the final graph exists. Its history cannot be reconstructed honestly.':run.method==='global'?'Global generation recorded the entire network in one batch. No per-person order was recorded.':`${run.method==='sequential'?'Sequential generation asks one persona at a time.':run.method==='local'?'Local generation records one persona\'s choices per update.':'Iterative generation records successive additions and removals.'} Whole-network replay shows every recorded update, not simultaneous agents. Several ties can belong to one update: they animate together because their internal order was not recorded.`;
  if(journey&&available)$('replay-note').textContent+=' Persona playback includes their own decisions and other actors changing their ties; unrelated events are skipped on screen but retained in the graph. Manual scrubbing returns to all-event mode.';
  if(run.study==='revised_calibration'&&run.method==='global'&&!run.edges.length)$('replay-note').textContent+=' This run recorded NONE: all 50 personas remain, with zero ties. No connection history exists to animate.';
  $('event-heading').textContent=mode==='compare'?'Final comparison':!trace?'No event log':!event?'Before the first decision':event.persona==null?'Global batch':`Event actor: persona ${event.persona}`;
  const incidentChange=event&&selected&&[...event.added,...event.removed].some(edge=>edge.includes(selected));
  const relevance=!event||!selected?'':event.persona===selected?' The selected persona made this recorded decision.':incidentChange?` ${event.persona==null?'The batch':'Another persona'} changed a tie involving persona ${selected}.`:' This event does not change the selected persona\'s ties.';
  $('event-explanation').textContent=mode==='compare'?'Select Formation to inspect the active run.':!trace?'Only the final edges are available.':!event?'Press Play formation or choose a decision below.':`${event.method}: ${event.added.length} new ties; ${event.removed.length} removed.${!event.added.length&&!event.removed.length?' No net change.':''}${relevance}${event.attempts?` Recorded attempts: ${event.attempts}.`:''} Ties are undirected.`;
  $('event-delta').textContent=mode==='compare'?'Final networks / no shared timeline':!event?trace?'Ready to replay recorded decisions':'No event history':`EVENT ${frame} / +${event.added.length} added / -${event.removed.length} removed${event.persona==null?' / batch':` / actor ${event.persona}`}`;
  $('active-badge').textContent=mode==='compare'?'COMPARISON / FINAL GRAPHS':'REPLAY / ONE SAVED RUN';
  $('event-changes').replaceChildren();
  if(event&&mode==='formation')for(const [kind,list] of [['add',event.added],['remove',event.removed]]){
    const relevant=selected?list.filter(edge=>edge.includes(selected)):list;
    if(relevant.length){const p=document.createElement('p');p.className=`change-${kind}`;p.textContent=`${kind==='add'?'Added':'Removed'}${selected?' involving selected persona':''}: ${relevant.map(edge=>edge.join(' - ')).join(', ')}`;$('event-changes').append(p);}
  }
  if(mode==='formation'){
    const degree=selected?neighbors(edges,selected).size:null;
    $('quick-summary').replaceChildren();
    for(const [name,value] of [['Current ties',edges.length],['Current density',fmt(2*edges.length/(data.personas.length*(data.personas.length-1)))],['Persona neighbors',degree??'--'],['Recorded event',trace?frame:'--']]){const box=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=name;box.append(strong,span);$('quick-summary').append(box);}
    $('status').textContent=`${data.personas.length} roster members / ${edges.length} current ties${selected?` / persona ${selected}: ${degree} neighbors`:''}`;
  }
  $('trajectory-title').textContent=`${run.model}: ${selected?`persona ${selected} connections`:'network growth'}`;
  if(!renderer)$('status').textContent='WebGL unavailable. Playback counts, the ledger and tables remain available.';
  $('trajectory').innerHTML=trajectorySvg(trace,selected,mode==='compare'&&trace?trace.length-1:frame);
  const key=`${activeId}|${selected}`;
  if(ledgerKey!==key){ledgerKey=key;$('events').replaceChildren();const indices=selected?personaSteps(run.events,selected):Array.from({length:run.events?.length||0},(_,i)=>i+1);
    $('ledger-count').textContent=`${indices.length} ${selected?'relevant':'recorded'} events`;
    if(!indices.length){const p=document.createElement('p');p.className='empty-note';p.textContent=trace?'No recorded decisions involve this persona.':'No event log was saved for this run.';$('events').append(p);}
    for(const index of indices){const e=run.events[index-1],button=document.createElement('button');button.className='event-row';button.dataset.step=index;button.setAttribute('aria-label',`Seek event ${index}`);const n=document.createElement('span'),who=document.createElement('span'),delta=document.createElement('span');n.textContent=String(index).padStart(3,'0');who.textContent=e.persona==null?'Global batch':`Persona ${e.persona} / ${e.method}`;const added=selected?e.added.filter(edge=>edge.includes(selected)).length:e.added.length,removed=selected?e.removed.filter(edge=>edge.includes(selected)).length:e.removed.length;delta.textContent=`${selected?'Person':'Run'} +${added} / -${removed}`;button.title=`Whole recorded batch: ${e.added.length} added, ${e.removed.length} removed. No within-batch ordering exists.`;button.append(n,who,delta);button.onclick=()=>{mode='formation';seek(index);};$('events').append(button);}
  }
  for(const button of $('events').querySelectorAll('button')){button.setAttribute('aria-current',String(Number(button.dataset.step)===(mode==='compare'&&trace?trace.length-1:frame)));button.setAttribute('aria-label',`Seek event ${button.dataset.step}: ${[...button.children].slice(1).map(child=>child.textContent).join('; ')}`);}
}
function updateAdd() {
  const run=data.runs.find(r=>r.run_id===$('candidate').value);
  $('candidate-detail').textContent=run?`${label(run)} | ${run.study}`:'No matching saved run. Broaden or reset filters.';
  $('add').disabled=!run||runs.length>=6||runs.some(r=>r.run_id===run.run_id)||run.roster_id!==runs[0].roster_id;
  $('add').textContent=runs.length>=6?'6-layer limit reached':run&&runs.some(r=>r.run_id===run.run_id)?'Already displayed':'Add selected layer';
}
function candidates() {
  const previous=$('candidate').value, found=layerCandidates(data.runs,readFilters());
  $('candidate').replaceChildren();
  for (const r of found) option($('candidate'),r.run_id,`${runs.some(active=>active.run_id===r.run_id)?'[Displayed] ':''}${label(r)} / ${r.study}`);
  if (!found.length) option($('candidate'),'','No saved run matches');
  const available=found.filter(r=>!runs.some(active=>active.run_id===r.run_id));
  $('candidate').value=(available.find(r=>r.run_id===previous)||available[0]||found.find(r=>r.run_id===previous)||found[0])?.run_id||'';
  const same=found.length===runs.length&&found.every(r=>runs.some(active=>active.run_id===r.run_id));
  $('filter-preview').textContent=`${found.length} saved runs match. ${same?'These are the displayed layers.':'Displayed layers stay unchanged until you apply or add.'}${found.length>6?' Narrow filters to 6 or fewer, or add runs individually.':''}`;
  $('apply').disabled=!found.length||found.length>6;
  $('apply').textContent=`Apply ${found.length} matching ${found.length===1?'run':'runs'}`;
  // Keep impossible choices visible but disabled. Never silently substitute a model/language.
  for(const key of filterKeys.filter(key=>key!=='collection')){
    const valid=filterOptions(data.runs,readFilters(),key);
    for(const item of $(key).options)item.disabled=!!item.value&&!valid.has(item.value);
  }
  updateAdd();
}
function requestFrame() {if(!scheduled&&renderer){scheduled=true;requestAnimationFrame(now=>{
  scheduled=false;
  if(animation){
    const progress=Math.min(1,(now-animation.start)/animation.duration),blend=progress*progress*(3-2*progress);
    for(const {material,from,to} of animatedMaterials)material.opacity=from+(to-from)*blend;
    if(progress===1)animation=null;
  }
  controls.update();
  // Keep dots legible in screen space, even when the camera moves away.
  const height=Math.max(1,$('viewport').clientHeight);
  for(const node of meshes){const distance=camera.position.distanceTo(node.position);node.scale.setScalar(Math.max(1,distance*2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))/height*3.2/2.2));}
  renderer.render(scene,camera);
  const occupied=[];
  for(const {element,position,priority=0} of [...planeLabels].sort((a,b)=>(b.priority||0)-(a.priority||0))){
    const p=position.clone().project(camera),x=(p.x+1)*$('viewport').clientWidth/2,y=(1-p.y)*height/2;
    element.hidden=p.z>1||p.z< -1||x<0||x>$('viewport').clientWidth||y<0||y>height;
    element.style.left=`${(p.x+1)*50}%`;element.style.top=`${(1-p.y)*50}%`;
    if(element.hidden)continue;
    const box={x:x+6,y:y-element.offsetHeight/2,w:element.offsetWidth+4,h:element.offsetHeight+4};
    if($('node-labels').value==='auto'&&priority<3&&occupied.some(r=>box.x<r.x+r.w&&box.x+box.w>r.x&&box.y<r.y+r.h&&box.y+box.h>r.y))element.hidden=true;
    else occupied.push(box);
  }
  if(controls.autoRotate||animation)requestFrame();
});}}

function tieLines(points,color,opacity,width=1.4,dashed=false) {
  if(!points.length)return;
  const material=new LineMaterial({color,linewidth:width,transparent:true,opacity,depthWrite:false,dashed,dashSize:2,gapSize:2});
  const line=new LineSegments2(new LineSegmentsGeometry().setPositions(points.flatMap(p=>p.toArray())),material);
  line.computeLineDistances();group.add(line);
  return material;
}
function setupScene() {
  try {renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});}
  catch { $('status').textContent='WebGL unavailable. Measurements and condition plots remain usable.';return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('viewport').append(renderer.domElement);
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(42,1,.1,10000);
  controls=new OrbitControls(camera,renderer.domElement);controls.autoRotateSpeed=.35;
  renderer.domElement.tabIndex=0;
  renderer.domElement.setAttribute('aria-label','Network camera. Drag to rotate, Shift-drag to pan, scroll or pinch to zoom toward pointer. Arrow keys pan; Shift-arrow rotates.');
  controls.listenToKeyEvents(renderer.domElement);
  const interaction=$('graph-interaction');
  const configureInteraction=()=>{configureNavigation(controls,interaction.checked,$('drag-mode').value);renderer.domElement.style.setProperty('touch-action',interaction.checked?'none':'pan-y','important');};
  interaction.onchange=configureInteraction;$('drag-mode').onchange=configureInteraction;configureInteraction();
  renderer.domElement.addEventListener('pointerdown',()=>{renderer.domElement.focus({preventScroll:true});controls.autoRotate=false;$('rotate').setAttribute('aria-pressed','false');},true);
  renderer.domElement.addEventListener('wheel',event=>{
    if(!controls.enabled||!event.shiftKey)return;
    event.preventDefault();event.stopImmediatePropagation();tap.cancel();
    const unit=event.deltaMode===1?16:event.deltaMode===2?renderer.domElement.clientHeight:1;
    panCamera(camera,controls,event.deltaX*unit,event.deltaY*unit,renderer.domElement.clientHeight);requestFrame();
  },{capture:true,passive:false});
  controls.addEventListener('change',requestFrame);
  new ResizeObserver(()=>{const {width,height}=$('viewport').getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();if(fitOnResize){fitOnResize=false;pose('orbit');}requestFrame();}).observe($('viewport'));
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2(),tap=tapTracker();
  renderer.domElement.addEventListener('pointermove',e=>tap.move(e));
  renderer.domElement.addEventListener('pointercancel',()=>tap.cancel());
  renderer.domElement.addEventListener('wheel',()=>tap.cancel(),{passive:true});
  renderer.domElement.addEventListener('pointermove',e=>{
    const rect=renderer.domElement.getBoundingClientRect();pointer.set(2*(e.clientX-rect.left)/rect.width-1,1-2*(e.clientY-rect.top)/rect.height);ray.setFromCamera(pointer,camera);
    const hit=ray.intersectObjects(meshes,false)[0],tooltip=$('node-tooltip');tooltip.hidden=!hit;
    if(hit){const person=data.personas.find(p=>p.id===hit.object.userData.person),run=runs.find(r=>r.run_id===hit.object.userData.run);tooltip.textContent=`${run.model} / ${run.method} | Persona ${person.id} | ${Object.entries(person.attributes).map(([k,v])=>`${k}: ${v}`).join(' | ')}`;}
  });
  renderer.domElement.addEventListener('pointerleave',()=>$('node-tooltip').hidden=true);
  renderer.domElement.addEventListener('pointerdown',e=>tap.down(e));
  renderer.domElement.addEventListener('pointerup',e=>{if(!tap.up(e))return;const rect=renderer.domElement.getBoundingClientRect();pointer.set(2*(e.clientX-rect.left)/rect.width-1,1-2*(e.clientY-rect.top)/rect.height);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(meshes,false)[0];selectPersona(hit?hit.object.userData.person:'',!!hit);});
  pose('orbit');
}
function pose(view) {if(!camera)return;const radius=mode==='formation'?145:Math.hypot(156,(runs.length-1)*Number($('separation').value)/2),d=radius/Math.sin(THREE.MathUtils.degToRad(camera.fov/2))/Math.min(1,camera.aspect);camera.position.set(...(view==='top'?[0,1,.0001]:view==='front'?[0,.8,1]:[.55,.95,.75])).normalize().multiplyScalar(d);controls.target.set(0,0,0);controls.update();requestFrame();}
function category(p) {const key=$('color').value,decade=Math.floor(Number(p.attributes.age)/10)*10;return key==='age'?`${decade}-${decade+9}`:String(p.attributes[key]??'Unknown');}
function draw() {
  const focusedRun=document.activeElement?.classList.contains('select-run')?document.activeElement.getAttribute('aria-label'):null;
  const focusedPersona=document.activeElement?.dataset.person,focusedPersonaRun=document.activeElement?.dataset.run;
  animation=null;animatedMaterials=[];
  const comparison=layerComparison(runs), selected=$('person').value, spacing=Number($('separation').value);
  const categories=[...new Set(data.personas.map(category))].sort();
  const categoryColor = i => $('color').value==='age'?`hsl(${175+i*5}, ${32+i*2}%, ${78-i*4}%)`:`hsl(${Math.round(25+i*360/Math.max(1,categories.length))}, 44%, 68%)`;
  $('legend').replaceChildren();
  if($('color').value!=='layer') categories.forEach((c,i)=>{const el=document.createElement('span');el.textContent=c;el.style.setProperty('--swatch',categoryColor(i));$('legend').append(el);});
  else runs.forEach((run,i)=>{const el=document.createElement('span');el.textContent=layerLabel(runs,i);el.style.setProperty('--swatch',colors[i]);$('legend').append(el);});
  $('legend-heading').textContent=`Color: ${$('color').value==='layer'?'run identity':$('color').value}`;
  // Dispose old GPU buffers on every selection change; no accumulating scenes.
  if(group){group.traverse(o=>{o.geometry?.dispose();if(o.material)o.material.dispose();});scene.remove(group);}
  if(scene){group=new THREE.Group();scene.add(group);}meshes=[];
  const point=(id,index)=>{const p=layout.get(id),scale=mode==='formation'?1.8:1.25;return new THREE.Vector3(p[0]*scale,mode==='formation'?0:(index-(runs.length-1)/2)*spacing,p[1]*scale);};
  $('layers').replaceChildren();$('rows').replaceChildren();$('plane-labels').replaceChildren();$('person-degrees').replaceChildren();planeLabels=[];
  runs.forEach((run,index)=>{
    const li=document.createElement('li');li.style.setProperty('--layer',colors[index]);li.classList.toggle('active',run.run_id===activeId);const title=document.createElement('button');title.className='select-run';title.setAttribute('aria-label',`Inspect run ${index+1}`);title.setAttribute('aria-pressed',String(run.run_id===activeId));title.title=label(run);title.textContent=`${index+1}. ${run.model}\n${run.method} / ${countryNames[run.culture]||run.culture}\n${run.language} / seed ${run.seed}\n${run.events?.length?run.events.length+' recorded updates':'Final graph only'}`;title.onclick=()=>activate(run);li.append(title);
    const remove=document.createElement('button');remove.textContent='×';remove.className='remove-run';remove.setAttribute('aria-label',`Remove layer ${index+1}`);remove.disabled=runs.length===1;remove.onclick=()=>{removed.push({run,index});runs.splice(index,1);refresh();candidates();announce(`Removed ${run.model} / ${run.method}. Undo remove restores it even if filters change.`);};li.append(remove);$('layers').append(li);
    const renderedEdges=mode==='formation'&&run.run_id===activeId&&trace?trace[frame].edges:run.edges;
    const currentEvent=mode==='formation'&&run.run_id===activeId?trace?.[frame].event:null;
    const ego=personaView(renderedEdges,selected,$('incident').checked),adjacent=ego.neighbors;
    const removedNeighbors=new Set((currentEvent?.removed||[]).filter(edge=>selected&&edge.includes(selected)).flat());
    if(ego.visibleIds)for(const id of removedNeighbors)ego.visibleIds.add(id);
    if(selected&&(mode==='compare'||run.run_id===activeId)){const item=document.createElement('p');item.textContent=`Run ${index+1} / ${run.model}: ${adjacent.length} ${mode==='compare'?'final':'current'} neighbors${adjacent.length?'':' (isolated)'}`;item.style.borderLeft=`3px solid ${colors[index]}`;for(const id of adjacent.sort((a,b)=>Number(a)-Number(b))){const button=document.createElement('button');button.className='neighbor';button.textContent=id;button.title=`Inspect persona ${id} across selected runs`;button.onclick=()=>selectPersona(id);item.append(button);}$('person-degrees').append(item);}
    const finalAdjacent=selected?[...neighbors(run.edges,selected)].sort():[];
    const row=document.createElement('tr');for(const value of [`${index+1}. ${label(run)}`,data.personas.length,run.edges.length,fmt(run.metrics.density),fmt(run.metrics.avg_clustering_coef),selected?`${finalAdjacent.length}: ${finalAdjacent.join(', ')||'none'}`:'Select a persona']){const td=document.createElement('td');td.textContent=value;row.append(td);}$('rows').append(row);
    if(!group||(mode==='formation'&&run.run_id!==activeId))return;
    if(mode==='compare'){const badge=document.createElement('span');badge.className='plane-label run-label';badge.textContent=layerLabel(runs,index);badge.title=label(run);badge.style.borderColor=colors[index];$('plane-labels').append(badge);planeLabels.push({element:badge,position:point(data.personas[0].id,index).set(-112,point(data.personas[0].id,index).y,-95),priority:4});}
    if(mode==='compare'){
      const plane=new THREE.Mesh(new THREE.PlaneGeometry(220,220),new THREE.MeshBasicMaterial({color:colors[index],transparent:true,opacity:.045,side:THREE.DoubleSide,depthWrite:false}));plane.rotation.x=-Math.PI/2;plane.position.y=point(data.personas[0].id,index).y;group.add(plane);
      const border=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[-110,-110],[110,-110],[110,110],[-110,110]].map(([x,z])=>new THREE.Vector3(x,plane.position.y,z))),new THREE.LineBasicMaterial({color:colors[index],transparent:true,opacity:.3}));group.add(border);
    }
    const vertices=[],newVertices=[],newKeys=new Set((currentEvent?.added||[]).map(edgeKey));
    for(const edge of ego.edges){const count=comparison.counts.get(edgeKey(edge));if(mode==='compare'&&($('ties').value==='shared'&&count!==runs.length||$('ties').value==='different'&&count===runs.length))continue;for(const id of edge)(newKeys.has(edgeKey(edge))?newVertices:vertices).push(point(id,index));}
    tieLines(vertices,selected?'#aad8c5':'#88948f',selected?0.9:Number($('opacity').value)/100,selected?2.3:1.4);
    const additions=tieLines(newVertices,'#80e2c4',1,2.5);
    if(additions)animatedMaterials.push({material:additions,from:.05,to:1});
    for(const edge of currentEvent?.removed||[]){if($('incident').checked&&selected&&!edge.includes(selected))continue;const material=tieLines(edge.map(id=>point(id,index)),'#ff9a79',.65,2,true);if(material)animatedMaterials.push({material,from:1,to:.25});}
    const activeNodes=new Set(renderedEdges.flat());
    for(const person of data.personas){if(ego.visibleIds&&!ego.visibleIds.has(person.id))continue;const actor=currentEvent?.persona===person.id,focus=person.id===selected||actor;const color=$('color').value==='layer'?colors[index]:categoryColor(categories.indexOf(category(person)));const node=new THREE.Mesh(new THREE.SphereGeometry(focus?3.3:2.2,12,10),new THREE.MeshBasicMaterial({color,transparent:true,opacity:focus?1:mode==='formation'&&!activeNodes.has(person.id)?.65:1}));node.position.copy(point(person.id,index));node.userData.person=person.id;node.userData.run=run.run_id;group.add(node);meshes.push(node);
      if(focus){const outline=new THREE.Mesh(new THREE.SphereGeometry(3.9,12,10),new THREE.MeshBasicMaterial({color:actor?'#efbc7e':'#f0f1eb',side:THREE.BackSide}));node.add(outline);}
      if(person.id===selected||$('node-labels').value==='all'||$('node-labels').value==='auto'&&(!selected||focus||adjacent.includes(person.id)||removedNeighbors.has(person.id))){const badge=document.createElement('span');badge.className='plane-label persona-label';badge.dataset.person=person.id;badge.dataset.run=run.run_id;const description=personaTag(person.id,selected,actor?person.id:null,frame);badge.textContent=person.id===selected?`Persona ${person.id} / ${adjacent.length} neighbors`:focus?description:person.id;badge.setAttribute('aria-label',description);badge.style.borderColor=actor?'#ffbd82':'#80e2c4';$('plane-labels').append(badge);planeLabels.push({element:badge,position:point(person.id,index),priority:focus?3:0});}
    }
  });
  if(group&&selected&&runs.length>1&&mode==='compare'){
    const points=[];for(let i=1;i<runs.length;i++)points.push(point(selected,i-1),point(selected,i));
    tieLines(points,'#f0d59b',1,3,true);
  }
  const person=data.personas.find(p=>p.id===selected);
  $('person-detail').textContent=person?`Persona ${selected}: ${Object.entries(person.attributes).map(([k,v])=>`${k}: ${v}`).join('; ')}`:'Choose a node or persona to compare their neighbors.';
  if(renderer)$('status').textContent=`${runs.length} layers · ${data.personas.length} people per layer · ${comparison.shared} ties shared across all layers`;
  $('comparison').textContent=`${comparison.shared} shared ties / ${comparison.union} distinct ties across selected final networks.`;
  $('quick-summary').replaceChildren();
  const personComparison=selected?personaComparison(runs,selected):null;
  const stats=personComparison?[['Selected persona',selected],['Distinct neighbors',personComparison.union],['Neighbors in every run',personComparison.shared],['Neighbor overlap',runs.length<2?'Add 2nd run':personComparison.jaccard===null?'No neighbors':`${(100*personComparison.jaccard).toFixed(1)}%`]]:[['Layers',runs.length],['People / layer',data.personas.length],['Shared ties',runs.length>1?comparison.shared:'Add 2nd run'],['Intersection / union',runs.length<2?'Add 2nd run':comparison.union?`${(100*comparison.shared/comparison.union).toFixed(1)}%`:'No ties']];
  for(const [name,value] of stats){const item=document.createElement('div'),number=document.createElement('strong'),caption=document.createElement('span');number.textContent=value;caption.textContent=name;item.append(number,caption);$('quick-summary').append(item);}
  const varies=['model','method','culture','language','seed'].filter(key=>new Set(runs.map(r=>r[key])).size>1);
  $('comparison-scope').textContent=runs.length===1?'Add a second run to compare edge overlap.':`Varies: ${varies.join(', ')||'study or source variant'}. Overlap is descriptive, not a significance test.`;
  if(new Set(runs.map(r=>r.prompt_variant??'historical')).size>1)$('comparison-scope').textContent+=' Source variants also differ; this is not a controlled model-only test.';
  if(runs.some(r=>r.language_confound))$('comparison-scope').textContent+=' Historical language confounding is present.';
  $('layer-count').textContent=`${runs.length}/6`;$('undo').disabled=!removed.length||runs.length>=6;
  $('warning').textContent=runs.some(r=>r.language_confound)?'Historical language confounding is present: instructions and persona language were not cleanly separated.':'';
  if(new Set(runs.map(r=>r.prompt_variant??'historical')).size>1)$('warning').textContent+=' Selected runs use different prompt/source variants; this comparison is descriptive.';
  $('plot').innerHTML=metricBars(runs,$('metric').value,topologyMetrics[$('metric').value]);
  const reference=activeRun(),metric=$('metric').value,baseline=reference.metrics[metric];
  const contrasts=runs.filter(run=>run!==reference).map(run=>{
    const value=run.metrics[metric];return `${layerLabel(runs,runs.indexOf(run))}: ${Number.isFinite(value)&&Number.isFinite(baseline)?`${value-baseline>=0?'+':''}${(value-baseline).toFixed(4)}`:'undefined'}`;
  });
  $('metric-deltas').textContent=contrasts.length?`Absolute difference from active run (${label(reference)}): ${contrasts.join('; ')}. Descriptive differences, not significance or accuracy.`:'Add another run for a metric difference. No comparison is defined for one run.';
  const definitions={density:'Density: observed edges divided by all possible undirected pairs. Higher means denser, not better.',avg_clustering_coef:'Mean clustering: how often a person\'s neighbors are connected to one another. Higher means more closed triangles, not greater accuracy.',prop_nodes_lcc:'Largest component share: fraction of people in the largest connected group.',modularity:'Modularity: community structure relative to the metric\'s null model. It is not a validation score.',global_efficiency:'Global efficiency: mean inverse shortest-path distance, with disconnected pairs contributing zero.',num_components:'Connected components: separate groups with no path between them.'};
  $('metric-definition').textContent=definitions[$('metric').value];
  $('ties').disabled=mode==='formation';$('separation').disabled=mode==='formation'||runs.length<2;$('spacing-value').textContent=$('separation').disabled?'(comparison only)':spacing;
  $('clear-person').disabled=!selected;$('incident').disabled=!selected;
  $('focus-person').disabled=!selected||!camera;
  for(const button of document.querySelectorAll('[data-match]')){
    const matches=matchedRuns(data.runs,activeRun(),button.dataset.match);
    button.disabled=matches.length<2||matches.length>6||(button.dataset.match==='culture'&&activeRun().language!=='english');
  }
  for(const button of $('layers').querySelectorAll('.select-run'))button.setAttribute('aria-label',`Inspect run: ${button.textContent.replace(/\n/g,', ')}`);
  renderPlayback();renderEvidence();saveView();
  if(animateNext&&!reducedMotion.matches&&animatedMaterials.length){animation={start:performance.now(),duration:Math.min(550,Number($('speed').value)*.7)};for(const item of animatedMaterials)item.material.opacity=item.from;}animateNext=false;
  if(focusedRun)[...$('layers').querySelectorAll('.select-run')].find(button=>button.getAttribute('aria-label')===focusedRun)?.focus({preventScroll:true});
  if(focusedPersona)([...$('plane-labels').querySelectorAll('button')].find(button=>button.dataset.person===focusedPersona&&button.dataset.run===focusedPersonaRun)||$('person')).focus({preventScroll:true});
  requestFrame();
}
function renderEvidence() {
  const run=activeRun(),question=$('research-question').value,result=researchEvidence(runs,question);
  const selectionKey=runs.map(r=>r.run_id).join('|'),summaryKey=selectionKey+'|'+$('person').value;
  if(summaryKey!==aggregateKey){aggregateKey=summaryKey;$('aggregate-graph').innerHTML=frequencySvg(data.personas,runs,layout,$('person').value);$('aggregate-note').textContent=`Based on ${runs.length} selected final graphs${$('person').value?`, showing persona ${$('person').value}'s direct ties`:''}. It does not follow the playback cursor. Inputs: ${runs.map(label).join('; ')}.`;}
  if(selectionKey!==galleryKey){galleryKey=selectionKey;$('source-gallery').replaceChildren();for(const item of runs){
    const figure=document.createElement('figure'),caption=document.createElement('figcaption'),record=presentation[item.run_id];caption.textContent=label(item);
    if(record&&record.source_sha256===item.source_sha256&&record.source===item.source&&record.png_url===`./data/presentation/${item.run_id}.png`&&/^\.\/data\/presentation\/[a-zA-Z0-9_.-]+\.png$/.test(record.png_url)){
      const image=document.createElement('img');image.src=record.png_url;image.alt=`Presentation rendering of ${label(item)}. ${item.edges.length} saved ties; no new generation.`;image.loading='lazy';figure.append(image);
      const link=document.createElement('a');link.href=record.png_url;link.target='_blank';link.rel='noopener';link.textContent='PNG panel (400 dpi)';caption.append(document.createElement('br'),link);
      if(record.svg_url===`./data/presentation/${item.run_id}.svg`){const vector=document.createElement('a');vector.href=record.svg_url;vector.target='_blank';vector.rel='noopener';vector.textContent='Vector SVG';caption.append(vector);}
      if(record.caption){const details=document.createElement('details'),summary=document.createElement('summary'),text=document.createElement('p');summary.textContent='Suggested manuscript caption';text.textContent=record.caption;details.append(summary,text);caption.append(details);}
    }else if(['calibration','revised_calibration'].includes(item.study)&&item.png_url===`./data/${item.study==='calibration'?'calibration':'revised-calibration'}/${item.run_id}.png`){
      const link=document.createElement('a');link.href=item.png_url;link.textContent='Original calibration PNG';link.target='_blank';link.rel='noopener';caption.append(document.createElement('br'),link);
      const adjacency=document.createElement('a');adjacency.href=item.adjacency_url;adjacency.textContent='Saved adjacency list';adjacency.download='';caption.append(adjacency);
      const note=document.createElement('p');note.textContent='Original artifacts, not a publication panel. Their hashes were checked during export.';caption.append(note);
    }else caption.append(document.createTextNode(' / Presentation figure not available; no placeholder graph.'));
    figure.append(caption);$('source-gallery').append(figure);
  }}
  const descriptions={rq1:'RQ1: Does changing country framing alter topology and homophily with English instructions held constant? Match model, method, seed and roster. This tests prompt framing, not entire national cultures.',rq2:'RQ2: Which recorded attributes show within-group mixing? Compare categorical Coleman homophily within each run. Age uses a separate assortativity measure. These descriptive scores cannot establish what caused a tie.',rq3:'RQ3: How similar are different models under otherwise matched conditions? Compare final topology and exact shared edges. Code/prompt changes prevent clean model-only attribution.',rq4:'RQ4: Does instruction language change the graph at a fixed country framing? Match model, method, seed and roster. Translation equivalence still needs independent review.'};
  $('rq-explanation').textContent=descriptions[question];
  $('rq-status').textContent=result.matched?'Selection matches the recorded comparison fields. Still exploratory, not a significance test.':'Comparison is not controlled for this research question.';
  $('rq-status').className=result.matched?'selection-matched':'selection-unmatched';
  $('rq-issues').replaceChildren();
  for(const issue of result.issues){const li=document.createElement('li');li.textContent=issue;$('rq-issues').append(li);}
  if(question==='rq1'&&runs.every(r=>r.study==='engineering_pilot')){const li=document.createElement('li');li.textContent='Pilot country coverage: United States only. No new pilot evidence for a country-framing effect.';$('rq-issues').append(li);}
  if(dataset!=='legacy'){
    const note=document.createElement('li');note.textContent=data.calibration_note||'Historical V5 calibration: two repetitions for GPT-6-Luna and one US-English repetition for each other model. The earlier eight-repetition plan is not an approved main study.';$('rq-issues').append(note);
    if(question==='rq3'){const config=document.createElement('li');config.textContent='Model configurations differ in supported decoding settings. Treat this as configuration comparison, not a causal model-architecture test.';$('rq-issues').append(config);}
  }
  $('rq-homophily').hidden=question!=='rq2';$('rq-homophily').innerHTML=question==='rq2'?homophilyMatrix(runs):'';
  $('rq-pairs').replaceChildren();
  if(question==='rq3')for(const pair of result.pairs){const p=document.createElement('p');p.textContent=`Runs ${pair.a} and ${pair.b}: ${pair.shared} shared / ${pair.union} distinct ties. Edge Jaccard ${pair.jaccard===null?'NA':(100*pair.jaccard).toFixed(1)+'%'} (100% means identical edges; not accuracy).`;$('rq-pairs').append(p);}
  $('provenance-summary').textContent=dataset!=='legacy'?data.verification:run.study==='engineering_pilot'?'Saved engineering output from synthetic personas. Receipt hashes, adjacency, final replay and recomputed metrics are checked offline. This does not establish human validity or independently reconcile every API response.':'Historical model output. Conditions inferred from filenames; original API logs, snapshot and event history unavailable.';
  $('provenance-fields').replaceChildren();
  const fields=[['Experiment',label(run)],['Collection',dataset!=='legacy'?`${dataset==='revised_calibration'?'Revised V6':'Historical V5'} calibration (not main study)`:run.study==='engineering_pilot'?'Engineering pilot (not confirmatory)':'Historical archive'],['Graph source',run.source],['Replay',trace?'Recorded deltas reconstruct the final saved graph':'Unavailable; final graph only'],['PNG',run.png_provenance||'Historical PNG-to-graph correspondence unknown; no verified image is offered here.'],['Analysis',run.analysis_version||data.analysis_version]];
  for(const [key,value] of fields){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;$('provenance-fields').append(dt,dd);}
  const png=$('source-png');png.hidden=!run.png_url;
  if(run.png_url&&/^\.\/data\/(artifacts|calibration|revised-calibration)\/[a-zA-Z0-9_.-]+\.png$/.test(run.png_url))png.href=run.png_url;else{png.hidden=true;png.removeAttribute('href');}
  $('provenance-hashes').textContent=JSON.stringify({run_id:run.run_id,roster_sha256:run.roster_id,adjacency_sha256:run.source_sha256,png_sha256:run.png_sha256||null,receipt_sha256:run.receipt_sha256||null,protocol_sha256:run.protocol_sha256||null,generation_source_variant:run.prompt_variant||'unrecorded'},null,2);
}
function refresh() {stop();$('match-note').textContent='';layerComparison(runs);if(!runs.some(run=>run.run_id===activeId)){activeId=runs[0].run_id;frame=Number.MAX_SAFE_INTEGER;journey=false;}loadTrace();layout=comparisonLayout(data.personas,runs.flatMap(r=>r.edges),[]);draw();}
function renderCoverage() {
  $('coverage-panel').hidden=dataset==='legacy';
  if(dataset==='legacy')return;
  $('coverage-summary').textContent=`${data.runs.length} / ${data.planned_networks} saved`;
  const table=document.createElement('table'),caption=document.createElement('caption'),head=document.createElement('thead'),body=document.createElement('tbody'),header=document.createElement('tr');
  caption.textContent=dataset==='revised_calibration'?'Revised calibration: saved / allocated repetitions. Not in scope means no calibration was allocated; it is not an empty graph.':'Historical V5 coverage against the former eight-repetition plan, not a current collection authorization.';
  for(const text of ['Model / method',...data.settings.map(([country,language])=>`${countryNames[country]} / ${language}`)]){const th=document.createElement('th');th.scope='col';th.textContent=text;header.append(th);}head.append(header);
  for(const model of data.models)for(const method of data.methods){
    const row=document.createElement('tr'),name=document.createElement('th');name.scope='row';name.textContent=`${model} / ${method}`;row.append(name);
    for(const [culture,language] of data.settings){
      const {saved:selected,planned}=conditionCoverage(data,model,method,culture,language),td=document.createElement('td'),button=document.createElement('button');
      button.textContent=planned?`${selected.length} / ${planned}`:'Not in scope';button.disabled=!selected.length;
      button.setAttribute('aria-label',`${model}, ${method}, ${countryNames[culture]}, ${language}: ${selected.length} saved of ${planned} allocated`);
      button.title=selected.length?'Inspect every saved repetition in this condition':'Not collected; no generated graph exists';
      button.onclick=()=>{try{runs=applyLayerSelection(selected);removed=[];mode='compare';for(const [key,value] of Object.entries({collection:dataset,model,method,culture,language,seed:''}))$(key).value=value;refresh();candidates();pose('orbit');$('match-note').textContent='Inspecting repetitions of one condition, not a cross-model or cross-country contrast.';announce(`Loaded ${selected.length} recorded repetitions.`);}catch(error){announce(error.message);}};
      td.append(button);row.append(td);
    }body.append(row);
  }table.append(caption,head,body);$('coverage-table').replaceChildren(table);
}
async function start() {
  let saved;try{saved=JSON.parse(url.searchParams.get('view')||'null');}catch{saved=null;}
  if(!saved||typeof saved!=='object'||Array.isArray(saved)||!Array.isArray(saved.runs))saved=null;
  dataset=evidenceDataset(url.searchParams.get('dataset'),saved?.runs);
  const paths={revised_calibration:'revised-calibration',calibration:'calibration',legacy:'networks'};
  const response=await fetch(`./data/${paths[dataset]}.json`,{cache:'no-store'});if(!response.ok)throw new Error(`Data request failed: ${response.status}. Run the offline viewer export; no substitute data is loaded.`);data=validateData(await response.json());
  $('dataset').value=dataset;
  $('dataset').onchange=()=>{stop();const next=new URL(location.href);next.search='';next.searchParams.set('dataset',$('dataset').value);location.assign(next);};
  $('dataset-summary').textContent=dataset!=='legacy'?`${data.runs.length} saved calibration graphs | ${data.personas.length} fictional adults | ${data.models.length} models | ${data.settings.length} settings`:`${data.runs.length} saved historical/pilot graphs | ${data.personas.length} original personas | separate from fresh calibration`;
  $('evidence-notice').textContent=data.evidence_notice||(dataset==='calibration'?'Historical V5 evidence, retained separately. Superseded by revised V6 calibration; do not pool these runs.':'Historical and engineering evidence only. Not the revised calibration.');
  $('roster-limitations').textContent=data.limitations||'These are descriptive model outputs, not observed human ties. This US-structured historical roster includes nine minors. Single-pilot differences are not general population or causal evidence.';
  if(dataset!=='legacy'){$('collection').replaceChildren();option($('collection'),dataset,dataset==='revised_calibration'?'Revised V6 calibration':'Historical V5 calibration');}
  else{const figures=await loadPresentation(fetch);presentation=figures.runs;if(figures.error)announce(figures.error);}
  for(const key of ['method','culture','language','model','seed'])for(const value of [...new Set(data.runs.map(r=>String(r[key])))].sort()){
    const run=data.runs.find(r=>String(r[key])===value);
    option($(key),value,key==='culture'?countryNames[value]||value:key==='seed'&&Number.isInteger(run.repetition)?`Rep ${run.repetition+1} / seed ${value}`:value);
  }
  for(const p of [...data.personas].sort((a,b)=>Number(a.id)-Number(b.id)))option($('person'),p.id,`Persona ${p.id}`);
  $('color').replaceChildren();option($('color'),'layer','Run identity');
  for(const key of Object.keys(data.personas[0].attributes))option($('color'),key,key);
  $('color').value='age';
  for(const [key,value] of Object.entries(topologyMetrics))option($('metric'),key,value);
  if(saved&&Array.isArray(saved.runs)){runs=saved.runs.map(id=>data.runs.find(r=>r.run_id===id)).filter(Boolean);try{layerComparison(runs);}catch{runs=[];}}
  if(!runs.length)runs=(data.models||comparisonModels).map(model=>data.runs.find(r=>r.study===(dataset!=='legacy'?dataset:'engineering_pilot')&&r.model===model&&r.method==='sequential'&&r.culture==='us'&&r.language==='english')).filter(Boolean);
  if(!runs.length)runs=[data.runs[0]];
  startingRuns=[...runs];
  activeId=runs.some(run=>run.run_id===saved?.active)?saved.active:(runs.find(run=>run.model==='gpt-6-luna')||runs[0]).run_id;mode=saved?.mode==='formation'?'formation':'compare';frame=Number.isSafeInteger(saved?.frame)&&saved.frame>=0?saved.frame:Number.MAX_SAFE_INTEGER;
  for(const key of filterKeys){const inferred=key==='collection'?(dataset!=='legacy'?dataset:runs.every(r=>r.study!=='engineering_pilot')?'historical':'engineering_pilot'):(new Set(runs.map(r=>String(r[key]))).size===1?String(runs[0][key]):'');const value=saved?.filters?.[key]??inferred;if([...$(key).options].some(o=>o.value===value))$(key).value=value;}
  for(const id of ['person','color','ties','metric','research-question','node-labels'])if(saved&&[...$(id).options].some(o=>o.value===saved[id]))$(id).value=saved[id];
  for(const id of ['separation','opacity'])if(saved&&Number.isFinite(Number(saved[id])))$(id).value=saved[id];
  $('incident').checked=saved?.incident===true;
  journey=saved?.journey===true&&!!$('person').value;
  for(const id of filterKeys)$(id).onchange=()=>{candidates();saveView();};
  $('candidate').onchange=updateAdd;
  $('filter-form').onsubmit=event=>{event.preventDefault();try{runs=applyLayerSelection(layerCandidates(data.runs,readFilters()));removed=[];refresh();candidates();pose('orbit');announce(`Applied filters: ${runs.length} saved networks now displayed.`);}catch(error){announce(`${error.message} Current layers were kept.`);}};
  $('reset-filters').onclick=()=>{for(const key of filterKeys)$(key).value=key==='collection'?(dataset!=='legacy'?dataset:'engineering_pilot'):'';candidates();saveView();announce('Filters reset. Displayed layers were kept.');};
  $('add').onclick=()=>{const run=data.runs.find(r=>r.run_id===$('candidate').value);if(!run)return;try{runs=applyLayerSelection([...runs,run]);removed=removed.filter(entry=>entry.run.run_id!==run.run_id);refresh();candidates();pose('orbit');announce(`Added ${run.model} / ${run.method}. ${runs.length} layers displayed.`);}catch(error){announce(error.message);}};
  $('undo').onclick=()=>{const last=removed.at(-1);if(!last)return;try{const next=[...runs];next.splice(Math.min(last.index,next.length),0,last.run);runs=applyLayerSelection(next);removed.pop();refresh();candidates();pose('orbit');announce(`Restored ${last.run.model} / ${last.run.method}.`);}catch(error){announce(error.message);}};
  $('restore').onclick=()=>{runs=[...startingRuns];removed=[];refresh();candidates();pose('orbit');announce('Restored the comparison present when you opened this page.');};
  $('person').onchange=()=>selectPersona($('person').value,false);
  $('open-filters').onclick=()=>{const drawer=document.querySelector('.run-drawer');drawer.open=true;drawer.scrollIntoView({block:'start'});$('model').focus({preventScroll:true});};
  $('focus-person').onclick=()=>{
    if(!camera||!$('person').value)return;
    const p=layout.get($('person').value),scale=mode==='formation'?1.8:1.25;
    const target=new THREE.Vector3(p[0]*scale,0,p[1]*scale),offset=camera.position.clone().sub(controls.target);
    controls.target.copy(target);camera.position.copy(target).add(offset);controls.update();requestFrame();
  };
  $('clear-person').onclick=()=>selectPersona('',false);
  for(const id of ['color','ties','opacity','incident','metric','research-question','node-labels'])$(id).oninput=draw;
  $('formation-mode').onclick=()=>{stop();mode='formation';journey=false;frame=0;$('incident').checked=false;$('person').value='';draw();pose('orbit');};
  $('compare-mode').onclick=()=>{stop();mode='compare';draw();pose('orbit');};
  $('play').onclick=()=>play();$('play-persona').onclick=()=>{stop();mode='formation';journey=!journey;$('incident').checked=journey;if(journey)frame=0;draw();pose('orbit');};
  $('rewind').onclick=()=>seek(0,true);$('show-final').onclick=()=>seek(trace?.length-1||0);
  $('previous').onclick=()=>seek([...steps()].reverse().find(step=>step<frame)||0,true);
  $('next').onclick=()=>seek(steps().find(step=>step>frame)??frame,true);
  $('timeline').oninput=()=>seek(Number($('timeline').value));
  $('speed').onchange=()=>{if(playing){clearTimeout(timer);timer=setTimeout(advance,Number($('speed').value));}};
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();if(data)draw();}});
  $('separation').oninput=draw;
  for(const id of ['orbit','top','front'])$(id).onclick=()=>pose(id);
  for(const [id,factor] of [['zoom-in',.8],['zoom-out',1.25]])$(id).onclick=()=>{if(!camera)return;const offset=camera.position.clone().sub(controls.target),distance=Math.min(controls.maxDistance,Math.max(controls.minDistance,offset.length()*factor));camera.position.copy(controls.target).add(offset.setLength(distance));controls.update();requestFrame();};
  $('rotate').onclick=()=>{if(!controls)return;controls.autoRotate=!controls.autoRotate;$('rotate').setAttribute('aria-pressed',String(controls.autoRotate));requestFrame();};
  $('share').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$('share').textContent='Link copied';}catch{$('share').textContent='Copy the URL from your address bar';}};
  $('download-selection').onclick=()=>{
    const value={schema_version:1,dataset,scope:'Selected saved final graphs; not a population estimate or completed main study.',view:JSON.parse(url.searchParams.get('view')),personas:data.personas,runs};
    const objectUrl=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=objectUrl;link.download=`network-selection-${dataset}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(objectUrl),1000);
    announce(`Download requested: ${runs.length} saved networks with personas, events, measurements and source hashes.`);
  };
  $('expand-stage').onclick=()=>{fitOnResize=true;const expanded=document.querySelector('.laboratory').classList.toggle('expanded');$('expand-stage').setAttribute('aria-pressed',String(expanded));$('expand-stage').textContent=expanded?'Collapse':'Expand';};
  for(const button of document.querySelectorAll('[data-match]'))button.onclick=()=>{
    const reference=activeRun(),dimension=button.dataset.match;
    try{runs=applyLayerSelection(matchedRuns(data.runs,reference,dimension));removed=[];mode='compare';
      for(const key of filterKeys)$(key).value=key==='collection'?(dataset!=='legacy'?dataset:reference.study==='engineering_pilot'?'engineering_pilot':'historical'):key===dimension?'':String(reference[key]);
      if(dimension!=='method')$('research-question').value={model:'rq3',culture:'rq1',language:'rq4'}[dimension];
      refresh();candidates();pose('orbit');announce('Matched comparison applied. No missing conditions were filled.');
      $('match-note').textContent=`${runs.length} saved runs matched to ${reference.model} / ${reference.method} / ${reference.culture} / ${reference.language} / seed ${reference.seed}. Only ${dimension} varies among the matched fields.`;
    }catch(error){announce(error.message);}
  };
  renderCoverage();
  if(!saved&&dataset==='legacy')$('collection').value='revised';
  setupScene();refresh();candidates();pose('orbit');
  const missing=comparisonModels.filter(model=>!data.runs.some(r=>r.study==='engineering_pilot'&&r.model===model));
  if(dataset==='legacy'&&missing.length)announce(`Requested comparison: ${comparisonModels.join(', ')}. No saved results for ${missing.join(', ')}; no substitute graph displayed. Older results remain in the archive/all-pilot collection.`);
}
start().catch(error=>{$('status').textContent=`Cannot load layers: ${error.message}`;console.error(error);});
