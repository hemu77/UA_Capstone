import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {comparisonLayout,neighbors,tapTracker} from './graph-data.js?v=8fa1a1a66ea0';
import {configureNavigation} from './navigation.js?v=8fa1a1a66ea0';

// Single-graph renderer for the supplement panel only (SPEC.md C.7). Primary graphs are drawn by the layered 3D
// engine; this canvas never shows them. One on-demand WebGL context, three draw calls (instanced lit personas,
// one tie buffer, one ground disc), rendered only when something changes. Positions are an algorithmic layout of
// this graph, not geography or social distance.
const BACKGROUND=0x0b1216,BASE=new THREE.Color(0xc9d3d6),SELECTED=new THREE.Color(0xffffff),NEIGHBOUR=new THREE.Color(0xf0b56c);
const reducedMotion=()=>typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
export function createNetworkStage(host,personas,onSelect){
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(BACKGROUND);
  const canvas=renderer.domElement;canvas.tabIndex=0;
  canvas.setAttribute('aria-label','Supplemental graph in 3D. Drag to rotate, Shift-drag to pan, scroll to zoom; arrow keys pan. The buttons below offer the same camera actions.');
  host.replaceChildren(canvas);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,1,3000),controls=new OrbitControls(camera,canvas);
  scene.fog=new THREE.FogExp2(BACKGROUND,.0021);
  scene.add(new THREE.HemisphereLight(0xdfeef2,0x141c20,1.5));
  const key=new THREE.DirectionalLight(0xffffff,1.9);key.position.set(80,160,110);scene.add(key);
  configureNavigation(controls,true,'orbit');controls.enableDamping=!reducedMotion();controls.dampingFactor=.12;controls.listenToKeyEvents(canvas);
  const ground=new THREE.Mesh(new THREE.CircleGeometry(92,64),new THREE.MeshStandardMaterial({color:0x162127,roughness:.95,metalness:0,transparent:true,opacity:.75}));
  ground.rotation.x=-Math.PI/2;ground.position.y=-34;scene.add(ground);
  const sphere=new THREE.SphereGeometry(1,20,14),material=new THREE.MeshStandardMaterial({roughness:.42,metalness:.08});
  const nodes=new THREE.InstancedMesh(sphere,material,personas.length);nodes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);scene.add(nodes);
  const tieGeometry=new THREE.BufferGeometry(),tieMaterial=new THREE.LineBasicMaterial({color:0xf2c078,transparent:true,opacity:.42,depthWrite:false});
  const ties=new THREE.LineSegments(tieGeometry,tieMaterial);scene.add(ties);
  const ids=personas.map(p=>String(p.id)),matrix=new THREE.Matrix4(),color=new THREE.Color();
  let graph=null,positions=new Map(),selected='',frame=0;
  const render=()=>{frame=0;if(!host.clientWidth||!host.clientHeight)return;renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();
    if(controls.update()&&controls.enableDamping)request();renderer.render(scene,camera);canvas.dataset.camera=camera.position.toArray().map(v=>v.toFixed(2)).join(',');};
  const request=()=>{if(!frame)frame=requestAnimationFrame(render);};
  controls.addEventListener('change',request);
  const observer=new ResizeObserver(request);observer.observe(host);
  function paint(){
    const peers=selected&&graph?neighbors(graph.edges,selected):new Set();
    ids.forEach((id,i)=>{color.copy(id===selected?SELECTED:peers.has(id)?NEIGHBOUR:BASE);if(selected&&id!==selected&&!peers.has(id))color.multiplyScalar(.32);nodes.setColorAt(i,color);});
    if(nodes.instanceColor)nodes.instanceColor.needsUpdate=true;
    tieMaterial.opacity=selected?.16:.42;request();
  }
  function select(id){
    selected=graph&&ids.includes(String(id))?String(id):'';paint();
    const peers=selected?[...neighbors(graph.edges,selected)].sort((a,b)=>Number(a)-Number(b)):[];
    canvas.dataset.selected=selected;onSelect?.(selected,peers);
  }
  function reset(){camera.position.set(0,105,190);controls.target.set(0,0,0);controls.update();request();}
  function show(next,{tieColor=0xf2c078}={}){
    graph=next;selected='';
    positions=comparisonLayout(personas,graph.edges,[]);
    const degree=new Map(ids.map(id=>[id,0]));for(const [a,b] of graph.edges){degree.set(a,degree.get(a)+1);degree.set(b,degree.get(b)+1);}
    const maxDegree=Math.max(1,...degree.values());
    ids.forEach((id,i)=>{const [x,y,z]=positions.get(id),radius=1.6+2.2*Math.sqrt(degree.get(id)/maxDegree);matrix.makeScale(radius,radius,radius).setPosition(x,z,y);nodes.setMatrixAt(i,matrix);});
    nodes.instanceMatrix.needsUpdate=true;nodes.computeBoundingSphere();
    const points=new Float32Array(graph.edges.length*6);
    graph.edges.forEach(([a,b],i)=>{const p=positions.get(a),q=positions.get(b);points.set([p[0],p[2],p[1],q[0],q[2],q[1]],i*6);});
    tieGeometry.setAttribute('position',new THREE.BufferAttribute(points,3));tieGeometry.computeBoundingSphere();
    tieMaterial.color.setHex(tieColor);
    canvas.dataset.run=graph.run_id;canvas.dataset.nodes=String(ids.length);canvas.dataset.edges=String(graph.edges.length);
    reset();paint();onSelect?.('',[]);
  }
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2(),tap=tapTracker();
  canvas.addEventListener('pointerdown',e=>{canvas.focus({preventScroll:true});tap.down(e);});
  canvas.addEventListener('pointermove',e=>tap.move(e));canvas.addEventListener('pointercancel',()=>tap.cancel());
  canvas.addEventListener('pointerup',e=>{if(!graph||!tap.up(e))return;const rect=canvas.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);
    const hit=ray.intersectObject(nodes)[0],id=hit?ids[hit.instanceId]:'';select(id===selected?'':id);});
  return {show,select,reset,
    zoom(factor){camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target);controls.update();request();},
    dispose(){if(frame)cancelAnimationFrame(frame);observer.disconnect();controls.dispose();sphere.dispose();material.dispose();tieGeometry.dispose();tieMaterial.dispose();ground.geometry.dispose();ground.material.dispose();nodes.dispose();renderer.dispose();canvas.remove();}};
}
