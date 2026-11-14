import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {comparisonLayout,neighbors,tapTracker} from './graph-data.js';
import {configureNavigation} from './navigation.js';

// One on-demand canvas for all 896 graphs, not 896 WebGL contexts.
export function createNetworkStage(host,personas,onSelect){
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x0b1216);
  const canvas=renderer.domElement;canvas.tabIndex=0;canvas.setAttribute('aria-label','Final saved network. Drag to rotate; Shift-drag to pan; scroll to zoom. Arrow keys pan. Use persona and camera controls below for keyboard access.');
  host.replaceChildren(canvas);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,.1,3000),controls=new OrbitControls(camera,canvas);
  configureNavigation(controls,true,'orbit');controls.enableDamping=false;controls.listenToKeyEvents(canvas);
  let group,run,positions,nodes=[],selected='',color=0x6fd6ba;
  const render=()=>{if(!host.clientWidth||!host.clientHeight)return;renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();renderer.render(scene,camera);canvas.dataset.camera=camera.position.toArray().join(',');};
  controls.addEventListener('change',render);
  const observer=new ResizeObserver(render);observer.observe(host);
  function clear(){
    if(!group)return;
    const geometries=new Set(),materials=new Set();
    group.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)materials.add(o.material);});
    for(const g of geometries)g.dispose();for(const m of materials)m.dispose();scene.remove(group);
  }
  function select(id){
    selected=id;const peers=neighbors(run.edges,id);
    for(const mesh of nodes){mesh.material.color.setHex(mesh.userData.id===id?0xffffff:peers.has(mesh.userData.id)?0xf0b56c:color);mesh.material.opacity=!id||mesh.userData.id===id||peers.has(mesh.userData.id)?1:.18;}
    onSelect(id,id?[...peers].sort((a,b)=>Number(a)-Number(b)):[]);render();
  }
  function reset(){camera.position.set(130,100,155);controls.target.set(0,0,0);controls.update();render();}
  function show(graph,method){
    clear();run=graph;selected='';color={global:0x6fd6ba,local:0xf0b56c,sequential:0x8fb6ff,iterative:0xe995c4}[method];
    positions=comparisonLayout(personas,graph.edges,[]);group=new THREE.Group();nodes=[];
    const points=graph.edges.flatMap(edge=>edge.flatMap(id=>positions.get(id)));
    const lines=new THREE.BufferGeometry();lines.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
    group.add(new THREE.LineSegments(lines,new THREE.LineBasicMaterial({color,transparent:true,opacity:.32})));
    const geometry=new THREE.SphereGeometry(2.2,12,8);
    for(const p of personas){const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color,transparent:true}));mesh.position.fromArray(positions.get(p.id));mesh.userData.id=p.id;group.add(mesh);nodes.push(mesh);}
    scene.add(group);canvas.dataset.run=graph.run_id;canvas.dataset.nodes=String(nodes.length);canvas.dataset.edges=String(graph.edges.length);
    reset();select('');
  }
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2(),tap=tapTracker();
  canvas.addEventListener('pointerdown',e=>{canvas.focus({preventScroll:true});tap.down(e);});
  canvas.addEventListener('pointermove',e=>tap.move(e));canvas.addEventListener('pointercancel',()=>tap.cancel());
  canvas.addEventListener('pointerup',e=>{if(!run||!tap.up(e))return;const rect=canvas.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);const id=ray.intersectObjects(nodes)[0]?.object.userData.id||'';select(id===selected?'':id);});
  return {show,select,reset,zoom(factor){camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target);controls.update();render();},
    dispose(){observer.disconnect();controls.dispose();clear();renderer.dispose();canvas.remove();}};
}
