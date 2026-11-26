// WORKSTREAM 2 (3D engine). Stack: 1..8 saved runs as lit glass planes with shared persona positions, and the
// single-plane Replay of one run's recorded generation order (SPEC.md B.1, B.3, B.5, C.4).
// Every GPU buffer is allocated once with capacity for 8 planes + 1 consensus plane and then updated in place:
// changing runs, frame, persona, tie filter, encoding or opacity rewrites attributes or uniforms only.
// Draw calls (independent of the number of planes): slabs 1, node contact shadows 1, nodes 1, halos 1, stems 1,
// ties 1, highlighted ties 1, new ties 1, dashed lines 1, shared-tie ribbons 1.
import * as THREE from 'three';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
import {STAGE_COLORS,raw,createStateTexture,createSlabMaterial,keyLightInView,easeInOutCubic} from './layers-3d-scene.js';

export const STACK_CAPACITY=9; // MAX_STACK_LAYERS planes + one optional consensus plane
const PAIRS=n=>n*(n-1)/2;
// Order-independent tie key on persona ids (ids are numeric strings).
export const keyOf=(a,b)=>Number(a)<Number(b)||(Number(a)===Number(b)&&String(a)<String(b))?`${a}|${b}`:`${b}|${a}`;

// Pure: typed arrays for one stack. runs: [{run_id, edges:[[a,b]]}]; layout: Map(id -> [x, z, ...]) in layout units.
// options: {personas:[id], spacing, scale, plane (half size), lift, liftMax, rMin, rMax, ties:'all'|'shared'|'different',
//           persona, incident, edgesOf(run, index) -> edges}
export function stackGeometry(runs,layout,options={}) {
  const ids=options.personas||[...layout.keys()];
  const N=ids.length,L=runs.length,index=new Map(ids.map((id,i)=>[String(id),i]));
  const {spacing=55,scale=1.25,lift=false,liftMax=Math.min(26,spacing*.38),rMin=1.7,rMax=4.4,ties:tieMode='all',persona='',incident=false}=options;
  const edgesOf=options.edgesOf||(run=>run.edges);
  const planeY=new Float32Array(L);for(let l=0;l<L;l++)planeY[l]=(l-(L-1)/2)*spacing;
  const xz=ids.map(id=>{const p=layout.get(String(id));if(!p)throw new Error(`Layout lacks persona ${id}.`);return [p[0]*scale,p[1]*scale];});
  // Tie presence across planes, keyed by person-index pair.
  const lists=runs.map((run,l)=>edgesOf(run,l).map(([a,b])=>{const i=index.get(String(a)),j=index.get(String(b));if(i===undefined||j===undefined||i===j)throw new Error('Invalid tie endpoint.');return i<j?i*N+j:j*N+i;}));
  const presence=new Map();
  for(const list of lists)for(const key of new Set(list))presence.set(key,(presence.get(key)||0)+1);
  const shared=[...presence.values()].filter(c=>c===L).length;
  // Degree per plane (final saved graph or replay frame) and a stack-wide maximum so sizes compare across planes.
  const degree=new Uint16Array(L*N);
  lists.forEach((list,l)=>{for(const key of list){degree[l*N+Math.floor(key/N)]++;degree[l*N+key%N]++;}});
  const maxDegree=Math.max(1,...degree);
  const sel=persona?index.get(String(persona)):undefined;
  const nodes={count:L*N,positions:new Float32Array(L*N*3),feet:new Float32Array(L*N*3),radius:new Float32Array(L*N),degree,maxDegree,visible:new Uint8Array(L*N).fill(1)};
  for(let l=0;l<L;l++)for(let i=0;i<N;i++){
    const k=l*N+i,share=Math.sqrt(degree[k]/maxDegree),r=rMin+(rMax-rMin)*share;
    nodes.radius[k]=r;
    nodes.feet.set([xz[i][0],planeY[l],xz[i][1]],k*3);
    nodes.positions.set([xz[i][0],planeY[l]+r+(lift?liftMax*share:0),xz[i][1]],k*3);
  }
  // Ties lie in their plane. kind: 0 normal, 1 context (persona selected, tie not incident), 2 incident to persona.
  const tieCount=lists.reduce((sum,list)=>sum+list.length,0);
  const ties={count:0,positions:new Float32Array(tieCount*6),layer:new Uint8Array(tieCount),pair:new Uint16Array(tieCount*2),kind:new Uint8Array(tieCount),presence:new Uint8Array(tieCount)};
  const visibleKeys=runs.map(()=>new Set());
  lists.forEach((list,l)=>{
    for(const key of list){
      const count=presence.get(key);
      if(tieMode==='shared'&&count!==L||tieMode==='different'&&count===L)continue;
      const i=Math.floor(key/N),j=key%N,touches=sel!==undefined&&(i===sel||j===sel);
      if(incident&&sel!==undefined&&!touches)continue;
      const t=ties.count++;
      ties.positions.set([xz[i][0],planeY[l],xz[i][1],xz[j][0],planeY[l],xz[j][1]],t*6);
      ties.layer[t]=l;ties.pair[2*t]=i;ties.pair[2*t+1]=j;ties.presence[t]=count;
      ties.kind[t]=sel===undefined?0:touches?2:1;
      visibleKeys[l].add(key);
    }
  });
  if(incident&&sel!==undefined)for(let l=0;l<L;l++)for(let i=0;i<N;i++){
    const keyed=[...visibleKeys[l]].some(key=>Math.floor(key/N)===i||key%N===i);
    nodes.visible[l*N+i]=i===sel||keyed?1:0;
  }
  // Ribbons: a tie visible in two adjacent planes gets a vertical curtain between them ("shared with next layer").
  const ribbonPairs=[];
  for(let l=0;l+1<L;l++)for(const key of visibleKeys[l])if(visibleKeys[l+1].has(key))ribbonPairs.push([l,key]);
  const ribbons={count:ribbonPairs.length,positions:new Float32Array(ribbonPairs.length*18),t:new Float32Array(ribbonPairs.length*6),layer:new Uint8Array(ribbonPairs.length)};
  ribbonPairs.forEach(([l,key],q)=>{
    const i=Math.floor(key/N),j=key%N,y0=planeY[l],y1=planeY[l+1];
    const a0=[xz[i][0],y0,xz[i][1]],b0=[xz[j][0],y0,xz[j][1]],a1=[xz[i][0],y1,xz[i][1]],b1=[xz[j][0],y1,xz[j][1]];
    ribbons.positions.set([...a0,...b0,...b1,...a0,...b1,...a1],q*18);
    ribbons.t.set([0,0,1,0,1,1],q*6);ribbons.layer[q]=l;
  });
  return {layers:L,personas:ids,planeY,nodes,ties,ribbons,counts:{shared,union:presence.size,ribbons:ribbons.count},presence,N};
}

// Pure: "same persona, not a tie" thread through one persona's node on every plane.
export function identityThread(geometry,persona) {
  const i=geometry.personas.findIndex(id=>String(id)===String(persona));
  if(i<0||geometry.layers<2)return new Float32Array(0);
  const out=new Float32Array((geometry.layers-1)*6),p=geometry.nodes.positions,N=geometry.N;
  for(let l=0;l+1<geometry.layers;l++)out.set([p[(l*N+i)*3],p[(l*N+i)*3+1],p[(l*N+i)*3+2],p[((l+1)*N+i)*3],p[((l+1)*N+i)*3+1],p[((l+1)*N+i)*3+2]],l*6);
  return out;
}

// Fat screen-space line set with fixed capacity, updated in place.
class DynamicLines {
  constructor(capacity,{width=1.4,dashed=false,dashSize=3,gapSize=2.5,additive=true}={}) {
    this.capacity=capacity;
    this.geometry=new LineSegmentsGeometry();
    this.positions=new Float32Array(capacity*6);this.colors=new Float32Array(capacity*6);
    this.geometry.setPositions(this.positions);this.geometry.setColors(this.colors);
    if(dashed){
      this.distances=new Float32Array(capacity*2);
      const buffer=new THREE.InstancedInterleavedBuffer(this.distances,2,1);
      this.geometry.setAttribute('instanceDistanceStart',new THREE.InterleavedBufferAttribute(buffer,1,0));
      this.geometry.setAttribute('instanceDistanceEnd',new THREE.InterleavedBufferAttribute(buffer,1,1));
    }
    this.geometry.instanceCount=0;
    this.material=new LineMaterial({color:0xffffff,linewidth:width,vertexColors:true,transparent:true,depthWrite:false,dashed,dashSize,gapSize,blending:additive?THREE.AdditiveBlending:THREE.NormalBlending,fog:true});
    this.object=new LineSegments2(this.geometry,this.material);this.object.frustumCulled=false;
    this.count=0;
  }
  // Write `count` segments from a positions array (6 floats each) and a colour callback (index -> [r,g,b] linear).
  write(count,positions,colorAt) {
    count=Math.min(count,this.capacity);this.count=count;
    this.positions.set(positions.subarray?positions.subarray(0,count*6):positions.slice(0,count*6));
    for(let s=0;s<count;s++){const c=colorAt(s);this.colors.set([c[0],c[1],c[2],c[0],c[1],c[2]],s*6);}
    if(this.distances)for(let s=0;s<count;s++){const p=this.positions;this.distances[2*s]=0;this.distances[2*s+1]=Math.hypot(p[6*s+3]-p[6*s],p[6*s+4]-p[6*s+1],p[6*s+5]-p[6*s+2]);}
    this.geometry.attributes.instanceStart.data.needsUpdate=true;
    this.geometry.attributes.instanceColorStart.data.needsUpdate=true;
    if(this.distances)this.geometry.attributes.instanceDistanceStart.data.needsUpdate=true;
    this.geometry.instanceCount=count;this.object.visible=count>0;
  }
  dispose(){this.geometry.dispose();this.material.dispose();}
}

// Pure: split a segment into dash pieces (geometry dashes, so dashed and solid lines can share one draw call).
export function dashSegments(a,b,dash=3.2,gap=2.6) {
  const length=Math.hypot(b[0]-a[0],b[1]-a[1],b[2]-a[2]),out=[];
  if(!(length>0))return out;
  for(let t=0;t<length;t+=dash+gap){
    const t1=Math.min(length,t+dash),p=k=>[a[0]+(b[0]-a[0])*k/length,a[1]+(b[1]-a[1])*k/length,a[2]+(b[2]-a[2])*k/length];
    out.push(...p(t),...p(t1));
  }
  return out;
}

// Sprites: kind 0 = soft contact shadow on the plane under a node, kind 1 = bright outline ring around a node
// (selected persona, recorded actor). One draw call for both.
const spriteVertex=`
attribute float aSize;attribute float aKind;attribute vec3 aColor;uniform float uScale;uniform float uNodeScale;
varying float vKind;varying vec3 vColor;
#include <fog_pars_vertex>
void main(){vec4 mvPosition=modelViewMatrix*vec4(position,1.);gl_PointSize=aSize<.01?0.:max(3.,aSize*uNodeScale*uScale/max(1.,-mvPosition.z));
vKind=aKind;vColor=aColor;gl_Position=projectionMatrix*mvPosition;
#include <fog_vertex>
}`;
const spriteFragment=`
uniform float uStrength;varying float vKind;varying vec3 vColor;
void main(){vec2 c=gl_PointCoord*2.-1.;float r=length(c);if(r>1.)discard;
if(vKind<.5){float k=1.-r*r;gl_FragColor=vec4(0.,0.,0.,uStrength*k*k);return;}
float w=fwidth(r);float ring=smoothstep(.70-w,.70+w,r)*(1.-smoothstep(.93-w,.93+w,r));if(ring<.01)discard;
gl_FragColor=vec4(vColor,ring);}`;
const ribbonVertex=`
attribute float aT;varying float vT;varying vec3 vColor;varying float vDepth;
void main(){vT=aT;vColor=color;vec4 mvPosition=modelViewMatrix*vec4(position,1.);vDepth=-mvPosition.z;gl_Position=projectionMatrix*mvPosition;}`;
const ribbonFragment=`
uniform float uAlpha;uniform float uFogDensity;varying float vT;varying vec3 vColor;varying float vDepth;
void main(){float edge=pow(abs(2.*vT-1.),2.);float fog=1.-exp(-uFogDensity*uFogDensity*vDepth*vDepth);
gl_FragColor=vec4(vColor*(1.-fog),uAlpha*(.18+.82*edge));}`;

export function createStack(stage,dataset,{colors={}}={}) {
  const personas=dataset.personas,N=personas.length,CAP=STACK_CAPACITY,HALOS=64;
  const indexOf=new Map(personas.map((p,i)=>[p.id,i]));
  const group=new THREE.Group();group.name='stack';
  const state=createStateTexture(CAP);
  // 1. Glass slabs, one instance per plane.
  const slabGeometry=new THREE.BoxGeometry(1,1,1);
  slabGeometry.setAttribute('aSlot',new THREE.InstancedBufferAttribute(new Float32Array([...Array(CAP).keys()]),1));
  const slabMaterial=createSlabMaterial(state,{fill:.07,dim:.25,grid:10,rimWidth:1.8,rimAlpha:.85});
  const slabs=new THREE.InstancedMesh(slabGeometry,slabMaterial,CAP);slabs.count=0;slabs.renderOrder=1;slabs.frustumCulled=false;
  // 2. Sprites: contact shadows + outline rings.
  const spriteCount=CAP*N+HALOS;
  const spritePositions=new Float32Array(spriteCount*3),spriteSize=new Float32Array(spriteCount),spriteKind=new Float32Array(spriteCount),spriteColor=new Float32Array(spriteCount*3);
  const spriteGeometry=new THREE.BufferGeometry();
  spriteGeometry.setAttribute('position',new THREE.BufferAttribute(spritePositions,3));
  spriteGeometry.setAttribute('aSize',new THREE.BufferAttribute(spriteSize,1));
  spriteGeometry.setAttribute('aKind',new THREE.BufferAttribute(spriteKind,1));
  spriteGeometry.setAttribute('aColor',new THREE.BufferAttribute(spriteColor,3));
  spriteGeometry.setDrawRange(0,0);
  const spriteMaterial=new THREE.ShaderMaterial({uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uScale:{value:800},uNodeScale:{value:1},uStrength:{value:.55}}]),vertexShader:spriteVertex,fragmentShader:spriteFragment,transparent:true,depthWrite:false,fog:true});
  const sprites=new THREE.Points(spriteGeometry,spriteMaterial);sprites.frustumCulled=false;sprites.renderOrder=5;
  // 3. Nodes: lit instanced spheres. uNodeScale enforces a legibility floor when the camera is far away.
  const nodeScale={value:1};
  const nodeMaterial=new THREE.MeshStandardMaterial({roughness:.38,metalness:.08});
  nodeMaterial.onBeforeCompile=shader=>{shader.uniforms.uNodeScale=nodeScale;shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nuniform float uNodeScale;').replace('#include <begin_vertex>','#include <begin_vertex>\ntransformed*=uNodeScale;');};
  const nodes=new THREE.InstancedMesh(new THREE.SphereGeometry(1,22,16),nodeMaterial,CAP*N);nodes.count=0;nodes.frustumCulled=false;
  nodes.instanceColor=new THREE.InstancedBufferAttribute(new Float32Array(CAP*N*3),3);
  // 4. Thin lines: every tie plus the stems of lifted nodes. 5. Wide lines: ties incident to the persona, majority
  // consensus ties, ties added by the current recorded update, and dashed pieces (removed ties, identity thread).
  const maxTies=CAP*PAIRS(N);
  const thin=new DynamicLines(maxTies+CAP*N,{width:1.3});thin.object.renderOrder=3;
  const wide=new DynamicLines(maxTies+PAIRS(N)*24+CAP*40,{width:2.5});wide.object.renderOrder=4;
  // 6. Shared-tie ribbons between adjacent planes (Stack only).
  const ribbonCapacity=(CAP-1)*PAIRS(N);
  const ribbonPositions=new Float32Array(ribbonCapacity*18),ribbonT=new Float32Array(ribbonCapacity*6),ribbonColor=new Float32Array(ribbonCapacity*18);
  const ribbonGeometry=new THREE.BufferGeometry();
  ribbonGeometry.setAttribute('position',new THREE.BufferAttribute(ribbonPositions,3));
  ribbonGeometry.setAttribute('aT',new THREE.BufferAttribute(ribbonT,1));
  ribbonGeometry.setAttribute('color',new THREE.BufferAttribute(ribbonColor,3));
  ribbonGeometry.setDrawRange(0,0);
  const ribbonMaterial=new THREE.ShaderMaterial({uniforms:{uAlpha:{value:.075},uFogDensity:{value:.001}},vertexShader:ribbonVertex,fragmentShader:ribbonFragment,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,side:THREE.DoubleSide,vertexColors:true});
  const ribbons=new THREE.Mesh(ribbonGeometry,ribbonMaterial);ribbons.frustumCulled=false;ribbons.renderOrder=2;
  group.add(slabs,ribbons,thin.object,wide.object,nodes,sprites);
  group.visible=false;stage.scene.add(group);
  const lines=[thin,wide];
  const offResize=stage.onResize(size=>{for(const line of lines)line.material.resolution.set(size.width,size.height);});
  for(const line of lines)line.material.resolution.set(stage.size.width,stage.size.height);

  let runs=[],activeId='',layout=null,geometry=null,frame=null,persona='',incident=false,tieMode='all',showRibbons=true,consensus=null;
  let options={spacing:55,scale:1.25,plane:110,lift:false,replay:false,opacity:.5};
  let encoding={colorOf:(id,l)=>colors.layers?.[l]||STAGE_COLORS.accent,dimmed:()=>false,classOf:null};
  // Animated segments of the wide set (fade-in of new ties, fade-out of removed ones) and their full colours.
  let animation=null,animated={fresh:[],removed:[],colors:null};
  const lightV=new THREE.Vector3(),tmp=new THREE.Color(),matrix=new THREE.Matrix4(),back=new THREE.Color('#0f171b');
  const hueOf=l=>new THREE.Color(colors.layers?.[l%(colors.layers?.length||1)]||STAGE_COLORS.accent);
  const neutral=new THREE.Color(STAGE_COLORS.tie),same=new THREE.Color(colors.same||'#80e2c4'),mixed=new THREE.Color(colors.mixed||'#f2c078');
  function animate(k){
    const set=(segments,scale)=>{for(const s of segments)for(let v=0;v<6;v++)wide.colors[s*6+v]=animated.colors[s*6+v]*scale;};
    set(animated.fresh,.05+.95*k);set(animated.removed,1-.75*k);
    wide.geometry.attributes.instanceColorStart.data.needsUpdate=true;
  }
  const unhook=stage.addHook(now=>{
    if(!group.visible)return;
    keyLightInView(stage.camera,lightV);slabMaterial.uniforms.uLightV.value.copy(lightV);
    const distance=stage.camera.position.distanceTo(stage.controls.target),pxPerWorld=stage.pixelScale()/Math.max(1,distance);
    nodeScale.value=Math.max(1,2.4/(1.7*pxPerWorld));
    spriteMaterial.uniforms.uScale.value=stage.pixelScale();spriteMaterial.uniforms.uNodeScale.value=nodeScale.value;
    ribbonMaterial.uniforms.uFogDensity.value=stage.scene.fog.density;
    if(animation){
      const t=Math.min(1,(now-animation.start)/animation.duration);animate(easeInOutCubic(t));
      if(t>=1)animation=null;else return true;
    }
  });

  function planeCount(){return runs.length+(consensus?1:0);}
  function rebuild() {
    if(!runs.length||!layout){group.visible=false;stage.requestRender();return;}
    const active=runs.findIndex(r=>r.run_id===activeId);
    const edgesOf=(run)=>options.replay&&frame&&run.run_id===activeId?frame.edges:run.edges;
    const stackRuns=consensus?[...runs,{run_id:'__consensus__',edges:consensus.list.map(item=>item.edge)}]:runs;
    geometry=stackGeometry(stackRuns,layout,{personas:personas.map(p=>p.id),spacing:options.spacing,scale:options.scale,lift:options.lift,
      liftMax:options.replay?34:Math.min(24,options.spacing*.36),rMin:options.replay?3:1.7,rMax:options.replay?7.5:4.4,ties:consensus?'all':tieMode,persona,incident,edgesOf});
    const L=geometry.layers,half=options.plane,isConsensusLayer=l=>!!consensus&&l===L-1;
    // Slabs: run hue rim, active plane in focus; consensus plane in neutral gold.
    slabs.count=L;
    for(let l=0;l<L;l++){
      matrix.makeScale(half*2,1.6,half*2).setPosition(0,geometry.planeY[l]-.85,0);slabs.setMatrixAt(l,matrix);
      slabs.setColorAt(l,raw(isConsensusLayer(l)?STAGE_COLORS.gold:colors.layers?.[l]||STAGE_COLORS.accent));
      state.set(l,0,255);state.set(l,1,!options.replay&&l===active&&L>1?255:0);state.set(l,2,0);
    }
    state.commit();slabs.instanceMatrix.needsUpdate=true;if(slabs.instanceColor)slabs.instanceColor.needsUpdate=true;
    // Nodes, contact shadows, outline rings and stems.
    const event=options.replay?frame?.event:null,activeNodes=options.replay&&frame?new Set(frame.edges.flat()):null;
    const stems=[];let halo=0;nodes.count=L*N;
    spriteSize.fill(0);
    for(let l=0;l<L;l++)for(let i=0;i<N;i++){
      const k=l*N+i,id=personas[i].id,visible=geometry.nodes.visible[k],consensusNode=isConsensusLayer(l),r=consensusNode?1.5:geometry.nodes.radius[k];
      const [x,y,z]=geometry.nodes.positions.subarray(k*3,k*3+3),[fx,fy,fz]=geometry.nodes.feet.subarray(k*3,k*3+3);
      const actor=event?.persona===id,focus=!consensusNode&&(id===persona||actor),size=visible?r*(focus?1.25:1):0;
      matrix.makeScale(size,size,size).setPosition(x,consensusNode?fy+r:y,z);nodes.setMatrixAt(k,matrix);
      tmp.set(consensusNode?'#9c968a':encoding.colorOf(id,l));
      if(encoding.dimmed(id)&&id!==persona)tmp.lerp(back,.72);
      if(activeNodes&&!activeNodes.has(id)&&!focus)tmp.lerp(back,.38);
      nodes.setColorAt(k,tmp);
      spritePositions.set([fx,fy+.05,fz],k*3);spriteKind[k]=0;spriteSize[k]=visible&&!consensusNode?r*2.6+(options.lift?2:0):0;
      if(options.lift&&visible&&!consensusNode&&y-r>fy+.5)stems.push(fx,fy,fz,x,y-r,z);
      if(visible&&focus&&halo<HALOS){
        const h=CAP*N+halo++;spritePositions.set([x,y,z],h*3);spriteKind[h]=1;spriteSize[h]=size*2.5;
        tmp.set(actor?STAGE_COLORS.actor:'#f0f1eb');spriteColor.set([tmp.r,tmp.g,tmp.b],h*3);
      }
    }
    nodes.instanceMatrix.needsUpdate=true;nodes.instanceColor.needsUpdate=true;
    spriteGeometry.setDrawRange(0,CAP*N+halo);sprites.visible=L*N+halo>0;
    for(const name of ['position','aSize','aKind','aColor'])spriteGeometry.attributes[name].needsUpdate=true;
    // Tie categories and colours (linear RGB, premultiplied by intensity: lines are additive).
    const t=geometry.ties,freshKeys=new Set((event?.added||[]).map(([a,b])=>keyOf(a,b)));
    const lens=encoding.classOf;
    const tieColor=s=>{
      const l=t.layer[s],a=personas[t.pair[2*s]].id,b=personas[t.pair[2*s+1]].id;
      if(isConsensusLayer(l)){const share=consensus.share.get(keyOf(a,b))||0;return tmp.set(STAGE_COLORS.gold).multiplyScalar(.03+.97*share*share).toArray();}
      let c;const edge=[a,b],cls=lens?lens(edge):null,hex=encoding.tieColor?.(edge);
      if(hex)c=new THREE.Color(hex);else if(cls==='same')c=same.clone();else if(cls==='mixed')c=mixed.clone();
      else c=neutral.clone().lerp(hueOf(l),runs.length>1?.38:.18).multiplyScalar(lens?.35:1);
      if(encoding.tieDimmed?.(edge))c.multiplyScalar(.14);
      if(t.kind[s]===1)c.multiplyScalar(.42);
      return c.toArray();
    };
    const thinPos=[],thinCol=[],widePos=[],wideCol=[],fresh=[],removed=[];
    for(let s=0;s<t.count;s++){
      const l=t.layer[s],a=personas[t.pair[2*s]].id,b=personas[t.pair[2*s+1]].id,segment=t.positions.subarray(s*6,s*6+6);
      if(options.replay&&freshKeys.has(keyOf(a,b))){fresh.push(widePos.length/6);widePos.push(...segment);wideCol.push(tmp.set(STAGE_COLORS.accent).multiplyScalar(1.2).toArray());}
      else if(isConsensusLayer(l)&&(consensus.share.get(keyOf(a,b))||0)>=.5||t.kind[s]===2&&!isConsensusLayer(l)){widePos.push(...segment);wideCol.push(isConsensusLayer(l)?tieColor(s):tmp.set('#9fe8d0').toArray());}
      else{thinPos.push(...segment);thinCol.push(tieColor(s));}
    }
    const stemColor=tmp.set('#5f7a85').toArray();
    for(let q=0;q<stems.length/6;q++){thinPos.push(...stems.slice(q*6,q*6+6));thinCol.push(stemColor);}
    // Removed ties of the current update (Replay) and the selected persona's identity thread (Stack), dashed.
    const dash=(a,b,color,list)=>{const pieces=dashSegments(a,b);for(let q=0;q<pieces.length/6;q++){list?.push(widePos.length/6);widePos.push(...pieces.slice(q*6,q*6+6));wideCol.push(color);}};
    if(options.replay&&event)for(const edge of event.removed){
      if(incident&&persona&&!edge.includes(persona))continue;
      const i=indexOf.get(edge[0]),j=indexOf.get(edge[1]),fi=geometry.nodes.feet;
      dash([fi[i*3],fi[i*3+1],fi[i*3+2]],[fi[j*3],fi[j*3+1],fi[j*3+2]],new THREE.Color(STAGE_COLORS.remove).toArray(),removed);
    }
    const thread=persona&&!options.replay?identityThread({...geometry,layers:runs.length},persona):new Float32Array(0);
    for(let s=0;s<thread.length/6;s++)dash([...thread.subarray(s*6,s*6+3)],[...thread.subarray(s*6+3,s*6+6)],new THREE.Color(STAGE_COLORS.gold).toArray());
    thin.write(thinCol.length,new Float32Array(thinPos),k=>thinCol[k]);
    wide.write(wideCol.length,new Float32Array(widePos),k=>wideCol[k]);
    animated={fresh,removed,colors:wide.colors.slice(0,wideCol.length*6)};
    if(animation)animate(0);
    else for(const q of removed)for(let v=0;v<6;v++)wide.colors[q*6+v]*=.65; // resting state: removed ties drawn dim
    if(!animation&&removed.length)wide.geometry.attributes.instanceColorStart.data.needsUpdate=true;
    thin.material.opacity=options.opacity;wide.material.opacity=.95;
    // Ribbons between adjacent planes (Stack only).
    const rb=geometry.ribbons,count=showRibbons&&!options.replay&&!consensus?rb.count:0;
    ribbonPositions.set(rb.positions.subarray(0,count*18));ribbonT.set(rb.t.subarray(0,count*6));
    for(let q=0;q<count;q++){const c=hueOf(rb.layer[q]).lerp(hueOf(rb.layer[q]+1),.5).toArray();for(let v=0;v<6;v++)ribbonColor.set(c,q*18+v*3);}
    ribbonGeometry.setDrawRange(0,count*6);ribbons.visible=count>0; // an empty draw range still costs a draw call
    for(const name of ['position','aT','color'])ribbonGeometry.attributes[name].needsUpdate=true;
    group.visible=true;stage.requestRender();
  }

  const api={
    group,
    get geometry(){return geometry;},
    get planes(){return geometry?.layers||0;},
    // runs: run objects in plane order; layoutMap: Map(id -> [x, z]); opts: {spacing, scale, plane, lift, replay, opacity}
    setRuns(nextRuns,nextActive,layoutMap,opts={}){runs=nextRuns;activeId=nextActive;if(layoutMap)layout=layoutMap;options={...options,...opts};if(!options.replay)frame=null;rebuild();},
    setActive(id){activeId=id;rebuild();},
    // Batch update with a single in-place rewrite: any subset of the setters above.
    configure({runs:nextRuns,activeId:nextActive,layout:layoutMap,options:opts,frame:nextFrame,persona:nextPersona,isolate,ties:mode,ribbons:on,encoding:enc,consensus:agg,animate=false,duration=500}={}) {
      if(nextRuns)runs=nextRuns;if(nextActive!==undefined)activeId=nextActive;if(layoutMap)layout=layoutMap;if(opts)options={...options,...opts};
      if(nextFrame!==undefined){frame=nextFrame?{edges:nextFrame.edges,event:nextFrame.event||null}:null;animation=animate&&!stage.reducedMotion&&frame?.event?{start:performance.now(),duration}:null;}
      if(!options.replay)frame=null;
      if(nextPersona!==undefined){persona=nextPersona||'';incident=!!(isolate&&persona);}
      if(mode)tieMode=['all','shared','different'].includes(mode)?mode:'all';
      if(on!==undefined)showRibbons=!!on;
      if(enc)encoding={...encoding,...enc.node,classOf:enc.edge?.classOf||null,tieColor:enc.edge?.colorOf||null,tieDimmed:enc.edge?.dimmed||null};
      if(agg!==undefined){if(!agg?.list?.length)consensus=null;else consensus={list:agg.list,total:agg.total,share:new Map(agg.list.map(item=>[keyOf(...item.edge),item.count/(item.total||agg.total||1)]))};}
      rebuild();
    },
    setOptions(opts){options={...options,...opts};rebuild();},
    // Replay cursor: edges at the frame and the recorded update that produced it (null at event 0 or final-only).
    setFrame(edges,event,{animate=false,duration=500}={}){
      frame=edges?{edges,event:event||null}:null;
      animation=animate&&!stage.reducedMotion&&event?{start:performance.now(),duration}:null;
      rebuild();
    },
    setPersona(id,isolate=false){persona=id||'';incident=!!(isolate&&persona);rebuild();},
    // node: {colorOf(id, layerIndex) -> '#rrggbb', dimmed(id) -> bool}; edge: {classOf([a,b]) -> 'same'|'mixed'|null} or null
    setEncoding(node,edge){encoding={...encoding,...node,classOf:edge?.classOf||null,tieColor:edge?.colorOf||null,tieDimmed:edge?.dimmed||null};rebuild();},
    setTies(mode){tieMode=['all','shared','different'].includes(mode)?mode:'all';rebuild();},
    setRibbons(on){showRibbons=!!on;rebuild();},
    // Aggregate plane above the stack: list [{edge,count,total}] -> opacity and width by share. Not a run.
    setConsensus(list,total){
      if(!list?.length){consensus=null;rebuild();return;}
      const share=new Map(list.map(item=>[keyOf(...item.edge),item.count/(item.total||total||1)]));
      consensus={list,share,total:total||list[0].total};rebuild();
    },
    get consensusPlane(){return consensus&&geometry?geometry.layers-1:-1;},
    setOpacity(value){options.opacity=value;thin.material.opacity=value;stage.requestRender();},
    setVisible(visible){group.visible=!!visible&&!!runs.length;stage.requestRender();},
    get visible(){return group.visible;},
    // World position of a persona's node on a plane, and the label anchor of a plane (back-left corner).
    nodePosition(layer,id){const i=personas.findIndex(p=>p.id===id);if(!geometry||i<0||layer>=geometry.layers)return null;const k=(layer*N+i)*3,p=geometry.nodes.positions;return [p[k],p[k+1]+geometry.nodes.radius[layer*N+i]*nodeScale.value,p[k+2]];},
    nodeVisible(layer,id){const i=personas.findIndex(p=>p.id===id);return !!geometry&&i>=0&&layer<geometry.layers&&!!geometry.nodes.visible[layer*N+i];},
    planeAnchor(layer){return geometry?[-options.plane,geometry.planeY[layer]+1,options.plane]:null;},
    bounds(){
      const L=Math.max(1,planeCount()),height=(L-1)*options.spacing;
      return {center:[0,0,0],radius:Math.hypot(options.plane*1.05,height/2+(options.lift?20:6)),floor:-(height/2)-(options.replay?24:28)};
    },
    // Ray picking with a pixel tolerance: nodes first (nearest along the ray), then plane surfaces.
    pick(ray,{pixelTolerance=7}={}){
      if(!geometry||!group.visible)return null;
      const L=geometry.layers,pos=geometry.nodes.positions,o=ray.origin,d=ray.direction,worldPerPixel=1/stage.pixelScale();
      let best=null;
      for(let l=0;l<L;l++){
        if(consensus&&l===L-1)continue;
        for(let i=0;i<N;i++){
          const k=l*N+i;if(!geometry.nodes.visible[k])continue;
          const cx=pos[k*3]-o.x,cy=pos[k*3+1]-o.y,cz=pos[k*3+2]-o.z,t=cx*d.x+cy*d.y+cz*d.z;if(t<=0)continue;
          const dist2=cx*cx+cy*cy+cz*cz-t*t,reach=Math.max(geometry.nodes.radius[k]*nodeScale.value*1.15,pixelTolerance*worldPerPixel*t);
          if(dist2<=reach*reach&&(!best||t<best.t))best={t,layer:l,person:personas[i].id};
        }
      }
      if(best)return {run_id:runs[best.layer].run_id,person:best.person,layer:best.layer,distance:best.t};
      for(let l=0;l<runs.length;l++){
        if(Math.abs(d.y)<1e-9)continue;
        const t=(geometry.planeY[l]-o.y)/d.y;if(t<0)continue;
        const x=o.x+d.x*t,z=o.z+d.z*t;
        if(Math.abs(x)<=options.plane&&Math.abs(z)<=options.plane&&(!best||t<best.t))best={t,layer:l};
      }
      return best?{run_id:runs[best.layer].run_id,person:null,layer:best.layer,distance:best.t}:null;
    },
    dispose(){unhook();offResize();stage.scene.remove(group);for(const line of lines)line.dispose();for(const o of [slabs,nodes,sprites,ribbons]){o.geometry.dispose();o.material.dispose();}state.dispose();}
  };
  return api;
}
