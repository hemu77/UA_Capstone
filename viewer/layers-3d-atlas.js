// WORKSTREAM 2 (3D engine). Atlas: every one of the 896 primary attempt slots at once (SPEC.md B.1, B.3, C.4).
// 112 condition towers (16 model x method columns along X, 7 setting rows along Z), each a stack of 8 seed planes
// (Y = seed order 44000..44007, bottom to top; NOT time). Valid graphs are glass slabs carrying 50 lit node sprites
// and their ties; valid empty graphs keep their nodes and have no ties; the 6 failed attempts are dashed, hatched,
// hollow frames with no nodes and are never pickable as graphs.
// Draw calls: tower shadows 1, slabs 1 (instanced), failed frames 1, ties 1 (one buffer for all ties),
// node sprites 1 (one buffer for all nodes), hover outline 1 (only while hovering). Filter, focus, persona and
// colour changes only rewrite small textures or attributes; no geometry is rebuilt after createAtlas().
import * as THREE from 'three';
import {STAGE_COLORS,raw,createStateTexture,createSlabMaterial,keyLightInView} from './layers-3d-scene.js';

export const ATLAS_GEOMETRY=Object.freeze({slab:10,thickness:.3,pitchY:4,columnPitch:13,modelGap:9,rowPitch:19,base:2});

// Pure: slot -> centre, tower boxes and axis label anchors. slot = ((model*4+method)*7+setting)*8+seed.
export function atlasLayout(design,geometry=ATLAS_GEOMETRY) {
  const {slab,pitchY,columnPitch,modelGap,rowPitch,base}=geometry;
  const nModels=design.models.length,nMethods=design.methods.length,nSettings=design.settings.length,nSeeds=design.seeds.length;
  const columns=nModels*nMethods,total=columns*nSettings*nSeeds;
  const columnX=c=>c*columnPitch+Math.floor(c/nMethods)*modelGap;
  const width=columnX(columns-1),xOffset=width/2,depth=(nSettings-1)*rowPitch;
  const x=c=>columnX(c)-xOffset,z=s=>depth/2-s*rowPitch; // first setting in front, nearest the default camera
  const y=b=>base+b*pitchY;
  const centers=new Float32Array(total*3),towers=[];
  for(let m=0;m<nModels;m++)for(let p=0;p<nMethods;p++)for(let s=0;s<nSettings;s++){
    const column=m*nMethods+p,index=column*nSettings+s,slots=[];
    for(let b=0;b<nSeeds;b++){const slot=index*nSeeds+b;slots.push(slot);centers.set([x(column),y(b),z(s)],slot*3);}
    towers.push({index,model:design.models[m],method:design.methods[p],setting:design.settings[s],column,row:s,x:x(column),z:z(s),slots,
      min:[x(column)-slab/2,y(0)-1,z(s)-slab/2],max:[x(column)+slab/2,y(nSeeds-1)+1,z(s)+slab/2]});
  }
  const top=y(nSeeds-1),front=depth/2+slab/2,left=-xOffset-slab/2;
  const axes=[];
  for(let m=0;m<nModels;m++)axes.push({id:`model-${m}`,kind:'model',key:design.models[m],position:[(x(m*nMethods)+x(m*nMethods+nMethods-1))/2,0,front+13]});
  for(let c=0;c<columns;c++)axes.push({id:`method-${c}`,kind:'method',key:design.methods[c%nMethods],position:[x(c),0,front+5.5]});
  for(let s=0;s<nSettings;s++)axes.push({id:`setting-${s}`,kind:'setting',key:design.settings[s],position:[left-6,0,z(s)]});
  // Seed axis on the front-right tower, away from the setting labels on the left.
  const right=xOffset+slab/2;
  for(let b=0;b<nSeeds;b++)axes.push({id:`seed-${b}`,kind:'seed',key:design.seeds[b],position:[right+1.5,y(b),front]});
  axes.push({id:'seed-axis',kind:'seed-axis',key:'',position:[right+1.5,top+5,front]});
  const radius=Math.hypot(width/2+slab,depth/2+slab,top/2)*1.0;
  return {geometry,total,columns,towers,centers,axes,bounds:{center:[0,top/2,0],radius,width:width+slab,depth:depth+slab,height:top+1}};
}

// Normalised persona positions in [-1,1]^2 from whatever layout the dataset carries.
export function personaPlanePositions(personas,layout) {
  const points=personas.map((p,i)=>{
    if(layout?.xy&&layout.xy.length>=personas.length*2)return [layout.xy[2*i],layout.xy[2*i+1]];
    if(layout instanceof Map&&layout.has(p.id)){const q=layout.get(p.id);return [q[0],q[1]];}
    return [p.position[0],p.position[2]??p.position[1]];
  });
  const scale=Math.max(1e-9,...points.map(([a,b])=>Math.hypot(a,b)));
  return points.map(([a,b])=>[a/scale,b/scale]);
}

// Ray against an axis-aligned box: entry distance or Infinity.
export function rayBox(origin,direction,min,max) {
  let near=-Infinity,far=Infinity;
  for(let k=0;k<3;k++){
    if(Math.abs(direction[k])<1e-12){if(origin[k]<min[k]||origin[k]>max[k])return Infinity;continue;}
    let t1=(min[k]-origin[k])/direction[k],t2=(max[k]-origin[k])/direction[k];
    if(t1>t2)[t1,t2]=[t2,t1];
    near=Math.max(near,t1);far=Math.min(far,t2);
    if(near>far||far<0)return Infinity;
  }
  return near>=0?near:0;
}

// Pure picking against the layout: nearest tower hit, then the seed plane the ray meets first inside it.
export function pickSlot(layout,origin,direction) {
  const {slab}=layout.geometry,half=slab/2;
  let best=null;
  for(const tower of layout.towers){const t=rayBox(origin,direction,tower.min,tower.max);if(t<(best?.t??Infinity))best={tower,t};}
  if(!best)return null;
  let slot=null,nearest=Infinity;
  for(const candidate of best.tower.slots){
    const cy=layout.centers[candidate*3+1];
    if(Math.abs(direction[1])<1e-9)continue;
    const t=(cy-origin[1])/direction[1];if(t<0)continue;
    const px=origin[0]+direction[0]*t-best.tower.x,pz=origin[2]+direction[2]*t-best.tower.z;
    if(Math.abs(px)<=half&&Math.abs(pz)<=half&&t<nearest){nearest=t;slot=candidate;}
  }
  if(slot===null){
    // Entered through a side between planes: take the plane closest to the entry height.
    const entryY=origin[1]+direction[1]*best.t;
    slot=best.tower.slots.reduce((a,b)=>Math.abs(layout.centers[b*3+1]-entryY)<Math.abs(layout.centers[a*3+1]-entryY)?b:a);
  }
  return {tower:best.tower.index,slot,distance:best.t};
}

const pointVertex=`
attribute float aSlot;attribute float aPerson;attribute float aSize;
uniform sampler2D uState;uniform sampler2D uPersona;uniform float uScale;uniform float uMinPx;uniform float uMaxPx;
varying vec3 vColor;varying float vEmph;varying float vSel;
#include <fog_pars_vertex>
void main(){
  vec4 slot=texelFetch(uState,ivec2(int(aSlot+.5),0),0);
  vec4 persona=texelFetch(uPersona,ivec2(int(aPerson+.5),0),0);
  vec4 flags=texelFetch(uPersona,ivec2(int(aPerson+.5),1),0);
  float match=slot.r,focus=slot.g,sel=flags.r,dim=flags.g;
  vec4 mvPosition=modelViewMatrix*vec4(position,1.);
  float world=aSize*(1.+sel*1.4+focus*.35);
  float px=world*uScale/max(1.,-mvPosition.z);
  gl_PointSize=clamp(px,mix(.9,uMinPx,match)*(1.+sel*.8),uMaxPx);
  vColor=mix(persona.rgb,vec3(1.,.86,.52),sel);
  vEmph=mix(.12,1.,match)*mix(1.,.3,dim*(1.-sel));
  vSel=sel;
  gl_Position=projectionMatrix*mvPosition;
  #include <fog_vertex>
}`;
const pointFragment=`
uniform vec3 uLightV;uniform vec3 uBack;
varying vec3 vColor;varying float vEmph;varying float vSel;
#include <fog_pars_fragment>
void main(){
  vec2 c=gl_PointCoord*2.-1.;float r2=dot(c,c);if(r2>1.)discard;
  vec3 n=vec3(c.x,-c.y,sqrt(1.-r2));
  float diff=max(dot(n,uLightV),0.),spec=pow(max(dot(reflect(-uLightV,n),vec3(0.,0.,1.)),0.),28.);
  vec3 col=vColor*(.34+.8*diff)+vec3(spec*.45);
  col*=1.-.35*smoothstep(.72,1.,r2);
  col=mix(uBack,col,vEmph);
  gl_FragColor=vec4(col,1.);
  #include <fog_fragment>
}`;
const tieVertex=`
attribute float aSlot;attribute float aClass;
uniform sampler2D uState;uniform vec3 uTie;uniform vec3 uSame;uniform vec3 uMixed;uniform float uLens;
varying vec3 vColor;varying float vAlpha;varying float vDepth;
void main(){
  vec4 slot=texelFetch(uState,ivec2(int(aSlot+.5),0),0);
  vec3 col=uTie;float a=1.;
  if(uLens>.5){col=aClass<.5?uTie:(aClass<1.5?uSame:uMixed);a=aClass<.5?.25:1.15;}
  vColor=col;vAlpha=a*slot.r*(1.+slot.g*1.2);
  vec4 mvPosition=modelViewMatrix*vec4(position,1.);vDepth=-mvPosition.z;
  gl_Position=projectionMatrix*mvPosition;
}`;
const tieFragment=`
uniform float uAlpha;uniform float uFogDensity;
varying vec3 vColor;varying float vAlpha;varying float vDepth;
void main(){
  if(vAlpha<.001)discard;
  float fog=1.-exp(-uFogDensity*uFogDensity*vDepth*vDepth);
  gl_FragColor=vec4(vColor*(1.-fog),uAlpha*vAlpha);
}`;
const frameVertex=`
attribute float aSlot;attribute float aDist;
uniform sampler2D uState;
varying float vDist;varying float vMatch;
#include <fog_pars_vertex>
void main(){
  vMatch=texelFetch(uState,ivec2(int(aSlot+.5),0),0).r;vDist=aDist;
  vec4 mvPosition=modelViewMatrix*vec4(position,1.);gl_Position=projectionMatrix*mvPosition;
  #include <fog_vertex>
}`;
const frameFragment=`
uniform vec3 uColor;uniform float uDash;
varying float vDist;varying float vMatch;
#include <fog_pars_fragment>
void main(){
  if(fract(vDist/uDash)>.55)discard;
  gl_FragColor=vec4(uColor,mix(.18,.95,vMatch));
  #include <fog_fragment>
}`;
const shadowVertex=`
varying vec2 vUv;
#include <fog_pars_vertex>
void main(){vUv=uv;vec4 mvPosition=viewMatrix*modelMatrix*instanceMatrix*vec4(position,1.);gl_Position=projectionMatrix*mvPosition;
#include <fog_vertex>
}`;
const shadowFragment=`
uniform float uStrength;varying vec2 vUv;
#include <fog_pars_fragment>
void main(){
  vec2 q=abs(vUv-.5)*2.;float d=max(q.x,q.y);
  float a=(1.-smoothstep(.45,1.,d))*uStrength;
  gl_FragColor=vec4(0.,0.,0.,a);
}`;

export function createAtlas(stage,dataset,{layout=dataset.layout,colors={}}={}) {
  const design=dataset.design,grid=atlasLayout(design),{slab,thickness}=grid.geometry;
  const personas=dataset.personas,personIndex=new Map(personas.map((p,i)=>[p.id,i]));
  const plane=personaPlanePositions(personas,layout).map(([a,b])=>[a*slab*.43,b*slab*.43]);
  const runBySlot=new Map(dataset.runs.map(run=>[run.slot,run]));
  const statusOf=slot=>dataset.slots[slot]?.status||'failed';
  const group=new THREE.Group();group.name='atlas';
  const state=createStateTexture(grid.total);
  for(let slot=0;slot<grid.total;slot++){state.set(slot,0,255);state.set(slot,3,{valid:0,empty:128,failed:255}[statusOf(slot)]);}
  state.commit();
  // Persona palette: row 0 colour, row 1 flags (r = selected persona, g = dimmed group).
  const paletteWidth=Math.max(64,personas.length);
  const paletteData=new Uint8Array(paletteWidth*2*4);
  const palette=new THREE.DataTexture(paletteData,paletteWidth,2,THREE.RGBAFormat,THREE.UnsignedByteType);
  palette.magFilter=palette.minFilter=THREE.NearestFilter;palette.needsUpdate=true;
  const lightV=new THREE.Vector3();

  // Tower contact shadows on the ground.
  const shadowMaterial=new THREE.ShaderMaterial({uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uStrength:{value:.55}}]),vertexShader:shadowVertex,fragmentShader:shadowFragment,transparent:true,depthWrite:false,fog:true});
  const shadows=new THREE.InstancedMesh(new THREE.PlaneGeometry(1,1),shadowMaterial,grid.towers.length);
  const matrix=new THREE.Matrix4(),rotation=new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI/2,0,0));
  grid.towers.forEach((tower,i)=>{matrix.compose(new THREE.Vector3(tower.x+.8,.03,tower.z-.6),rotation,new THREE.Vector3(slab*1.55,slab*1.55,1));shadows.setMatrixAt(i,matrix);});
  shadows.renderOrder=-5;shadows.frustumCulled=false;group.add(shadows);

  // Slabs for every graph slot (valid and valid-empty). Failed attempts get frames instead.
  const graphSlots=[];for(let slot=0;slot<grid.total;slot++)if(statusOf(slot)!=='failed')graphSlots.push(slot);
  const slabGeometry=new THREE.BoxGeometry(1,1,1);
  const slotAttribute=new Float32Array(graphSlots.length);
  slabGeometry.setAttribute('aSlot',new THREE.InstancedBufferAttribute(slotAttribute,1));
  const slabMaterial=createSlabMaterial(state,{fill:.04,dim:.1,grid:0,rimWidth:1.2,rimAlpha:.6});
  const slabs=new THREE.InstancedMesh(slabGeometry,slabMaterial,graphSlots.length);
  const validRim=raw(colors.valid||STAGE_COLORS.accent),emptyRim=raw(colors.empty||STAGE_COLORS.empty);
  graphSlots.forEach((slot,i)=>{
    slotAttribute[i]=slot;
    matrix.makeScale(slab,thickness,slab).setPosition(grid.centers[slot*3],grid.centers[slot*3+1],grid.centers[slot*3+2]);
    slabs.setMatrixAt(i,matrix);slabs.setColorAt(i,statusOf(slot)==='empty'?emptyRim:validRim);
  });
  slabs.renderOrder=1;slabs.frustumCulled=false;group.add(slabs);

  // Failed attempts: dashed hollow frame with three hatch strokes, no fill, no nodes.
  const failedSlots=[];for(let slot=0;slot<grid.total;slot++)if(statusOf(slot)==='failed')failedSlots.push(slot);
  const framePositions=[],frameSlots=[],frameDist=[];
  for(const slot of failedSlots){
    const [cx,cy,cz]=[grid.centers[slot*3],grid.centers[slot*3+1],grid.centers[slot*3+2]],h=slab/2;
    const corners=[[-h,-h],[h,-h],[h,h],[-h,h]];
    const segments=corners.map((c,i)=>[c,corners[(i+1)%4]]);
    segments.push([[-h,-h],[h,h]],[[0,-h],[h,0]],[[-h,0],[0,h]]); // hatch: three parallel diagonals
    for(const [[ax,az],[bx,bz]] of segments){
      framePositions.push(cx+ax,cy,cz+az,cx+bx,cy,cz+bz);frameSlots.push(slot,slot);frameDist.push(0,Math.hypot(bx-ax,bz-az));
    }
  }
  const frameGeometry=new THREE.BufferGeometry();
  frameGeometry.setAttribute('position',new THREE.Float32BufferAttribute(framePositions,3));
  frameGeometry.setAttribute('aSlot',new THREE.Float32BufferAttribute(frameSlots,1));
  frameGeometry.setAttribute('aDist',new THREE.Float32BufferAttribute(frameDist,1));
  const frameMaterial=new THREE.ShaderMaterial({uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uState:{value:null},uColor:{value:raw(colors.failed||STAGE_COLORS.failed)},uDash:{value:.9}}]),vertexShader:frameVertex,fragmentShader:frameFragment,transparent:true,depthWrite:false,fog:true});
  frameMaterial.uniforms.uState.value=state.texture;
  const frames=new THREE.LineSegments(frameGeometry,frameMaterial);frames.renderOrder=2;frames.frustumCulled=false;group.add(frames);

  // Nodes: one point buffer for every persona on every graph slot (failed slots have none).
  const runs=graphSlots.map(slot=>runBySlot.get(slot));
  const nodeCount=runs.length*personas.length;
  const pointPositions=new Float32Array(nodeCount*3),pointSlot=new Float32Array(nodeCount),pointPerson=new Float32Array(nodeCount),pointSize=new Float32Array(nodeCount);
  const degree=new Uint16Array(personas.length);
  let edgeTotal=0;for(const run of runs)edgeTotal+=run.edges.length;
  const tiePositions=new Float32Array(edgeTotal*6),tieSlot=new Float32Array(edgeTotal*2),tieClass=new Float32Array(edgeTotal*2),tiePairs=new Uint8Array(edgeTotal*2);
  let n=0,e=0;
  const lift=thickness/2+.06;
  runs.forEach((run,r)=>{
    const slot=graphSlots[r],cx=grid.centers[slot*3],cy=grid.centers[slot*3+1]+lift,cz=grid.centers[slot*3+2];
    degree.fill(0);
    for(const [a,b] of run.edges){
      const i=personIndex.get(a),j=personIndex.get(b);degree[i]++;degree[j]++;
      tiePositions.set([cx+plane[i][0],cy,cz+plane[i][1],cx+plane[j][0],cy,cz+plane[j][1]],e*6);
      tieSlot[e*2]=tieSlot[e*2+1]=slot;tiePairs[e*2]=i;tiePairs[e*2+1]=j;e++;
    }
    const maxDegree=Math.max(1,...degree);
    for(let i=0;i<personas.length;i++){
      pointPositions.set([cx+plane[i][0],cy+.08,cz+plane[i][1]],n*3);
      pointSlot[n]=slot;pointPerson[n]=i;pointSize[n]=.2+.34*Math.sqrt(degree[i]/maxDegree);n++;
    }
  });
  const pointGeometry=new THREE.BufferGeometry();
  pointGeometry.setAttribute('position',new THREE.BufferAttribute(pointPositions,3));
  pointGeometry.setAttribute('aSlot',new THREE.BufferAttribute(pointSlot,1));
  pointGeometry.setAttribute('aPerson',new THREE.BufferAttribute(pointPerson,1));
  pointGeometry.setAttribute('aSize',new THREE.BufferAttribute(pointSize,1));
  const pointMaterial=new THREE.ShaderMaterial({uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uState:{value:null},uPersona:{value:null},uScale:{value:800},uMinPx:{value:1.4},uMaxPx:{value:12},uLightV:{value:new THREE.Vector3(0,0,1)},uBack:{value:raw('#111a1e')}}]),vertexShader:pointVertex,fragmentShader:pointFragment,fog:true});
  pointMaterial.uniforms.uState.value=state.texture;pointMaterial.uniforms.uPersona.value=palette;
  const points=new THREE.Points(pointGeometry,pointMaterial);points.frustumCulled=false;points.renderOrder=3;group.add(points);

  // Ties: one segment buffer with every primary tie; out-of-scope towers lose their ties via the state texture.
  const tieGeometry=new THREE.BufferGeometry();
  tieGeometry.setAttribute('position',new THREE.BufferAttribute(tiePositions,3));
  tieGeometry.setAttribute('aSlot',new THREE.BufferAttribute(tieSlot,1));
  const classAttribute=new THREE.BufferAttribute(tieClass,1);tieGeometry.setAttribute('aClass',classAttribute);
  const tieMaterial=new THREE.ShaderMaterial({uniforms:{uState:{value:state.texture},uTie:{value:raw(colors.tie||'#5a9fb2')},uSame:{value:raw(colors.same||'#80e2c4')},uMixed:{value:raw(colors.mixed||'#f2c078')},uLens:{value:0},uAlpha:{value:.3},uFogDensity:{value:.0012}},vertexShader:tieVertex,fragmentShader:tieFragment,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending});
  const ties=new THREE.LineSegments(tieGeometry,tieMaterial);ties.frustumCulled=false;ties.renderOrder=4;group.add(ties);

  // Hover outline around one tower.
  const outline=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial({color:STAGE_COLORS.accent,transparent:true,opacity:.9,depthWrite:false,fog:false}));
  outline.visible=false;outline.renderOrder=6;group.add(outline);
  stage.scene.add(group);

  let match=new Uint8Array(grid.total).fill(1),focus=null,hover=null,stacked=new Set(),opacity=.3;
  const unhook=stage.addHook(()=>{
    if(!group.visible)return;
    keyLightInView(stage.camera,lightV);
    pointMaterial.uniforms.uLightV.value.copy(lightV);slabMaterial.uniforms.uLightV.value.copy(lightV);
    pointMaterial.uniforms.uScale.value=stage.pixelScale();
    tieMaterial.uniforms.uFogDensity.value=stage.scene.fog.density;
    // Thin out ties when the whole atlas is in view; full strength when close to a tower.
    const distance=stage.camera.position.distanceTo(stage.controls.target);
    tieMaterial.uniforms.uAlpha.value=opacity*THREE.MathUtils.clamp(70/distance,.16,.5);
  });

  const api={
    group,layout:grid,
    get slots(){return grid.total;},
    get drawables(){return {slabs:graphSlots.length,failed:failedSlots.length,nodes:nodeCount,ties:edgeTotal};},
    // mask: Uint8Array(896), 1 = in filter scope. Out-of-scope towers drop to 12% and lose their ties.
    setMatch(mask){match=mask;for(let slot=0;slot<grid.total;slot++)state.set(slot,0,mask[slot]?255:0);state.commit();stage.requestRender();},
    setFocus(slot){
      if(focus!==null)state.set(focus,1,0);
      focus=Number.isInteger(slot)&&slot>=0&&slot<grid.total?slot:null;
      if(focus!==null)state.set(focus,1,255);
      state.commit();api.setHover(focus===null?null:Math.floor(focus/design.seeds.length));
    },
    setHover(towerIndex){
      hover=Number.isInteger(towerIndex)?grid.towers[towerIndex]:null;
      outline.visible=!!hover;
      if(hover){const [x0,y0,z0]=hover.min,[x1,y1,z1]=hover.max;outline.position.set((x0+x1)/2,(y0+y1)/2,(z0+z1)/2);outline.scale.set(x1-x0+.8,y1-y0+.6,z1-z0+.8);}
      stage.requestRender();
    },
    setStacked(slots){for(const slot of stacked)state.set(slot,2,0);stacked=new Set(slots.filter(Number.isInteger));for(const slot of stacked)state.set(slot,2,255);state.commit();stage.requestRender();},
    // enc: {colorOf(personId)->'#rrggbb', dimmed(personId)->bool, persona:'', classOf([a,b])->'same'|'mixed'|null|undefined}
    setEncoding({colorOf,dimmed=()=>false,persona='',classOf=null}={}){
      const color=new THREE.Color();
      personas.forEach((p,i)=>{
        color.setStyle(colorOf(p.id),THREE.LinearSRGBColorSpace);
        paletteData.set([Math.round(color.r*255),Math.round(color.g*255),Math.round(color.b*255),255],i*4);
        paletteData.set([p.id===persona?255:0,dimmed(p.id)?255:0,0,255],(paletteWidth+i)*4);
      });
      palette.needsUpdate=true;
      if(classOf){
        const table=new Uint8Array(personas.length*personas.length);
        for(let i=0;i<personas.length;i++)for(let j=i+1;j<personas.length;j++){const c=classOf([personas[i].id,personas[j].id]);table[i*personas.length+j]=table[j*personas.length+i]=c==='same'?1:c==='mixed'?2:0;}
        for(let k=0;k<edgeTotal;k++)tieClass[2*k]=tieClass[2*k+1]=table[tiePairs[2*k]*personas.length+tiePairs[2*k+1]];
        classAttribute.needsUpdate=true;tieMaterial.uniforms.uLens.value=1;
      }else tieMaterial.uniforms.uLens.value=0;
      stage.requestRender();
    },
    setOpacity(value){opacity=value;stage.requestRender();},
    setVisible(visible){group.visible=!!visible;if(!visible)outline.visible=false;stage.requestRender();},
    get visible(){return group.visible;},
    // Picking: {slot, tower, status, run_id|null}. Failed slots are reported as failures, never as graphs.
    pick(ray){
      const hit=pickSlot(grid,ray.origin.toArray(),ray.direction.toArray());
      if(!hit)return null;
      const status=statusOf(hit.slot),run=runBySlot.get(hit.slot)||null;
      return {...hit,status,run_id:status==='failed'?null:run?.run_id||null};
    },
    towerOf(slot){return grid.towers[Math.floor(slot/design.seeds.length)];},
    // Close framing of one tower (used for the fly-in before the Stack opens).
    towerFraming(index){
      const tower=grid.towers[index],cx=tower.x,cz=tower.z,cy=(tower.min[1]+tower.max[1])/2;
      const offset=new THREE.Vector3().subVectors(stage.camera.position,stage.controls.target).normalize().multiplyScalar(slab*3.2);
      return {position:new THREE.Vector3(cx,cy,cz).add(offset),target:new THREE.Vector3(cx,cy,cz)};
    },
    dispose(){unhook();stage.scene.remove(group);group.traverse(o=>{o.geometry?.dispose();o.material?.dispose?.();});state.dispose();palette.dispose();}
  };
  return api;
}
