// WORKSTREAM 2 (3D engine). Atlas: every one of the 896 primary attempt slots at once (SPEC.md B.1, B.3, C.4).
// 112 condition towers (16 model x method columns along X, 7 setting rows along Z), each a stack of 8 seed planes
// (Y = seed order 44000..44007, bottom to top; NOT time).
// Overview: every plane is a solid tile. A valid graph's tile is tinted by its tie count (brighter = more ties);
// a valid empty graph is a hollow neutral tile; a failed attempt is a hollow, hatched coral frame with no nodes and is
// never pickable as a graph. Rows are spaced so no tower hides the row behind it at the default camera angle.
// Close up (camera near a few towers): the 50 nodes and the real ties of each nearby plane fade in by distance, and
// the tiles turn translucent so lower planes stay visible. At overview zoom nodes and ties are not drawn at all.
// Draw calls: ground plinths + tower shadows 1, tiles 1 (instanced, all 896 slots), failed frames 1, nodes 1 and ties 1
// (only when close), hover outline 1 (only while hovering). Filter, focus, persona and colour changes only rewrite
// small textures or attributes; no geometry is rebuilt after createAtlas().
import * as THREE from 'three';
import {STAGE_COLORS,raw,createStateTexture,keyLightInView} from './layers-3d-scene.js';
import {spreadLayout} from './layers-3d-stack.js';

export const ATLAS_GEOMETRY=Object.freeze({slab:8,thickness:.5,pitchY:2,columnPitch:10.5,modelGap:7.5,rowPitch:20,base:1}); // binary-exact steps (Float32 centres)

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
  const top=y(nSeeds-1),front=depth/2+slab/2,left=-xOffset-slab/2,right=xOffset+slab/2;
  const axes=[];
  // Model names sit in front of their block; method names directly under each column (two staggered lines so full
  // words never collide); setting names at the left end of their own row, on the ground.
  for(let m=0;m<nModels;m++)axes.push({id:`model-${m}`,kind:'model',key:design.models[m],position:[(x(m*nMethods)+x(m*nMethods+nMethods-1))/2,0,front+15.5]});
  for(let c=0;c<columns;c++)axes.push({id:`method-${c}`,kind:'method',key:design.methods[c%nMethods],stagger:c%2,position:[x(c),0,front+(c%2?9.4:2.6)]});
  for(let s=0;s<nSettings;s++)axes.push({id:`setting-${s}`,kind:'setting',key:design.settings[s],position:[left-2.5,0,z(s)]});
  // Seed axis on the front-right tower, away from the setting labels on the left.
  for(let b=0;b<nSeeds;b++)axes.push({id:`seed-${b}`,kind:'seed',key:design.seeds[b],position:[right+1.2,y(b),front-slab/2]});
  axes.push({id:'seed-axis',kind:'seed-axis',key:'',position:[right+1.2,top+17,front-slab/2]});
  const blocks=[];
  for(let m=0;m<nModels;m++)blocks.push({model:design.models[m],min:[x(m*nMethods)-slab/2,-depth/2-slab/2],max:[x(m*nMethods+nMethods-1)+slab/2,depth/2+slab/2]});
  const radius=Math.hypot(width/2+slab,depth/2+slab,top/2);
  return {geometry,total,columns,towers,centers,axes,blocks,bounds:{center:[0,top/2,0],radius,width:width+slab,depth:depth+slab,height:top+1}};
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

// Pure: tile tint in [0,1] from a tie count (perceptual: square root of the share of the largest graph).
export const tieTint=(ties,max)=>max>0?Math.sqrt(Math.max(0,Math.min(1,ties/max))):0;

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

// Tiles. State texture: r = in filter scope, g = focus (255 hovered plane, 128 rest of the hovered tower),
// b = in stack, a = status (0 valid, 128 valid empty graph, 255 failed attempt).
const tileVertex=`
attribute float aSlot;attribute float aTint;
uniform sampler2D uState;
varying vec3 vLocal;varying vec3 vNormalV;varying vec4 vState;varying float vTint;varying vec3 vView;
#include <fog_pars_vertex>
void main(){
  vLocal=position;vTint=aTint;
  vState=texelFetch(uState,ivec2(int(aSlot+.5),0),0);
  mat4 m=modelMatrix*instanceMatrix;
  vec4 mvPosition=viewMatrix*m*vec4(position,1.);
  vNormalV=normalize(mat3(viewMatrix)*mat3(m)*normal);
  vView=-mvPosition.xyz;
  gl_Position=projectionMatrix*mvPosition;
  #include <fog_vertex>
}`;
const tileFragment=`
uniform vec3 uLightV;uniform vec3 uLow;uniform vec3 uMid;uniform vec3 uHigh;uniform vec3 uEmpty;uniform vec3 uFailed;
uniform vec3 uGold;uniform vec3 uAccent;uniform vec3 uBack;uniform float uClose;
varying vec3 vLocal;varying vec3 vNormalV;varying vec4 vState;varying float vTint;varying vec3 vView;
#include <fog_pars_fragment>
void main(){
  float match=vState.r,focus=vState.g,stacked=vState.b,status=vState.a;
  float empty=step(.3,status)*(1.-step(.8,status)),failed=step(.8,status);
  vec3 n=normalize(vNormalV);
  float isTop=step(.499,vLocal.y),isBottom=step(vLocal.y,-.499),side=1.-isTop-isBottom;
  vec2 uv=vLocal.xz+.5;
  float edge=min(min(uv.x,1.-uv.x),min(uv.y,1.-uv.y));
  float px=edge/max(fwidth(edge),1e-6);
  float rim=(1.-smoothstep(.6,1.6,px))*isTop;
  // Graph tile: tie-count ramp, deep teal -> mint. Empty: neutral and hollow. Failed: coral hatch, hollow.
  vec3 ramp=mix(mix(uLow,uMid,smoothstep(0.,.55,vTint)),uHigh,smoothstep(.55,1.,vTint));
  vec3 base=ramp;
  float hollow=0.;
  if(empty>.5){base=uEmpty*.28;hollow=1.;}
  if(failed>.5){
    float h=fract((uv.x+uv.y)*3.5);float stripe=1.-smoothstep(.36,.36+fwidth((uv.x+uv.y)*3.5)*1.5,abs(h-.5));
    base=mix(uBack*1.4,uFailed,stripe*.9);hollow=1.;
  }
  float lit=.62+.38*max(dot(n,uLightV),0.);
  vec3 col=base*(isTop*lit+side*.36+isBottom*.2);
  col=mix(col,uGold*(isTop*.9+side*.75),stacked*.85);
  // Crisp plane edge: a light rim on the top face (gold when in your stack, accent on the hovered tower).
  vec3 rimColor=mix(ramp*1.35+.06,uEmpty,empty);
  rimColor=mix(rimColor,uFailed*1.2,failed);
  rimColor=mix(rimColor,uGold,stacked);
  rimColor=mix(rimColor,uAccent*1.15,step(.3,focus)*(1.-stacked));
  float rimStrength=mix(.55,1.,max(hollow,max(stacked,step(.3,focus))));
  col=mix(col,rimColor,rim*rimStrength);
  // Hovered tower brightens; the hovered plane brightens most.
  col*=1.+step(.3,focus)*.28+step(.8,focus)*.35;
  // Out of filter scope: recede to a dark, desaturated ghost.
  float luma=dot(col,vec3(.3,.59,.11));
  col=mix(mix(uBack,vec3(luma),.18)*1.6,col,match);
  float a=mix(.82,1.,isTop+side*.5);
  a*=mix(1.,mix(.55,.95,rim),hollow*isTop);
  // Close up: tiles turn to tinted glass so the nodes and ties drawn on every plane stay readable.
  // The top face is nearly clear close up: each plane above veils the ties below it only a little, so a
  // tower reads top-down with its upper graphs crisp (the rim and side walls keep the slab shape).
  a*=mix(1.,mix(.13,.9,rim+side),uClose);col*=mix(1.,.42,uClose*isTop*(1.-rim));
  a*=mix(.55,1.,match);
  gl_FragColor=vec4(col,clamp(a,0.,1.));
  #include <fog_fragment>
}`;
const pointVertex=`
attribute float aSlot;attribute float aPerson;attribute float aSize;
uniform sampler2D uState;uniform sampler2D uPersona;uniform float uScale;uniform float uMinPx;uniform float uMaxPx;uniform float uNear;uniform float uFar;
varying vec3 vColor;varying float vEmph;varying float vFade;
void main(){
  vec4 slot=texelFetch(uState,ivec2(int(aSlot+.5),0),0);
  vec4 persona=texelFetch(uPersona,ivec2(int(aPerson+.5),0),0);
  vec4 flags=texelFetch(uPersona,ivec2(int(aPerson+.5),1),0);
  float match=slot.r,sel=flags.r,dim=flags.g;
  vec4 mvPosition=modelViewMatrix*vec4(position,1.);
  float depth=-mvPosition.z;
  vFade=(1.-smoothstep(uNear,uFar,depth))*mix(.15,1.,match);
  float world=aSize*(1.+sel*1.2);
  gl_PointSize=vFade<.02?0.:clamp(world*uScale/max(1.,depth),uMinPx*(1.+sel),uMaxPx);
  vColor=mix(persona.rgb,vec3(1.,.86,.52),sel);
  vEmph=mix(1.,.3,dim*(1.-sel));
  gl_Position=projectionMatrix*mvPosition;
}`;
const pointFragment=`
uniform vec3 uLightV;uniform vec3 uBack;
varying vec3 vColor;varying float vEmph;varying float vFade;
void main(){
  if(vFade<.02)discard;
  vec2 c=gl_PointCoord*2.-1.;float r2=dot(c,c);if(r2>1.)discard;
  vec3 n=vec3(c.x,-c.y,sqrt(1.-r2));
  float diff=max(dot(n,uLightV),0.);
  vec3 col=vColor*(.55+.6*diff);
  col=mix(col,col*.55,smoothstep(.6,1.,r2));
  col=mix(uBack,col,vEmph*vFade);
  gl_FragColor=vec4(col,1.);
}`;
const tieVertex=`
attribute float aSlot;attribute float aClass;
uniform sampler2D uState;uniform vec3 uTie;uniform vec3 uSame;uniform vec3 uMixed;uniform float uLens;uniform float uNear;uniform float uFar;
varying vec3 vColor;varying float vAlpha;
void main(){
  vec4 slot=texelFetch(uState,ivec2(int(aSlot+.5),0),0);
  vec3 col=uTie;float a=1.;
  if(uLens>.5){col=aClass<.5?uTie:(aClass<1.5?uSame:uMixed);a=aClass<.5?.3:1.1;}
  vec4 mvPosition=modelViewMatrix*vec4(position,1.);
  vColor=col;vAlpha=a*slot.r*(1.+slot.g*.6)*(1.-smoothstep(uNear,uFar,-mvPosition.z));
  gl_Position=projectionMatrix*mvPosition;
}`;
const tieFragment=`
uniform float uAlpha;
varying vec3 vColor;varying float vAlpha;
void main(){
  if(vAlpha<.01)discard;
  gl_FragColor=vec4(vColor,clamp(uAlpha*vAlpha,0.,1.));
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
  if(fract(vDist/uDash)>.6)discard;
  gl_FragColor=vec4(uColor,mix(.25,1.,vMatch));
  #include <fog_fragment>
}`;
// Ground pieces in one draw: kind 0 = soft contact shadow under a tower, kind 1 = model block plinth (a quiet raised
// panel with a hairline edge that groups the four method columns of one model).
const groundVertex=`
attribute float aKind;varying vec2 vUv;varying float vKind;varying vec2 vSize;
#include <fog_pars_vertex>
void main(){vUv=uv;vKind=aKind;vSize=vec2(length(instanceMatrix[0].xyz),length(instanceMatrix[2].xyz));
vec4 mvPosition=viewMatrix*modelMatrix*instanceMatrix*vec4(position,1.);gl_Position=projectionMatrix*mvPosition;
#include <fog_vertex>
}`;
const groundFragment=`
uniform float uStrength;uniform vec3 uPlinth;uniform vec3 uPlinthEdge;varying vec2 vUv;varying float vKind;varying vec2 vSize;
#include <fog_pars_fragment>
void main(){
  if(vKind<.5){vec2 q=abs(vUv-.5)*2.;float d=max(q.x,q.y);gl_FragColor=vec4(0.,0.,0.,(1.-smoothstep(.5,1.,d))*uStrength);}
  else{
    vec2 p=(vUv-.5)*vSize,h=vSize*.5-2.;float r=2.;
    vec2 q=abs(p)-h+r;float d=length(max(q,0.))+min(max(q.x,q.y),0.)-r;
    float w=fwidth(d);float inside=1.-smoothstep(-w,w,d);float line=1.-smoothstep(0.,w*1.5,abs(d+w));
    gl_FragColor=vec4(mix(uPlinth,uPlinthEdge,line),max(inside*.72,line*.9));
  }
  #include <fog_fragment>
}`;

export function createAtlas(stage,dataset,{layout=dataset.layout,colors={}}={}) {
  const design=dataset.design,grid=atlasLayout(design),{slab,thickness}=grid.geometry;
  const personas=dataset.personas,personIndex=new Map(personas.map((p,i)=>[p.id,i]));
  // Same treatment as the Stack: the shared layout spread evenly over a disc inscribed in each tile.
  const spread=spreadLayout(new Map(personaPlanePositions(personas,layout).map((p,i)=>[i,p])));
  const plane=personas.map((_,i)=>spread.get(i).map(v=>v*slab*.43));
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
  const lightV=new THREE.Vector3(),matrix=new THREE.Matrix4();
  const LOD={points:[200,300],ties:[165,260]}; // camera depth (world units) where nodes / ties are full and gone

  // Ground: tower contact shadows + model plinths, one instanced draw.
  const pieces=grid.towers.length+grid.blocks.length;
  const groundGeometry=new THREE.PlaneGeometry(1,1).rotateX(-Math.PI/2);
  const kinds=new Float32Array(pieces);groundGeometry.setAttribute('aKind',new THREE.InstancedBufferAttribute(kinds,1));
  const groundMaterial=new THREE.ShaderMaterial({uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uStrength:{value:.5},uPlinth:{value:raw('#111b20')},uPlinthEdge:{value:raw('#2a3c44')}}]),vertexShader:groundVertex,fragmentShader:groundFragment,transparent:true,depthWrite:false,fog:true});
  const ground=new THREE.InstancedMesh(groundGeometry,groundMaterial,pieces);
  grid.blocks.forEach((block,i)=>{const pad=4.2,w=block.max[0]-block.min[0]+pad*2,d=block.max[1]-block.min[1]+pad*2;matrix.makeScale(w,1,d).setPosition((block.min[0]+block.max[0])/2,.02,(block.min[1]+block.max[1])/2);ground.setMatrixAt(i,matrix);kinds[i]=1;});
  grid.towers.forEach((tower,i)=>{matrix.makeScale(slab*1.5,1,slab*1.5).setPosition(tower.x+.6,.05,tower.z-.4);ground.setMatrixAt(grid.blocks.length+i,matrix);kinds[grid.blocks.length+i]=0;});
  ground.renderOrder=-5;ground.frustumCulled=false;group.add(ground);

  // Tiles for all 896 slots (graph, empty graph and failed attempt are told apart in the shader by status).
  const runs=[],graphSlots=[],failedSlots=[];
  for(let slot=0;slot<grid.total;slot++){if(statusOf(slot)==='failed')failedSlots.push(slot);else{graphSlots.push(slot);runs.push(runBySlot.get(slot));}}
  const maxTies=Math.max(1,...runs.map(run=>run?.edges.length||0));
  const tileGeometry=new THREE.BoxGeometry(1,1,1);
  const slotAttribute=new Float32Array(grid.total),tintAttribute=new Float32Array(grid.total);
  tileGeometry.setAttribute('aSlot',new THREE.InstancedBufferAttribute(slotAttribute,1));
  tileGeometry.setAttribute('aTint',new THREE.InstancedBufferAttribute(tintAttribute,1));
  const tileMaterial=new THREE.ShaderMaterial({
    uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uState:{value:null},uLightV:{value:new THREE.Vector3(0,1,0)},
      uLow:{value:raw('#163539')},uMid:{value:raw('#2a7a72')},uHigh:{value:raw('#86dcc3')},uEmpty:{value:raw(colors.empty||STAGE_COLORS.empty)},
      uFailed:{value:raw('#ff8a70')},uGold:{value:raw(STAGE_COLORS.gold)},uAccent:{value:raw('#e8fff7')},uBack:{value:raw('#0e1619')},uClose:{value:0}}]),
    vertexShader:tileVertex,fragmentShader:tileFragment,transparent:true,depthWrite:true,fog:true});
  tileMaterial.uniforms.uState.value=state.texture;
  const tiles=new THREE.InstancedMesh(tileGeometry,tileMaterial,grid.total);
  for(let slot=0;slot<grid.total;slot++){
    slotAttribute[slot]=slot;tintAttribute[slot]=tieTint(runBySlot.get(slot)?.edges.length||0,maxTies);
    matrix.makeScale(slab,thickness,slab).setPosition(grid.centers[slot*3],grid.centers[slot*3+1],grid.centers[slot*3+2]);
    tiles.setMatrixAt(slot,matrix);
  }
  tiles.renderOrder=5;tiles.frustumCulled=false;group.add(tiles);

  // Failed attempts: an extra dashed outline around the hatched hollow tile. No nodes, never a graph.
  const framePositions=[],frameSlots=[],frameDist=[];
  for(const slot of failedSlots){
    const [cx,cy,cz]=[grid.centers[slot*3],grid.centers[slot*3+1]+thickness/2+.02,grid.centers[slot*3+2]],h=slab/2+.35;
    const corners=[[-h,-h],[h,-h],[h,h],[-h,h]];
    for(let i=0;i<4;i++){const [ax,az]=corners[i],[bx,bz]=corners[(i+1)%4];framePositions.push(cx+ax,cy,cz+az,cx+bx,cy,cz+bz);frameSlots.push(slot,slot);frameDist.push(0,Math.hypot(bx-ax,bz-az));}
  }
  const frameGeometry=new THREE.BufferGeometry();
  frameGeometry.setAttribute('position',new THREE.Float32BufferAttribute(framePositions,3));
  frameGeometry.setAttribute('aSlot',new THREE.Float32BufferAttribute(frameSlots,1));
  frameGeometry.setAttribute('aDist',new THREE.Float32BufferAttribute(frameDist,1));
  const frameMaterial=new THREE.ShaderMaterial({uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uState:{value:null},uColor:{value:raw('#ff9a80')},uDash:{value:1.3}}]),vertexShader:frameVertex,fragmentShader:frameFragment,transparent:true,depthWrite:false,fog:true});
  frameMaterial.uniforms.uState.value=state.texture;
  const frames=new THREE.LineSegments(frameGeometry,frameMaterial);frames.renderOrder=6;frames.frustumCulled=false;group.add(frames);

  // Nodes: one point buffer for every persona on every graph slot (failed slots have none). Close range only.
  const nodeCount=runs.length*personas.length;
  const pointPositions=new Float32Array(nodeCount*3),pointSlot=new Float32Array(nodeCount),pointPerson=new Float32Array(nodeCount),pointSize=new Float32Array(nodeCount);
  const degree=new Uint16Array(personas.length);
  let edgeTotal=0;for(const run of runs)edgeTotal+=run.edges.length;
  const tiePositions=new Float32Array(edgeTotal*6),tieSlot=new Float32Array(edgeTotal*2),tieClass=new Float32Array(edgeTotal*2),tiePairs=new Uint8Array(edgeTotal*2);
  let n=0,e=0;
  const lift=thickness/2+.05;
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
      pointPositions.set([cx+plane[i][0],cy+.06,cz+plane[i][1]],n*3);
      pointSlot[n]=slot;pointPerson[n]=i;pointSize[n]=.16+.2*Math.sqrt(degree[i]/maxDegree);n++;
    }
  });
  const pointGeometry=new THREE.BufferGeometry();
  pointGeometry.setAttribute('position',new THREE.BufferAttribute(pointPositions,3));
  pointGeometry.setAttribute('aSlot',new THREE.BufferAttribute(pointSlot,1));
  pointGeometry.setAttribute('aPerson',new THREE.BufferAttribute(pointPerson,1));
  pointGeometry.setAttribute('aSize',new THREE.BufferAttribute(pointSize,1));
  const pointMaterial=new THREE.ShaderMaterial({uniforms:{uState:{value:state.texture},uPersona:{value:palette},uScale:{value:800},uMinPx:{value:1.5},uMaxPx:{value:11},uNear:{value:LOD.points[0]},uFar:{value:LOD.points[1]},uLightV:{value:new THREE.Vector3(0,0,1)},uBack:{value:raw('#0e1619')}},vertexShader:pointVertex,fragmentShader:pointFragment});
  const points=new THREE.Points(pointGeometry,pointMaterial);points.frustumCulled=false;points.renderOrder=3;points.visible=false;group.add(points);

  // Ties: one segment buffer with every primary tie, faded in by camera distance; out-of-scope towers lose theirs.
  const tieGeometry=new THREE.BufferGeometry();
  tieGeometry.setAttribute('position',new THREE.BufferAttribute(tiePositions,3));
  tieGeometry.setAttribute('aSlot',new THREE.BufferAttribute(tieSlot,1));
  const classAttribute=new THREE.BufferAttribute(tieClass,1);tieGeometry.setAttribute('aClass',classAttribute);
  const tieMaterial=new THREE.ShaderMaterial({uniforms:{uState:{value:state.texture},uTie:{value:raw(colors.tie||'#e2f4f7')},uSame:{value:raw(colors.same||'#80e2c4')},uMixed:{value:raw(colors.mixed||'#f2c078')},uLens:{value:0},uAlpha:{value:.3},uNear:{value:LOD.ties[0]},uFar:{value:LOD.ties[1]}},vertexShader:tieVertex,fragmentShader:tieFragment,transparent:true,depthWrite:false});
  const ties=new THREE.LineSegments(tieGeometry,tieMaterial);ties.frustumCulled=false;ties.renderOrder=4;ties.visible=false;group.add(ties);

  // Hover outline around one tower.
  const outline=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial({color:'#dffcf2',transparent:true,opacity:.85,depthWrite:false,fog:false}));
  outline.visible=false;outline.renderOrder=7;group.add(outline);
  stage.scene.add(group);

  let focus=null,hover=null,stacked=new Set(),opacity=.3;
  const unhook=stage.addHook(()=>{
    if(!group.visible)return;
    keyLightInView(stage.camera,lightV);
    pointMaterial.uniforms.uLightV.value.copy(lightV);tileMaterial.uniforms.uLightV.value.copy(lightV);
    pointMaterial.uniforms.uScale.value=stage.pixelScale();
    // Level of detail by distance from the camera to the nearest content (the orbit target is on the content).
    const distance=stage.camera.position.distanceTo(stage.controls.target);
    points.visible=distance<LOD.points[1]+10;ties.visible=distance<LOD.ties[1]+10;
    tileMaterial.uniforms.uClose.value=1-THREE.MathUtils.smoothstep(distance,LOD.ties[0]*.8,LOD.ties[1]*1.1);
    tieMaterial.uniforms.uAlpha.value=Math.min(1,opacity*3.3);
  });

  const setTowerFocus=(slot,value)=>{const base=Math.floor(slot/design.seeds.length)*design.seeds.length;for(let b=0;b<design.seeds.length;b++)state.set(base+b,1,value&&base+b===slot?255:value);};
  const api={
    group,layout:grid,
    get slots(){return grid.total;},
    get drawables(){return {slabs:graphSlots.length,failed:failedSlots.length,nodes:nodeCount,ties:edgeTotal};},
    get lod(){return {points:points.visible,ties:ties.visible};},
    // mask: Uint8Array(896), 1 = in filter scope. Out-of-scope towers recede to a dark ghost and lose their ties.
    setMatch(mask){for(let slot=0;slot<grid.total;slot++)state.set(slot,0,mask[slot]?255:0);state.commit();stage.requestRender();},
    setFocus(slot){
      if(focus!==null)setTowerFocus(focus,0);
      focus=Number.isInteger(slot)&&slot>=0&&slot<grid.total?slot:null;
      if(focus!==null)setTowerFocus(focus,128);
      state.commit();api.setHover(focus===null?null:Math.floor(focus/design.seeds.length));
    },
    setHover(towerIndex){
      hover=Number.isInteger(towerIndex)?grid.towers[towerIndex]:null;
      outline.visible=!!hover;
      if(hover){const [x0,y0,z0]=hover.min,[x1,y1,z1]=hover.max;outline.position.set((x0+x1)/2,(y0+y1)/2,(z0+z1)/2);outline.scale.set(x1-x0+1.2,y1-y0+.4,z1-z0+1.2);}
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
    // Close framing of one tower (used for the fly-in before the Stack opens): near enough that its ties show.
    towerFraming(index){
      const tower=grid.towers[index],cx=tower.x,cz=tower.z,cy=(tower.min[1]+tower.max[1])/2;
      const offset=new THREE.Vector3().subVectors(stage.camera.position,stage.controls.target).normalize().multiplyScalar(slab*4.2);
      return {position:new THREE.Vector3(cx,cy,cz).add(offset),target:new THREE.Vector3(cx,cy,cz)};
    },
    dispose(){unhook();stage.scene.remove(group);group.traverse(o=>{o.geometry?.dispose();o.material?.dispose?.();});state.dispose();palette.dispose();}
  };
  return api;
}
