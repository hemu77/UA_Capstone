// WORKSTREAM 2 (3D engine). Shared stage for the Atlas, Stack and Replay views (SPEC.md B.3, C.4).
// Renders on demand only: a frame is drawn when something asked for one (input, damping, a camera move,
// an animation). When nothing moves, no frame is requested and the GPU is idle.
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

// Canvas-matched colours (sRGB hex). Kept here so every view shares one depth language.
export const STAGE_COLORS=Object.freeze({fog:'#0d1519',ground:'#2a3a42',groundMajor:'#3b5560',key:'#fff6ea',sky:'#cfe6f2',earth:'#0a1013',accent:'#7fe3c4',failed:'#c09088',empty:'#8aa0aa',gold:'#f0d59b',remove:'#ff9a79',actor:'#efbc7e',tie:'#7fb0c0'});
export const KEY_LIGHT_DIRECTION=Object.freeze([-.45,.82,.36]);
// Raw sRGB triple for custom shaders that write gl_FragColor without colour-space conversion.
export const raw=hex=>new THREE.Color().setStyle(hex,THREE.LinearSRGBColorSpace);
export const easeInOutCubic=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;

// Pure: camera placement that fits a sphere of `radius` around `center` for a viewing direction.
// With `extent` ([half width, half height] in view space) the fit uses that box instead of a bounding sphere.
// With `points` (world offsets from `center`, e.g. the corners of the content box) the fit is exact under
// perspective: every point lands inside the frame, near points included.
// `usable` ([width, height] fractions of the frame not covered by page chrome) shrinks the fit accordingly.
export function framing({center=[0,0,0],radius=1,extent=null,points=null,direction=[.55,.62,.85],fov=38,aspect=1.6,margin=1.08,usable=[1,1]}) {
  const v=new THREE.Vector3(...direction).normalize(),tanHalf=Math.tan(THREE.MathUtils.degToRad(fov/2));
  const tanV=tanHalf*usable[1],tanH=tanHalf*aspect*usable[0];
  const limit=Math.min(Math.atan(tanV),Math.atan(tanH));
  let distance;
  if(points?.length){
    const up=Math.abs(v.y)>.999?new THREE.Vector3(0,0,-1):new THREE.Vector3(0,1,0);
    const right=new THREE.Vector3().crossVectors(up,v).normalize(),upV=new THREE.Vector3().crossVectors(v,right).normalize(),p=new THREE.Vector3();
    distance=0;
    for(const point of points){p.set(...point);const x=Math.abs(p.dot(right))*margin,y=Math.abs(p.dot(upV))*margin,z=p.dot(v);distance=Math.max(distance,x/tanH+z,y/tanV+z);}
  }else distance=extent?Math.max(extent[0]/tanH,extent[1]/tanV)*margin+radius*.15:radius*margin/Math.sin(limit);
  return {position:v.multiplyScalar(distance).add(new THREE.Vector3(...center)).toArray(),target:[...center],distance};
}
// Pure: the 8 corners of an axis-aligned box as offsets from `center` (input for framing({points})).
export const boxCorners=(min,max,center=[0,0,0])=>{const out=[];for(const x of [min[0],max[0]])for(const y of [min[1],max[1]])for(const z of [min[2],max[2]])out.push([x-center[0],y-center[1],z-center[2]]);return out;};
export const POSES=Object.freeze({orbit:[.55,.5,.95],top:[0,1,.035],front:[0,.16,1]});

const groundVertex=`
varying vec3 vWorld;
#include <fog_pars_vertex>
void main(){
  vec4 world=modelMatrix*vec4(position,1.);vWorld=world.xyz;
  vec4 mvPosition=viewMatrix*world;gl_Position=projectionMatrix*mvPosition;
  #include <fog_vertex>
}`;
const groundFragment=`
uniform vec3 uMinor;uniform vec3 uMajor;uniform float uCell;uniform float uRadius;uniform vec2 uCenter;
uniform vec4 uShadow;uniform float uShadowSoft;uniform float uShadowStrength;
varying vec3 vWorld;
#include <fog_pars_fragment>
float gridLine(vec2 p,float cell){vec2 g=abs(fract(p/cell-.5)-.5)/fwidth(p/cell);return 1.-min(min(g.x,g.y),1.);}
void main(){
  vec2 p=vWorld.xz;
  float minor=gridLine(p,uCell),major=gridLine(p,uCell*5.);
  float d=length(p-uCenter)/uRadius;
  float fade=1.-smoothstep(.35,1.,d);
  vec3 col=mix(uMinor,uMajor,major);
  float a=(minor*.28+major*.5)*fade;
  // Soft contact shadow under the content footprint (a rounded box, blurred by distance).
  vec2 q=abs(p-uShadow.xy)-uShadow.zw;
  float box=length(max(q,0.))+min(max(q.x,q.y),0.);
  float shade=(1.-smoothstep(-uShadowSoft*.4,uShadowSoft,box))*uShadowStrength*fade;
  gl_FragColor=vec4(mix(col,vec3(0.),shade),max(a,shade));
  #include <fog_fragment>
}`;

export function createStage3D(host,{labels=null,reducedMotion=false,onRender=null}={}) {
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
  if(!renderer.getContext())throw new Error('WebGL unavailable');
  renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio||1,2));
  renderer.setClearColor(0x000000,0);
  renderer.info.autoReset=true;
  const canvas=renderer.domElement;host.append(canvas);
  const scene=new THREE.Scene();
  scene.fog=new THREE.FogExp2(STAGE_COLORS.fog,.0016);
  const camera=new THREE.PerspectiveCamera(38,1,.5,6000);
  camera.position.set(300,260,420);
  const controls=new OrbitControls(camera,canvas);
  controls.enableDamping=!reducedMotion;controls.dampingFactor=.09;controls.autoRotateSpeed=.35;
  // Lighting: cool sky / dark ground hemisphere plus one warm key and a faint cool rim from behind.
  const hemi=new THREE.HemisphereLight(STAGE_COLORS.sky,STAGE_COLORS.earth,1.25);
  const key=new THREE.DirectionalLight(STAGE_COLORS.key,2.1);key.position.set(...KEY_LIGHT_DIRECTION).multiplyScalar(500);
  const rim=new THREE.DirectionalLight('#8fc8ff',.7);rim.position.set(.4,.3,-1).multiplyScalar(500);
  scene.add(hemi,key,rim);
  // Ground grid with horizon fade and a soft footprint shadow. One draw call.
  const groundUniforms=THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{
    uMinor:{value:raw(STAGE_COLORS.ground)},uMajor:{value:raw(STAGE_COLORS.groundMajor)},
    uCell:{value:10},uRadius:{value:400},uCenter:{value:new THREE.Vector2()},
    uShadow:{value:new THREE.Vector4(0,0,0,0)},uShadowSoft:{value:30},uShadowStrength:{value:0}}]);
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.ShaderMaterial({uniforms:groundUniforms,vertexShader:groundVertex,fragmentShader:groundFragment,transparent:true,depthWrite:false,fog:true}));
  ground.rotation.x=-Math.PI/2;ground.renderOrder=-10;ground.frustumCulled=false;scene.add(ground);

  const size={width:1,height:1,offsetX:0,offsetY:0};
  const hooks=new Set();
  let scheduled=false,disposed=false,tween=null,bounds={center:[0,0,0],radius:100},lastPose='orbit',userMoved=false,raf=0;
  // Page chrome drawn over the canvas (heading, legend). Framing centres content in the free area: the projection
  // centre is shifted with a view offset, and the fit uses only the unobstructed fraction of the frame.
  const safe={top:0,bottom:0};
  // A view may also reserve screen space beside the content for its own labels (bounds.padPx = [left, right] px).
  // Both are applied with a view offset, so the camera still looks straight at the content centre and perspective
  // stays symmetric (no skew from aiming the camera off-centre).
  function shifts(){
    const [left,right]=bounds.padPx||[0,0];
    return {x:Math.round(left-right),y:Math.round(safe.bottom-safe.top),freeX:Math.max(.35,(size.width-left-right)/(size.width+Math.abs(left-right)))};
  }
  function applyProjection(){
    const {width,height}=size,{x,y}=shifts(),fullW=width+Math.abs(x),fullH=height+Math.abs(y);
    camera.aspect=fullW/fullH;
    if(x||y)camera.setViewOffset(fullW,fullH,x<0?-x:0,y>0?y:0,width,height);else camera.clearViewOffset();
    camera.updateProjectionMatrix();
  }
  const usable=()=>{const full=size.height+Math.abs(safe.bottom-safe.top);return [shifts().freeX,Math.max(.35,(size.height-safe.top-safe.bottom)/full)];};
  const lastMatrix=new THREE.Matrix4(),lastProjection=new THREE.Matrix4();
  const frameStats={calls:0,triangles:0,points:0,lines:0,frames:0};

  function requestRender(){if(!scheduled&&!disposed){scheduled=true;raf=requestAnimationFrame(frame);}}
  function frame(now){
    scheduled=false;if(disposed)return;
    let again=false;
    if(tween){
      const t=Math.min(1,(now-tween.start)/tween.duration),k=easeInOutCubic(t);
      camera.position.lerpVectors(tween.from.position,tween.to.position,k);
      controls.target.lerpVectors(tween.from.target,tween.to.target,k);
      if(t>=1){const done=tween.resolve;tween=null;done?.();}else again=true;
    }
    for(const hook of hooks)if(hook(now)===true)again=true;
    if(controls.update())again=true;
    if(controls.autoRotate)again=true;
    camera.updateMatrixWorld();
    onRender?.(camera,size);
    renderer.render(scene,camera);
    const info=renderer.info.render;
    frameStats.calls=info.calls;frameStats.triangles=info.triangles;frameStats.points=info.points;frameStats.lines=info.lines;frameStats.frames++;
    canvas.dataset.drawCalls=String(info.calls);canvas.dataset.frames=String(frameStats.frames);
    // Camera pose for checks (eased moves, reduced motion): position | target | fov, rounded.
    const r=v=>v.toArray().map(n=>Math.round(n*10)/10).join(',');
    canvas.dataset.camera=`${r(camera.position)}|${r(controls.target)}|${camera.fov}`;
    const moved=!lastMatrix.equals(camera.matrixWorld)||!lastProjection.equals(camera.projectionMatrix);
    if(labels&&(moved||labels.dirty))labels.project(camera,size);
    lastMatrix.copy(camera.matrixWorld);lastProjection.copy(camera.projectionMatrix);
    if(again)requestRender();
  }
  controls.addEventListener('change',requestRender);
  controls.addEventListener('start',()=>{userMoved=true;if(tween){const done=tween.resolve;tween=null;done?.();}});

  function resize(width,height,offsetX=0,offsetY=0){
    width=Math.max(1,Math.round(width));height=Math.max(1,Math.round(height));
    const changed=width!==size.width||height!==size.height;
    Object.assign(size,{width,height,offsetX,offsetY});
    if(changed){renderer.setSize(width,height,false);canvas.style.width='100%';canvas.style.height='100%';applyProjection();}
    labels?.markDirty();
    for(const listener of resizeListeners)listener(size);
    // Until the user takes the camera, keep the current pose framed for the new size (retargets a running move).
    if(changed&&!userMoved)pose(lastPose,{instant:!tween});
    requestRender();
  }
  const resizeListeners=new Set();
  const observer=new ResizeObserver(entries=>{
    const rect=entries.at(-1).contentRect;
    resize(rect.width,rect.height,size.offsetX,size.offsetY);
  });
  observer.observe(host);

  function flyTo(target,{duration=520,instant=false}={}){
    const to={position:new THREE.Vector3(...(target.position.toArray?target.position.toArray():target.position)),target:new THREE.Vector3(...(target.target.toArray?target.target.toArray():target.target))};
    if(tween){const done=tween.resolve;tween=null;done?.();}
    if(instant||reducedMotion||duration<=0){camera.position.copy(to.position);controls.target.copy(to.target);controls.update();requestRender();return Promise.resolve();}
    return new Promise(resolve=>{
      tween={start:performance.now(),duration,from:{position:camera.position.clone(),target:controls.target.clone()},to,resolve};
      requestRender();
    });
  }
  function poseTarget(name){
    const direction=Array.isArray(name)?name:bounds.poses?.[name]||POSES[name]||POSES.orbit;
    return framing({center:bounds.center,radius:bounds.radius,extent:name==='orbit'||name==='front'?bounds.extent:null,points:bounds.points||null,direction,fov:camera.fov,aspect:camera.aspect,margin:bounds.margin||1.05,usable:usable()});
  }
  function pose(name='orbit',options={}){
    if(typeof name==='string')lastPose=name;
    userMoved=false;
    const target=poseTarget(name);
    // Depth fog scales with the framing distance, so a tall stack framed from far away is not drowned in fog while
    // its far planes still recede behind the near ones.
    if(bounds.fogScale)scene.fog.density=bounds.fogScale/Math.max(1,target.distance);
    return flyTo(target,options);
  }
  return {
    scene,camera,controls,renderer,canvas,size,ground,
    requestRender,flyTo,pose,poseTarget,
    get reducedMotion(){return reducedMotion;},
    setReducedMotion(value){reducedMotion=!!value;controls.enableDamping=!reducedMotion;if(reducedMotion)controls.autoRotate=false;},
    // A view declares what the camera frames and how the ground and fog are tuned for its scale.
    setBounds(next){bounds={...bounds,padPx:null,points:null,extent:null,...next};if(next.fov)camera.fov=next.fov;applyProjection();},
    // Pixels of the canvas covered by page chrome at the top and bottom (measured by the page, never per frame).
    setSafeArea({top=0,bottom=0}={}){
      top=Math.max(0,Math.round(top));bottom=Math.max(0,Math.round(bottom));
      if(top===safe.top&&bottom===safe.bottom)return;
      safe.top=top;safe.bottom=bottom;applyProjection();labels?.markDirty();
      if(!userMoved)pose(lastPose,{instant:!tween});
      requestRender();
    },
    get bounds(){return bounds;},
    setEnvironment({fogDensity,groundY=0,groundRadius=400,cell=10,center=[0,0],shadow=null}={}){
      if(fogDensity!==undefined)scene.fog.density=fogDensity;
      ground.position.set(center[0],groundY,center[1]);ground.scale.set(groundRadius*2.2,groundRadius*2.2,1);
      const u=ground.material.uniforms;u.uCell.value=cell;u.uRadius.value=groundRadius;u.uCenter.value.set(center[0],center[1]);
      if(shadow){u.uShadow.value.set(shadow.x,shadow.z,shadow.halfX,shadow.halfZ);u.uShadowSoft.value=shadow.soft;u.uShadowStrength.value=shadow.strength;}else u.uShadowStrength.value=0;
      requestRender();
    },
    setOffset(offsetX,offsetY){if(offsetX!==size.offsetX||offsetY!==size.offsetY){size.offsetX=offsetX;size.offsetY=offsetY;labels?.markDirty();requestRender();}},
    onResize(listener){resizeListeners.add(listener);return()=>resizeListeners.delete(listener);},
    // hook(now) runs before each render; return true to request another frame (an animation in progress).
    addHook(hook){hooks.add(hook);return()=>hooks.delete(hook);},
    get userMoved(){return userMoved;},
    markUserMoved(){userMoved=true;},
    setAutoRotate(on){controls.autoRotate=!!on&&!reducedMotion;requestRender();return controls.autoRotate;},
    // Pixel size of one world unit at distance 1 (for size attenuation with a legibility floor).
    pixelScale(){return (size.height+Math.abs(safe.bottom-safe.top))/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)));},
    stats(){return {...frameStats,geometries:renderer.info.memory.geometries,textures:renderer.info.memory.textures};},
    renderNow(){if(scheduled){cancelAnimationFrame(raf);scheduled=false;}frame(performance.now());},
    resize,
    dispose(){disposed=true;cancelAnimationFrame(raf);observer.disconnect();controls.dispose();ground.geometry.dispose();ground.material.dispose();renderer.dispose();canvas.remove();}
  };
}

// ---------------------------------------------------------------------------------------------
// Shared materials. Custom shaders write raw sRGB (see raw()) and apply scene fog themselves.

// Per-slot state texture (one texel per slot or layer): r = in filter scope, g = focus, b = in stack, a = status
// (0 valid, 128 valid empty graph, 255 failed attempt). Updating it never touches geometry.
export function createStateTexture(count) {
  const width=Math.max(1,count),data=new Uint8Array(width*4);
  const texture=new THREE.DataTexture(data,width,1,THREE.RGBAFormat,THREE.UnsignedByteType);
  texture.magFilter=THREE.NearestFilter;texture.minFilter=THREE.NearestFilter;texture.generateMipmaps=false;texture.needsUpdate=true;
  return {texture,data,width,set(index,channel,value){data[index*4+channel]=value;},commit(){texture.needsUpdate=true;},dispose(){texture.dispose();}};
}

const slabVertex=`
attribute float aSlot;
uniform sampler2D uState;
varying vec3 vLocal;varying vec3 vNormalV;varying vec3 vRim;varying vec4 vState;varying vec3 vView;
#include <fog_pars_vertex>
void main(){
  vLocal=position;
  vState=texelFetch(uState,ivec2(int(aSlot+.5),0),0);
  #ifdef USE_INSTANCING_COLOR
    vRim=instanceColor;
  #else
    vRim=vec3(1.);
  #endif
  mat4 m=modelMatrix*instanceMatrix;
  vec4 mvPosition=viewMatrix*m*vec4(position,1.);
  vNormalV=normalize(mat3(viewMatrix)*mat3(m)*normal);
  vView=-mvPosition.xyz;
  gl_Position=projectionMatrix*mvPosition;
  #include <fog_vertex>
}`;
const slabFragment=`
uniform vec3 uLightV;uniform float uFill;uniform float uDim;uniform float uGrid;uniform vec3 uGold;uniform float uRimWidth;uniform float uRimAlpha;
varying vec3 vLocal;varying vec3 vNormalV;varying vec3 vRim;varying vec4 vState;varying vec3 vView;
#include <fog_pars_fragment>
void main(){
  float match=vState.r,focus=vState.g,stacked=vState.b;
  vec3 n=normalize(vNormalV);
  float isTop=step(.499,vLocal.y),isBottom=step(vLocal.y,-.499),side=1.-isTop-isBottom;
  vec2 uv=vLocal.xz+.5;
  #ifdef DISC
    float edge=.5-length(vLocal.xz);
  #else
    float edge=min(min(uv.x,1.-uv.x),min(uv.y,1.-uv.y));
  #endif
  float px=edge/max(fwidth(edge),1e-6);
  float rim=(1.-smoothstep(uRimWidth*.5,uRimWidth*1.5,px))*isTop;
  float glow=exp(-px*.35)*isTop;
  vec2 gp=uv*uGrid,gw=max(fwidth(gp),vec2(1e-6));vec2 g=abs(fract(gp-.5)-.5)/gw;
  // Fade the inner grid out before its cells get smaller than ~12 px, so distant slabs stay calm.
  float grid=(1.-min(min(g.x,g.y),1.))*isTop*(1.-rim)*(1.-smoothstep(.05,.09,max(gw.x,gw.y)));
  vec3 hue=mix(vRim,uGold,stacked*.8);
  float diff=.42+.58*max(dot(n,uLightV),0.);
  float fres=pow(1.-abs(dot(n,normalize(vView))),2.);
  vec3 col=hue*.55*diff*(.7+.6*side)+hue*(rim*1.15+glow*.25)+vec3(.78,.9,1.)*grid*.16+hue*fres*.2*side;
  float a=isTop*uFill+side*uFill*3.+isBottom*uFill*.5+rim*uRimAlpha+glow*.12+grid*.14;
  a*=mix(uDim,1.,match);col*=mix(.5,1.,match);
  col+=focus*hue*.45;a=min(1.,a+focus*.2);
  gl_FragColor=vec4(col,clamp(a,0.,1.));
  #include <fog_fragment>
}`;
// Glass slab: thin lit box (or disc, with shape:'disc' on a unit cylinder) with a bright rim in the given instance
// colour and a faint inner grid.
export function createSlabMaterial(state,{fill=.07,dim=.12,grid=8,rimWidth=2,rimAlpha=.9,shape='box'}={}) {
  const material=new THREE.ShaderMaterial({
    uniforms:THREE.UniformsUtils.merge([THREE.UniformsLib.fog,{uState:{value:null},uLightV:{value:new THREE.Vector3(0,1,0)},uFill:{value:fill},uDim:{value:dim},uGrid:{value:grid},uGold:{value:raw(STAGE_COLORS.gold)},uRimWidth:{value:rimWidth},uRimAlpha:{value:rimAlpha}}]),
    vertexShader:slabVertex,fragmentShader:slabFragment,transparent:true,depthWrite:false,fog:true,defines:shape==='disc'?{DISC:''}:{}
  });
  // Assigned after merge(): merge() clones textures, and the clone would not see in-place updates.
  material.uniforms.uState.value=state.texture;
  return material;
}
// View-space direction of the key light, for shaders that light sprites or slabs themselves.
const lightWorld=new THREE.Vector3();
export function keyLightInView(camera,target) {
  lightWorld.set(...KEY_LIGHT_DIRECTION).normalize();
  return target.copy(lightWorld).transformDirection(camera.matrixWorldInverse);
}
