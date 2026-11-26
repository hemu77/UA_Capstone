// Unit tests for the pure parts of the 3D engine (workstream 2). No DOM, no WebGL.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {atlasLayout,pickSlot,rayBox,personaPlanePositions,tieTint,ATLAS_GEOMETRY} from './layers-3d-atlas.js';
import {stackGeometry,identityThread,keyOf,dashSegments,spreadLayout,STACK_CAPACITY} from './layers-3d-stack.js';
import {cullBoxes} from './layers-3d-labels.js';
import {framing,easeInOutCubic,POSES,boxCorners} from './layers-3d-scene.js';
import {MAX_STACK_LAYERS} from './layers-app.js';

const design={models:['gpt-4.1','gpt-5.6-luna','gpt-6-luna','gpt-6-sol'],methods:['global','local','sequential','iterative'],
  settings:[['us','english'],['india','english'],['japan','english'],['brazil','english'],['us','hindi'],['us','japanese'],['us','portuguese']],
  seeds:[44000,44001,44002,44003,44004,44005,44006,44007]};
const slotOf=(m,p,s,b)=>((m*4+p)*7+s)*8+b;

test('atlas lays out all 896 slots as 112 towers of 8 seed planes in design order',()=>{
  const layout=atlasLayout(design);
  assert.equal(layout.total,896);assert.equal(layout.towers.length,112);assert.equal(layout.columns,16);
  assert.equal(layout.centers.length,896*3);
  assert.ok(layout.centers.every(Number.isFinite));
  for(const tower of layout.towers){
    assert.equal(tower.slots.length,8);
    // Height is seed order, strictly increasing upward, identical spacing in every tower.
    const ys=tower.slots.map(slot=>layout.centers[slot*3+1]);
    ys.forEach((y,i)=>i&&assert.ok(Math.abs(y-ys[i-1]-ATLAS_GEOMETRY.pitchY)<1e-9));
    for(const slot of tower.slots){assert.equal(layout.centers[slot*3],tower.x);assert.equal(layout.centers[slot*3+2],tower.z);}
  }
  // The slot formula of the data contract addresses the right tower and plane.
  const slot=slotOf(2,3,4,5),tower=layout.towers[Math.floor(slot/8)];
  assert.deepEqual([tower.model,tower.method,tower.setting],['gpt-6-luna','iterative',['us','hindi']]);
  assert.equal(tower.slots[5],slot);
  // No two slots share a position.
  const keys=new Set();for(let s=0;s<896;s++)keys.add([0,1,2].map(k=>layout.centers[s*3+k].toFixed(3)).join());
  assert.equal(keys.size,896);
  // Columns run model-major then method along X; settings recede along Z; model groups are separated by a gap.
  assert.ok(layout.towers[slotOf(0,1,0,0)/8].x>layout.towers[slotOf(0,0,0,0)/8].x);
  assert.ok(layout.towers[slotOf(0,0,1,0)/8].z<layout.towers[slotOf(0,0,0,0)/8].z);
  const step=layout.towers[slotOf(0,1,0,0)/8].x-layout.towers[slotOf(0,0,0,0)/8].x,gap=layout.towers[slotOf(1,0,0,0)/8].x-layout.towers[slotOf(0,3,0,0)/8].x;
  assert.ok(gap>step);
  // Axis anchors: 4 models, 16 method columns, 7 settings, 8 seeds and one honest seed-axis caption.
  const kinds=layout.axes.reduce((map,a)=>(map[a.kind]=(map[a.kind]||0)+1,map),{});
  assert.deepEqual(kinds,{model:4,method:16,setting:7,seed:8,'seed-axis':1});
  assert.ok(layout.bounds.radius>0&&layout.bounds.width>layout.bounds.depth);
});

test('atlas picking finds the tower and seed plane a ray meets first',()=>{
  const layout=atlasLayout(design),target=slotOf(1,2,3,6),[x,y,z]=[0,1,2].map(k=>layout.centers[target*3+k]);
  // Straight down onto a tower: the top plane is met first.
  const down=pickSlot(layout,[x,200,z],[0,-1,0]);
  assert.equal(down.tower,Math.floor(target/8));assert.equal(down.slot,Math.floor(target/8)*8+7);
  // A ray aimed just above the target plane from the side hits that tower; the plane closest to the entry height wins.
  const side=pickSlot(layout,[x-400,y+.4,z],[1,0,0]);
  assert.equal(side.tower,layout.towers.findIndex(t=>t.z===z&&t.x===Math.min(...layout.towers.filter(u=>u.z===z).map(u=>u.x))));
  // Missing everything returns null.
  assert.equal(pickSlot(layout,[0,500,0],[0,1,0]),null);
  assert.equal(rayBox([0,0,-5],[0,0,1],[-1,-1,-1],[1,1,1]),4);
  assert.equal(rayBox([5,5,5],[0,0,1],[-1,-1,-1],[1,1,1]),Infinity);
});

test('persona plane positions are normalised and accept the compact or Map layouts',()=>{
  const personas=[{id:'0',position:[75,0,0]},{id:'1',position:[0,0,-30]},{id:'2',position:[-10,0,10]}];
  const fromXy=personaPlanePositions(personas,{xy:[2,0,0,-1,-1,1]});
  assert.deepEqual(fromXy[0],[1,0]);assert.ok(fromXy.every(([a,b])=>Math.hypot(a,b)<=1+1e-12));
  const fromMap=personaPlanePositions(personas,new Map([['0',[4,0,9]],['1',[0,2,0]],['2',[-2,0,0]]]));
  assert.deepEqual(fromMap[0],[1,0]);
  const fallback=personaPlanePositions(personas,null);assert.deepEqual(fallback[0],[1,0]);
});

const personas=['0','1','2','3'];
const layout=new Map([['0',[0,0]],['1',[10,0]],['2',[0,10]],['3',[10,10]]]);
const runs=[{run_id:'a',edges:[['0','1'],['1','2'],['2','3']]},{run_id:'b',edges:[['0','1'],['1','2']]},{run_id:'c',edges:[['0','1'],['0','3']]}];

test('stack geometry: planes, degree-scaled nodes, in-plane ties, shared counts and ribbons',()=>{
  const g=stackGeometry(runs,layout,{personas,spacing:40,scale:2});
  assert.equal(g.layers,3);assert.deepEqual([...g.planeY],[-40,0,40]);
  assert.equal(g.nodes.count,12);assert.equal(g.ties.count,7);
  assert.deepEqual(g.counts,{shared:1,union:4,ribbons:3}); // 0-1 everywhere; 1-2 in a and b
  // Ties lie exactly in their plane; positions use the shared layout times scale.
  for(let s=0;s<g.ties.count;s++){assert.equal(g.ties.positions[s*6+1],g.planeY[g.ties.layer[s]]);assert.equal(g.ties.positions[s*6+4],g.planeY[g.ties.layer[s]]);}
  assert.deepEqual([...g.nodes.feet.subarray(3,6)],[20,-40,0]);
  // Radius grows with degree; flat encoding puts every node centre one radius above its plane.
  const N=4,r=k=>g.nodes.radius[k];
  assert.ok(r(1)>r(0)&&r(0)>=1.7);
  for(let k=0;k<12;k++)assert.ok(Math.abs(g.nodes.positions[k*3+1]-g.planeY[Math.floor(k/N)]-r(k))<1e-4);
  // Lift by degree raises nodes in proportion to degree; isolated nodes stay on the plane.
  const lifted=stackGeometry(runs,layout,{personas,spacing:40,lift:true,liftMax:20});
  assert.ok(lifted.nodes.positions[1*3+1]>g.nodes.positions[1*3+1]);
  const isolated=2*N+2; // persona 2 has no tie in run c
  assert.equal(lifted.nodes.positions[isolated*3+1],g.nodes.positions[isolated*3+1]);
  // Ribbons connect the same tie in adjacent planes only (a-b: 0-1, 1-2; b-c: 0-1).
  assert.equal(g.ribbons.count,3);assert.deepEqual([...g.ribbons.layer],[0,0,1]);
  assert.deepEqual([...g.ribbons.t.subarray(0,6)],[0,0,1,0,1,1]);
});

test('stack tie filters, persona emphasis and isolation',()=>{
  const shared=stackGeometry(runs,layout,{personas,ties:'shared'});
  assert.equal(shared.ties.count,3);assert.ok([...shared.ties.presence.subarray(0,3)].every(c=>c===3));
  const different=stackGeometry(runs,layout,{personas,ties:'different'});
  assert.equal(different.ties.count,4);
  const persona=stackGeometry(runs,layout,{personas,persona:'2'});
  const kinds=[...persona.ties.kind.subarray(0,persona.ties.count)];
  assert.equal(kinds.filter(k=>k===2).length,3);assert.equal(kinds.filter(k=>k===1).length,4);
  const isolated=stackGeometry(runs,layout,{personas,persona:'2',incident:true});
  assert.equal(isolated.ties.count,3);
  // Only persona 2 and its neighbours stay visible; in run c persona 2 has no tie and stands alone.
  assert.deepEqual([...isolated.nodes.visible],[0,1,1,1, 0,1,1,0, 0,0,1,0]);
  // Replay frames: the active plane can use frame edges instead of the final graph.
  const frame=stackGeometry([runs[0]],layout,{personas,edgesOf:()=>[['0','3']]});
  assert.equal(frame.ties.count,1);assert.equal(frame.nodes.degree[0],1);
  assert.throws(()=>stackGeometry([{run_id:'x',edges:[['0','9']]}],layout,{personas}),/Invalid tie endpoint/);
});

test('identity thread joins one persona across every plane, never as a tie',()=>{
  const g=stackGeometry(runs,layout,{personas,spacing:30});
  const thread=identityThread(g,'3');
  assert.equal(thread.length,12);
  assert.equal(thread[1],g.nodes.positions[(0*4+3)*3+1]);assert.equal(thread[10],g.nodes.positions[(2*4+3)*3+1]);
  assert.equal(identityThread(stackGeometry([runs[0]],layout,{personas}),'3').length,0);
  assert.equal(keyOf('10','2'),'2|10');assert.equal(keyOf('2','10'),'2|10');
  assert.ok(STACK_CAPACITY>=MAX_STACK_LAYERS+1);
  // Geometry dashes: pieces of `dash` length separated by `gap`, all on the segment, never beyond its end.
  const pieces=dashSegments([0,0,0],[0,10,0],2,1);
  assert.equal(pieces.length/6,4);assert.deepEqual(pieces.slice(0,6),[0,0,0,0,2,0]);assert.equal(pieces.at(-2),10);
  assert.deepEqual(dashSegments([1,1,1],[1,1,1]),[]);
});

test('label culling keeps high priority labels and hides overlapping low ones',()=>{
  const boxes=[{id:'a',x:0,y:0,w:50,h:10,priority:0,order:0},{id:'b',x:10,y:2,w:50,h:10,priority:3,order:1},{id:'c',x:100,y:0,w:20,h:10,priority:0,order:2},
    {id:'d',x:12,y:4,w:5,h:5,priority:0,order:3,cull:false},{id:'e',x:500,y:0,w:5,h:5,priority:9,order:4,offscreen:true}];
  assert.deepEqual([...cullBoxes(boxes)].sort(),['b','c','d']);
});

test('camera framing fits the content and easing is monotone',()=>{
  const wide=framing({center:[0,0,0],radius:100,direction:POSES.orbit,fov:38,aspect:1.6});
  const narrow=framing({center:[0,0,0],radius:100,direction:POSES.orbit,fov:38,aspect:.5});
  assert.ok(narrow.distance>wide.distance);
  assert.ok(Math.abs(Math.hypot(...wide.position)-wide.distance)<1e-6);
  const covered=framing({center:[0,0,0],radius:100,direction:POSES.orbit,fov:38,aspect:1.6,usable:[1,.6]});
  assert.ok(covered.distance>wide.distance,'chrome over the canvas pushes the camera back');
  const box=framing({center:[0,0,0],radius:100,extent:[100,20],direction:[0,.5,1],fov:26,aspect:1.6});
  assert.ok(box.distance<framing({center:[0,0,0],radius:100,direction:[0,.5,1],fov:26,aspect:1.6}).distance+1e-9);
  let previous=-1;for(let t=0;t<=1;t+=.05){const v=easeInOutCubic(t);assert.ok(v>=previous);previous=v;}
  assert.equal(easeInOutCubic(0),0);assert.equal(easeInOutCubic(1),1);
});

test('layout spread fills the unit disc and keeps direction and distance order',()=>{
  const raw=new Map([['a',[1,0]],['b',[0,2]],['c',[-.2,0]],['d',[0,-.1]],['e',[30,30]]]);
  const out=spreadLayout(raw);
  assert.equal(out.size,5);
  const cx=(1+0-.2+0+30)/5,cz=(0+2+0-.1+30)/5,radius=id=>Math.hypot(...out.get(id));
  for(const [id,[x,z]] of raw){
    const [u,v]=out.get(id);
    assert.ok(Math.abs(Math.atan2(v,u)-Math.atan2(z-cz,x-cx))<1e-9,'direction from the centre is kept');
    assert.ok(radius(id)<=1+1e-12);
  }
  assert.ok(Math.abs(Math.max(...[...out.keys()].map(radius))-1)<1e-12,'the outermost persona reaches the rim');
  const byRaw=[...raw].sort((p,q)=>Math.hypot(p[1][0]-cx,p[1][1]-cz)-Math.hypot(q[1][0]-cx,q[1][1]-cz)).map(([id])=>id);
  byRaw.forEach((id,i)=>i&&assert.ok(radius(id)>=radius(byRaw[i-1]),'distance order is kept'));
  // A far outlier no longer squeezes everyone else into the centre.
  assert.ok(radius(byRaw[1])>.3);
  assert.equal(spreadLayout(new Map()).size,0);
});

test('atlas tile tint grows with tie count and is 0 for an empty graph',()=>{
  assert.equal(tieTint(0,279),0);assert.equal(tieTint(279,279),1);
  assert.ok(tieTint(50,279)<tieTint(51,279));assert.equal(tieTint(400,279),1);assert.equal(tieTint(5,0),0);
});

test('point framing keeps every corner inside the frustum, near corners included',()=>{
  const center=[0,5,0],points=boxCorners([-90,0,-60],[90,20,70],center);
  assert.equal(points.length,8);
  const fov=22,aspect=1.7,usable=[.8,.7],direction=[0,1.55,1];
  const fit=framing({center,points,direction,fov,aspect,usable,margin:1});
  // Project with the same camera basis and check every corner lands inside the usable frame.
  const len=Math.hypot(...direction),v=direction.map(c=>c/len),right=[v[2],0,-v[0]].map((c,_,a)=>c/Math.hypot(...a)||0);
  const up=[v[1]*right[2]-v[2]*right[1],v[2]*right[0]-v[0]*right[2],v[0]*right[1]-v[1]*right[0]];
  const tanV=Math.tan(fov*Math.PI/360)*usable[1],tanH=Math.tan(fov*Math.PI/360)*aspect*usable[0];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  let tight=0;
  for(const p of points){
    const depth=fit.distance-dot(p,v);
    const x=Math.abs(dot(p,right))/depth/tanH,y=Math.abs(dot(p,up))/depth/tanV;
    assert.ok(x<=1+1e-9&&y<=1+1e-9);tight=Math.max(tight,x,y);
  }
  assert.ok(tight>.999,'the fit is tight, not merely safe');
});
