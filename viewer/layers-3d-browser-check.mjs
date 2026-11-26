// Workstream 2 browser check: the 3D engine on the built page (SPEC.md B.3, C.4, D "All 890 in 3D" / "3D quality").
// Run against a static build:  BROWSER_CHANNEL=chromium VIEWER_URL=http://127.0.0.1:8781 node layers-3d-browser-check.mjs
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge',args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const canvasData=page=>page.evaluate(()=>({...document.querySelector('#viewport canvas').dataset}));
const frames=page=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))));
// The page's own store instance: import the exact module URL the page loaded (same URL = same instance).
const store=(page,fn,arg)=>page.evaluate(async([source,arg])=>{const {app}=await import(document.querySelector('script[src*="layers.js"]').src);return new Function('app','arg',`return (${source})(app,arg);`)(app,arg);},[fn.toString(),arg]);
async function settle(page){let last='';for(let i=0;i<60;i++){await page.waitForTimeout(100);const now=(await canvasData(page)).camera;if(now===last)return;last=now;}}
try{
  const page=await browser.newPage({viewport:{width:1440,height:900},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push('pageerror: '+e.message));
  page.on('console',m=>{if(['error','warning'].includes(m.type())&&!/GPU stall|GL Driver Message|Automatic fallback to software WebGL/.test(m.text()))errors.push(m.type()+': '+m.text());});
  const started=Date.now();
  await page.goto(base+'/layers.html');await page.waitForSelector('body[data-ready=true]');
  await page.waitForFunction(()=>Number(document.querySelector('#viewport canvas')?.dataset.frames)>0);
  const firstFrame=Date.now()-started;
  await settle(page);

  // --- Atlas: every attempt slot, within the draw-call budget, idle when nothing moves.
  let data=await canvasData(page);
  assert.equal(data.view,'atlas');assert.equal(data.slots,'896');assert.equal(data.planes,'896');
  assert.deepEqual(JSON.parse(data.atlas),{slabs:890,failed:6,nodes:44500,ties:156730},'890 graph slabs (12 of them empty), 6 hollow failed frames, 50 nodes per graph, every primary tie');
  assert.ok(Number(data.drawCalls)<=12,`Atlas draw calls ${data.drawCalls} > 12`);
  const atlasCalls=Number(data.drawCalls),atlasGeometries=Number(data.geometries);
  const idle=Number(data.frames);await page.waitForTimeout(800);
  assert.equal(Number((await canvasData(page)).frames),idle,'No frames are rendered while nothing moves');
  // Filters change emphasis in place: no geometry is created, the budget holds.
  await store(page,app=>app.update({filters:{...app.getState().filters,model:['gpt-4.1'],method:['local']}},'check'));await frames(page);
  data=await canvasData(page);
  assert.equal(Number(data.geometries),atlasGeometries,'Filter emphasis must not rebuild geometry');
  assert.ok(Number(data.drawCalls)<=12);
  await store(page,app=>app.update({filters:{model:[],method:[],setting:[],seed:[],graph:'any',search:''},persona:'7',colorBy:'gender'},'check'));await frames(page);
  assert.equal(Number((await canvasData(page)).geometries),atlasGeometries,'Persona and colour changes must not rebuild geometry');
  // Hover: a tower card names the condition with "n of 8 valid"; click flies in and opens its Stack.
  const box=await page.locator('#viewport').boundingBox();
  let hit=null;
  for(const [fx,fy] of [[.42,.5],[.5,.48],[.6,.52],[.35,.55],[.55,.45],[.65,.5]]){
    await page.mouse.move(box.x+box.width*fx,box.y+box.height*fy);await frames(page);
    if(await page.locator('#node-tooltip').isVisible()){hit=[box.x+box.width*fx,box.y+box.height*fy];break;}
  }
  assert.ok(hit,'Hovering the atlas must find a tower');
  assert.match(await page.locator('#node-tooltip').innerText(),/\d of 8 valid/);
  assert.ok(await store(page,app=>app.getState().focusSlot)!==null,'Hover sets focusSlot in the store');
  await page.mouse.click(...hit);
  await page.waitForFunction(()=>document.querySelector('#viewport canvas').dataset.view==='stack');
  await settle(page);
  data=await canvasData(page);
  const towerLayers=Number(data.layers);
  assert.ok(towerLayers>=1&&towerLayers<=8);assert.equal(data.planes,String(towerLayers));
  assert.ok(Number(data.drawCalls)<=6*towerLayers,`Stack draw calls ${data.drawCalls} > ${6*towerLayers}`);
  const filters=await store(page,app=>app.getState().filters);
  assert.equal(filters.model.length,1);assert.equal(filters.method.length,1);assert.equal(filters.setting.length,1);

  // --- Stack: a whole condition of 8 seeds; a 9th is refused with a message, nothing truncated.
  const ids=await store(page,app=>app.data.runs.filter(r=>r.model==='gpt-6-sol'&&r.method==='iterative'&&r.culture==='us'&&r.language==='english').map(r=>r.run_id));
  assert.equal(ids.length,8);
  await store(page,(app,ids)=>app.update({layers:ids,active:ids[0],view:'stack'},'check'),ids);await settle(page);
  data=await canvasData(page);assert.equal(data.layers,'8');assert.equal(data.planes,'8');
  assert.ok(Number(data.drawCalls)<=48,`Stack of 8: ${data.drawCalls} draw calls`);
  // Shared-tie ribbons are off by default; their buffer is allocated with the stack but first uploaded when shown.
  // Show them once so the baseline counts every buffer the stack owns.
  await page.evaluate(()=>document.getElementById('g3-ribbons')?.click());await frames(page);
  await page.evaluate(()=>document.getElementById('g3-ribbons')?.click());await frames(page);
  const stackGeometries=Number((await canvasData(page)).geometries);
  const extra=await store(page,app=>app.data.runs.find(r=>r.model==='gpt-4.1').run_id);
  await store(page,(app,[ids,extra])=>app.update({layers:[...ids,extra]},'check'),[ids,extra]);await frames(page);await page.waitForTimeout(50);
  assert.match(await page.locator('#action-status').innerText(),/at most 8/);
  assert.equal((await store(page,app=>app.getState().layers)).length,8,'The stack keeps its 8 runs');
  // Opacity, persona, tie filter, encoding toggles: in-place updates only.
  await page.evaluate(()=>{const o=document.querySelector('#opacity');o.value='80';o.dispatchEvent(new Event('input',{bubbles:true}));});
  await store(page,app=>app.update({persona:'12'},'check'));
  await page.evaluate(()=>{const t=document.querySelector('#ties');t.value='shared';t.dispatchEvent(new Event('input',{bubbles:true}));});
  for(const id of ['g3-lift','g3-ribbons','g3-ribbons','g3-lift'])await page.evaluate(id=>document.getElementById(id)?.click(),id);
  await frames(page);
  data=await canvasData(page);
  assert.equal(Number(data.geometries),stackGeometries,'Stack updates must reuse geometry');
  assert.equal(await page.locator('.g3-thread').count(),1,'Identity thread is labelled');
  assert.match(await page.locator('.g3-thread').innerText(),/same persona, not a tie/);
  assert.equal(await page.locator('#plane-labels .g3-plane').count(),8);
  // A plane label is the second click from a tower: it opens that run's Replay.
  await page.evaluate(()=>{const t=document.querySelector('#ties');t.value='all';t.dispatchEvent(new Event('input',{bubbles:true}));});
  await store(page,app=>app.update({persona:''},'check'));
  await page.evaluate(()=>document.querySelectorAll('#plane-labels .g3-plane')[3].click());
  await page.waitForFunction(()=>document.querySelector('#viewport canvas').dataset.view==='replay');
  assert.equal(await store(page,app=>app.getState().active),ids[3]);

  // --- Replay: one plane, frame changes update buffers in place.
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  await settle(page);
  data=await canvasData(page);
  const listed=await page.evaluate(async()=>JSON.stringify((await import(document.querySelector('script[src*="layers.js"]').src)).drawables()));
  assert.equal(data.planes,'1');assert.ok(Number(data.drawCalls)<=6,`Replay draw calls ${data.drawCalls} > 6: ${listed}`);
  const replayGeometries=Number(data.geometries);
  for(const step of [0,1,5,12])await page.evaluate(step=>{const t=document.querySelector('#timeline');t.value=String(step);t.dispatchEvent(new Event('input',{bubbles:true}));},step);
  await frames(page);
  assert.equal(Number((await canvasData(page)).geometries),replayGeometries,'Replay frames must not rebuild geometry');
  assert.equal(await store(page,app=>app.getState().frame),12);

  // --- Label layer never reads layout inside animation frames (orbit drag + damping).
  await page.evaluate(()=>{
    window.__layoutReads=0;let inFrame=false;const raf=window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame=cb=>raf(t=>{inFrame=true;try{cb(t);}finally{inFrame=false;}});
    for(const [proto,keys] of [[HTMLElement.prototype,['offsetWidth','offsetHeight','offsetTop','offsetLeft','clientWidth','clientHeight']],[Element.prototype,['clientWidth','clientHeight','scrollWidth','scrollHeight']]])
      for(const key of keys){const d=Object.getOwnPropertyDescriptor(proto,key);if(d?.get)Object.defineProperty(proto,key,{...d,get(){if(inFrame)window.__layoutReads++;return d.get.call(this);}});}
    const rect=Element.prototype.getBoundingClientRect;Element.prototype.getBoundingClientRect=function(){if(inFrame)window.__layoutReads++;return rect.call(this);};
  });
  await store(page,app=>app.update({view:'stack'},'check'));await settle(page);
  const c=await page.locator('#viewport').boundingBox();
  await page.mouse.move(c.x+c.width*.5,c.y+c.height*.5);await page.mouse.down();await page.mouse.move(c.x+c.width*.5+160,c.y+c.height*.5+30,{steps:20});await page.mouse.up();
  await settle(page);
  assert.equal(await page.evaluate(()=>window.__layoutReads),0,'Forced layout reads inside the render loop');

  // --- Coverage cell = tower: a condition with a failed attempt stacks only its valid seeds.
  await page.evaluate(()=>[...document.querySelectorAll('#coverage-table button')].find(b=>/failed/.test(b.textContent))?.click());
  await page.waitForFunction(()=>document.querySelector('#viewport canvas').dataset.view==='stack');
  assert.match(await page.locator('#action-status').innerText(),/failed attempt not shown/);
  const failedStack=await store(page,app=>app.getState().layers.length);assert.ok(failedStack<8);

  // --- Every valid slot opens in Stack and in Replay without error (store-driven walk over all 890).
  const walk=await store(page,async app=>{
    const failures=[];
    for(const run of app.data.runs){
      try{
        app.update({layers:[run.run_id],active:run.run_id,view:'stack'},'check');
        if(app.getState().layers[0]!==run.run_id)failures.push(run.run_id+' stack');
        app.update({view:'replay'},'check');
        if(document.querySelector('#viewport canvas').dataset.planes!=='1')failures.push(run.run_id+' replay');
      }catch(error){failures.push(run.run_id+' '+error.message);}
    }
    return {count:app.data.runs.length,failures};
  });
  assert.equal(walk.count,890);assert.deepEqual(walk.failures,[]);

  // --- Back to the Atlas: same budget, emphasis of the stacked run.
  await store(page,app=>app.update({view:'atlas'},'check'));await settle(page);
  data=await canvasData(page);
  assert.equal(data.view,'atlas');assert.ok(Number(data.drawCalls)<=12);
  assert.ok(errors.length===0,errors.join('\n'));
  await page.close();

  // --- Reduced motion: camera moves are instant and auto-orbit is unavailable.
  const calm=await browser.newPage({viewport:{width:1440,height:900},reducedMotion:'reduce'});
  await calm.goto(base+'/layers.html');await calm.waitForSelector('body[data-ready=true]');await settle(calm);
  const before=(await canvasData(calm)).camera;
  await calm.evaluate(()=>document.querySelector('#top').click());await frames(calm);
  const after=(await canvasData(calm)).camera;assert.notEqual(after,before);
  await calm.waitForTimeout(300);assert.equal((await canvasData(calm)).camera,after,'No eased camera animation under reduced motion');
  assert.equal(await calm.locator('#rotate').isDisabled(),true,'No auto-orbit under reduced motion');
  await calm.close();

  // --- Phone: the stage renders the Atlas without horizontal overflow.
  const phone=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  const phoneErrors=[];phone.on('pageerror',e=>phoneErrors.push(e.message));
  await phone.goto(base+'/layers.html');await phone.waitForSelector('body[data-ready=true]');await settle(phone);
  assert.equal((await canvasData(phone)).view,'atlas');
  assert.ok(await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Phone overflow');
  assert.deepEqual(phoneErrors,[]);
  await phone.close();
  console.log(`PASS: Atlas 896 slots in ${atlasCalls} draw calls (first frame ${firstFrame} ms, idle GPU), filters/persona/colour in place; tower hover/click -> Stack; 8-seed stack and 9th refused; in-place stack and replay updates; no layout reads in frames; coverage = tower; all 890 valid slots open in Stack and Replay; reduced motion; phone.`);
}finally{await browser.close();}
