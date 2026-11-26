// Integration check: every one of the 890 valid primary graphs is reachable and renders in 3D, and the result
// list names each run_id exactly once (no repetitions, no supplements, no failed attempts).
// Saved data only. Run against a static build:
//   BROWSER_CHANNEL=chromium VIEWER_URL=http://127.0.0.1:8781 node layers-all-graphs-browser-check.mjs
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';

// The expected population comes straight from the deployed index, not from the page under test.
const index=await (await fetch(base+'/data/graphs/index.json')).json();
const column=name=>index.columns.indexOf(name);
const validSlots=index.attempts.filter(row=>row[column('status')]!=='failed').map(row=>row[column('slot')]);
const failedSlots=index.attempts.filter(row=>row[column('status')]==='failed').map(row=>row[column('slot')]);
const emptySlots=index.attempts.filter(row=>row[column('status')]==='empty').map(row=>row[column('slot')]);
assert.equal(index.attempts.length,896);assert.equal(validSlots.length,890);assert.equal(failedSlots.length,6);assert.equal(emptySlots.length,12);
const supplements=await (await fetch(base+'/data/graphs/supplements.json')).json();
const supplementIds=new Set(supplements.graphs.map(graph=>graph.run_id));assert.equal(supplementIds.size,6);

const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge',args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const store=(page,fn,arg)=>page.evaluate(async([source,arg])=>{const {app}=await import(document.querySelector('script[src*="layers.js"]').src);await app.ready;return new Function('app','arg',`return (${source})(app,arg);`)(app,arg);},[fn.toString(),arg]);
const ready=async page=>{await page.waitForSelector('body[data-ready=true]');await page.waitForFunction(()=>Number(document.querySelector('#viewport canvas')?.dataset.frames)>0);};
try{
  const page=await browser.newPage({viewport:{width:1440,height:900},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',error=>errors.push('pageerror: '+error.message));
  page.on('console',message=>{if(['error','warning'].includes(message.type())&&!/GPU stall|GL Driver Message|Automatic fallback to software WebGL/.test(message.text()))errors.push(message.type()+': '+message.text());});
  await page.goto(base+'/layers.html');await ready(page);

  // --- 1. Result list: with no filter set, every valid run_id appears exactly once.
  const expected=await store(page,app=>({ids:app.data.runs.map(run=>run.run_id),slots:app.data.runs.map(run=>run.slot),failed:app.data.failures.map(run=>run.run_id)}));
  assert.equal(expected.ids.length,890);assert.equal(new Set(expected.ids).size,890);
  assert.deepEqual([...expected.slots].sort((a,b)=>a-b),validSlots,'the page holds exactly the valid slots of the deployed index');
  assert.equal(expected.failed.length,6);
  if(!await page.locator('#run-results').isVisible())await page.locator('#tab-results').click();
  await page.waitForSelector('#run-results .flt-row');
  assert.equal(await page.locator('#run-results').getAttribute('data-count'),'890');
  const list=await page.evaluate(async()=>{
    const scroll=document.querySelector('#run-results .flt-scroll'),positionsOf=new Map(),idsAt=new Map();
    const read=()=>{for(const row of scroll.querySelectorAll('.flt-row')){const id=row.dataset.runId,position=Number(row.getAttribute('aria-posinset'));
      if(!positionsOf.has(id))positionsOf.set(id,new Set());positionsOf.get(id).add(position);
      if(!idsAt.has(position))idsAt.set(position,new Set());idsAt.get(position).add(id);}};
    // Step by less than one viewport so no row can be skipped; duplicates would show up as a second position.
    for(let top=0;;top+=Math.max(120,Math.floor(scroll.clientHeight*.6))){
      scroll.scrollTop=top;await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));read();
      if(top>=scroll.scrollHeight-scroll.clientHeight)break;
    }
    scroll.scrollTop=0;
    return {ids:[...positionsOf.keys()],repeated:[...positionsOf].filter(([,positions])=>positions.size!==1).map(([id])=>id),
      shared:[...idsAt].filter(([,ids])=>ids.size!==1).map(([position])=>position),positions:[...idsAt.keys()].sort((a,b)=>a-b),setsize:scroll.querySelector('.flt-row')?.getAttribute('aria-setsize')};
  });
  assert.equal(list.ids.length,890,'890 distinct run_ids in the result list');
  assert.deepEqual(list.repeated,[],'no run_id appears at two list positions');
  assert.deepEqual(list.shared,[],'no list position shows two runs');
  assert.deepEqual(list.positions,Array.from({length:890},(_,i)=>i+1),'positions 1..890, no gap');
  assert.equal(list.setsize,'890');
  assert.deepEqual([...list.ids].sort(),[...expected.ids].sort(),'the list is exactly the valid primary population');
  assert.ok(list.ids.every(id=>!supplementIds.has(id)),'no supplemental graph in the list');
  assert.ok(list.ids.every(id=>!expected.failed.includes(id)),'no failed attempt in the list');

  // --- 2. Every valid graph opens in 3D: Stack (one plane, a rendered frame) and Replay, driven through the store.
  const walk=await store(page,async app=>{
    const canvas=document.querySelector('#viewport canvas'),failures=[],frame=()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    let opened=0,ties=0,empty=0;
    for(const run of app.data.runs){
      try{
        const before=Number(canvas.dataset.frames);
        app.update({filters:{model:[],method:[],setting:[],seed:[],graph:'any',search:''},layers:[run.run_id],active:run.run_id,view:'stack',persona:''},'check');
        await frame();
        const s=app.getState();
        if(s.layers.length!==1||s.layers[0]!==run.run_id||s.active!==run.run_id)failures.push(run.run_id+': not the stacked run');
        if(canvas.dataset.view!=='stack'||canvas.dataset.layers!=='1'||canvas.dataset.planes!=='1')failures.push(run.run_id+': stack not on stage');
        if(!(Number(canvas.dataset.frames)>before))failures.push(run.run_id+': no frame rendered');
        if(!(Number(canvas.dataset.drawCalls)>0&&Number(canvas.dataset.drawCalls)<=6))failures.push(run.run_id+': draw calls '+canvas.dataset.drawCalls);
        // One update that changes nothing but the view must reach Replay of the same run.
        app.update({view:'replay'},'check');
        if(canvas.dataset.view!=='replay'||canvas.dataset.planes!=='1')failures.push(run.run_id+': replay not on stage');
        opened++;ties+=run.edges.length;if(!run.edges.length)empty++;
      }catch(error){failures.push(run.run_id+': '+error.message);}
    }
    return {opened,ties,empty,failures};
  });
  assert.deepEqual(walk.failures,[]);
  assert.equal(walk.opened,890);assert.equal(walk.empty,12,'12 valid graphs have no ties and still open');
  assert.equal(walk.ties,index.edges.bytes/2,'the graphs opened carry every primary tie exactly once');

  // --- 3. One store update that changes the active run and the view together applies both (no two-step workaround).
  const both=await store(page,async app=>{
    const ids=app.data.runs.slice(100,104).map(run=>run.run_id);
    app.update({layers:ids,active:ids[0],view:'stack'},'check');
    app.update({active:ids[2],view:'replay'},'check');
    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    const canvas=document.querySelector('#viewport canvas');
    return {active:app.getState().active===ids[2],view:app.getState().view,stage:canvas.dataset.view,planes:canvas.dataset.planes,title:document.getElementById('stage-title').textContent,label:app.data.runs[102].seed};
  });
  assert.deepEqual([both.active,both.view,both.stage,both.planes],[true,'replay','replay','1']);
  assert.match(both.title,new RegExp(`seed ${both.label}`),'the replay heading names the newly active run');

  // --- 4. By link: one slot per model x method (16), plus every empty graph (12), each opens its own run in Stack.
  const perCell=[];for(let cell=0;cell<16;cell++){const slot=validSlots.find(slot=>Math.floor(slot/56)===cell);perCell.push(slot);}
  for(const slot of [...new Set([...perCell,...emptySlots])]){
    await page.goto(`${base}/layers.html?v=stack&l=${slot}&a=${slot}`);await ready(page);
    const opened=await store(page,app=>({slot:app.data.runs.find(run=>run.run_id===app.getState().active).slot,layers:app.getState().layers.length,view:document.querySelector('#viewport canvas').dataset.view,planes:document.querySelector('#viewport canvas').dataset.planes}));
    assert.deepEqual(opened,{slot,layers:1,view:'stack',planes:'1'},`slot ${slot} opens by link`);
  }
  // A failed slot never opens as a graph: clear message, default view.
  await page.goto(`${base}/layers.html?v=stack&l=${failedSlots[0]}&a=${failedSlots[0]}`);await ready(page);
  assert.match(await page.locator('#action-status').textContent(),/failed attempt/);
  assert.equal(await page.evaluate(()=>document.querySelector('#viewport canvas').dataset.view),'atlas');

  assert.deepEqual(errors,[]);
  console.log(`PASS: result list names each of the 890 valid run_ids exactly once (no supplement, no failed attempt); all 890 graphs open in Stack and Replay with a rendered frame (${walk.ties} ties, 12 empty graphs); active+view in one update; ${new Set([...perCell,...emptySlots]).size} slots open by link; failed slot refused; zero console errors.`);
}finally{await browser.close();}
