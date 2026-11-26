// Workstream 3 browser check: live filters, active-filter bar, counts, result list, URL grammar, shortcuts.
// Run from viewer/ against a static build:  VIEWER_URL=http://127.0.0.1:8781 BROWSER_CHANNEL=chromium node layers-filters-browser-check.mjs
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const page_=process.env.FILTERS_PAGE||'layers.html';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
const saved=JSON.parse(await readFile(new URL('./public/data/saved-networks.json',import.meta.url),'utf8'));
const primaryIds=new Set(JSON.parse(await readFile(new URL('./public/data/primary-viewer.json',import.meta.url),'utf8')).runs.map(run=>run.run_id));
const supplementIds=saved.graphs.map(graph=>graph.run_id).filter(id=>!primaryIds.has(id));
assert.equal(supplementIds.length,6);
const errors=[];
// The build appends ?v=<hash> to module URLs; import the same URLs or a second app instance would start.
const modules=()=>{
  const version=()=>{const src=[...document.scripts].find(script=>/\/layers\.js(\?|$)/.test(script.src))?.src||'';return new URL(src||location.href).search;};
  window.__layersModule=()=>import(new URL(`./layers.js${version()}`,location.href).href);
  window.__filtersModule=()=>import(new URL(`./layers-filters.js${version()}`,location.href).href);
};
const watch=(page,label)=>{
  page.addInitScript(modules);
  // Chrome reports a missing favicon only as "Failed to load resource"; real failed requests are caught here.
  page.on('response',response=>{if(response.status()>=400&&!/favicon/.test(response.url()))errors.push(`${label} HTTP ${response.status()}: ${response.url()}`);});
  page.on('pageerror',error=>errors.push(`${label}: ${error.message}`));
  page.on('console',message=>{if(['error','warning'].includes(message.type())&&!/favicon|GL Driver Message|Failed to load resource/.test(message.text()))errors.push(`${label} console ${message.type()}: ${message.text()}`);});
};
const ready=async page=>{await page.waitForSelector('#active-filters[data-ready] .flt-summary',{state:'attached'});await page.waitForSelector('#run-results[data-ready]',{state:'attached'});};
const store=(page,fn,arg)=>page.evaluate(async([source,arg])=>{const {app}=await window.__layersModule();await app.ready;return new Function('app','arg',`return (${source})(app,arg);`)(app,arg);},[fn.toString(),arg]);
const summary=page=>page.locator('#active-filters .flt-summary').innerText();
const chips=page=>page.locator('#active-filters .flt-chip').allInnerTexts().then(list=>list.map(text=>text.replace(/\s*×\s*$/,'').trim()));
const option=(page,facet,value)=>page.locator(`#filter-root .flt-options[data-facet="${facet}"] .flt-option[data-value="${value}"]`);
// Keyboard activation: works wherever the shell docks the panel and proves keyboard operability.
const reveal=async(page,tab)=>{
  // Shell panels may sit in tabs inside a collapsible dock; open the dock, then the tab (as a user would).
  const button=page.locator(`#tab-${tab}`);if(!await button.count())return;
  if(!await button.isVisible()){const dock=await button.getAttribute('data-dock');const toggle=page.locator(`.dock-toggle[data-dock="${dock}"]`).first();if(dock&&await toggle.count()&&await toggle.isVisible()&&await toggle.getAttribute('aria-expanded')!=='true')await toggle.click();}
  if(await button.isVisible()&&await button.getAttribute('aria-selected')!=='true')await button.click();
};
// If the shell keeps filters and results in tabs, open the right tab first (a real user has to as well).
const press=async(locator,key='Space')=>{
  if(!await locator.isVisible()){const root=await locator.evaluate(node=>node.closest('#filter-root')?'filters':node.closest('#run-results')?'results':'');if(root)await reveal(locator.page(),root);}
  await locator.focus();await locator.press(key);
};
try{
  const page=await browser.newPage({viewport:{width:1440,height:900}});watch(page,'desktop');
  await page.goto(`${base}/${page_}`);await ready(page);

  // Unfiltered: exact study facts, no chips, reset hidden, nothing narrowed looks set.
  assert.equal(await summary(page),'All 890 valid graphs · no filters set');
  assert.match((await page.locator('#study-ledger').textContent()).replace(/\s+/g,' '),/896 attempts.*890 valid graphs \(12 empty\).*6 failures.*6 supplemental graphs excluded/);
  assert.equal(await page.locator('#active-filters').getAttribute('data-active'),'false');
  assert.equal(await page.locator('#active-filters .flt-chip').count(),0);
  assert.equal(await page.locator('#active-filters .flt-reset').isHidden(),true);
  assert.equal(await page.locator('#filter-root .flt-option[aria-pressed="true"]:not([data-value="any"])').count(),0);
  assert.equal(await page.locator('#run-results').getAttribute('data-count'),'890');

  // Facet counts in the DOM equal applyFilters for 20 random combinations, and the count line equals the list.
  const mismatches=await page.evaluate(async()=>{
    const {app}=await window.__layersModule(),F=await window.__filtersModule();await app.ready;
    const data=app.data,bad=[];let seed=7;const random=()=>((seed=(seed*1103515245+12345)%2147483648)/2147483648);
    const pick=values=>values.filter(()=>random()<0.3);
    for(let round=0;round<20;round++){
      const filters={model:pick(data.design.models),method:pick(data.design.methods),setting:pick(F.facetValues(data,'setting')),seed:pick(data.design.seeds),graph:['any','nonempty','empty'][Math.floor(random()*3)],search:random()<0.2?'english':''};
      app.update({filters},'check');
      await new Promise(requestAnimationFrame);
      for(const facet of F.FACETS)for(const value of F.facetValues(data,facet)){
        const expected=F.applyFilters(data.runs,{...filters,[facet]:[value]}).length;
        const shown=Number(document.querySelector(`#filter-root .flt-options[data-facet="${facet}"] .flt-option[data-value="${value}"] .flt-option-count`).textContent.replace(/,/g,''));
        if(shown!==expected)bad.push(`round ${round} ${facet}=${value}: ${shown} vs ${expected}`);
      }
      const n=F.applyFilters(data.runs,filters).length,line=document.querySelector('#active-filters .flt-summary').textContent;
      if(Number(document.querySelector('#run-results').dataset.count)!==n)bad.push(`round ${round}: list ${document.querySelector('#run-results').dataset.count} vs ${n}`);
      if(F.filtersActive(filters)&&!line.startsWith(`Showing ${n} of 890 valid graphs`))bad.push(`round ${round}: line "${line}" vs ${n}`);
      if(document.querySelectorAll('#active-filters .flt-chip').length!==F.activeChips(data,filters).length)bad.push(`round ${round}: chip count`);
    }
    app.update({filters:{model:[],method:[],setting:[],seed:[],graph:'any',search:''}},'check');
    return bad;
  });
  assert.deepEqual(mismatches,[]);

  // A filter change (store update, counts, bar, panel and visible rows) completes well inside 100 ms.
  const timing=await page.evaluate(async()=>{
    const {app}=await window.__layersModule(),times=[];
    for(const model of [['gpt-4.1'],['gpt-6-sol','gpt-6-luna'],[]]){const t=performance.now();app.update({filters:{model,method:[],setting:[],seed:[],graph:'any',search:''}},'check');times.push(performance.now()-t);}
    return Math.max(...times);
  });
  assert.ok(timing<100,`filter change took ${timing.toFixed(1)} ms`);

  // Set values look different from unset ones and only set values become chips.
  await press(option(page,'model','gpt-4.1'));
  await press(option(page,'method','iterative'));
  await press(option(page,'setting','us-japanese'));
  assert.equal(await option(page,'model','gpt-4.1').getAttribute('aria-pressed'),'true');
  assert.equal(await option(page,'model','gpt-6-sol').getAttribute('aria-pressed'),'false');
  await page.waitForTimeout(250); // let the .12s button transition settle before reading colours
  const look=await page.evaluate(()=>{const s=document.querySelector('#filter-root .flt-option[aria-pressed="true"]'),u=document.querySelector('#filter-root .flt-option[aria-pressed="false"]:not(:disabled)');const a=getComputedStyle(s),b=getComputedStyle(u);return {set:[a.backgroundColor,a.borderColor,a.fontWeight],unset:[b.backgroundColor,b.borderColor,b.fontWeight]};});
  for(let i=0;i<3;i++)assert.notEqual(look.set[i],look.unset[i],'set and unset facet values must differ visibly: '+JSON.stringify(look));
  assert.deepEqual(await chips(page),['Model: GPT-4.1','Method: Iterative','Setting: United States / Japanese']);
  assert.equal(await summary(page),'Showing 7 of 890 valid graphs · 1 failed attempt in scope (not rendered)','gpt-4.1 iterative US-Japanese lost seed 44006');
  assert.equal(await page.locator('#active-filters').getAttribute('data-active'),'true');
  // Visible focus on a facet toggle.
  await option(page,'seed','44001').focus();
  assert.notEqual(await option(page,'seed','44001').evaluate(node=>getComputedStyle(node).outlineStyle),'none','focus must be visible');
  // Arrow keys move within a facet group (one tab stop per group).
  await option(page,'method','global').focus();await page.keyboard.press('ArrowRight');
  assert.equal(await page.evaluate(()=>document.activeElement.dataset.value),'local');

  // Results: one condition minus a failure is 7 rows; "Stack these 7" replaces the stack in design order.
  await reveal(page,'results');
  assert.equal(await page.locator('#run-results').getAttribute('data-count'),'7');
  const stackButton=page.locator('#run-results .flt-stack-these');
  assert.equal(await stackButton.innerText(),'Stack these 7');
  const emitted=await page.evaluate(async()=>{const {app}=await window.__layersModule();return new Promise(resolve=>{const stop=app.subscribe((state,changed,source)=>{if(source==='layers-filters'&&changed.has('layers')){stop();resolve(state.layers);}});document.querySelector('#run-results .flt-stack-these').click();});});
  assert.equal(emitted.length,7);
  await page.waitForFunction(()=>document.querySelector('#run-results .flt-stack-these').disabled);
  assert.deepEqual(await store(page,app=>app.getState().layers),emitted);
  assert.equal(await page.locator('#active-filters .flt-stack li').count(),7,'stack chips mirror the store');
  assert.equal(await page.locator('#active-filters .flt-stack [data-outside="true"]').count(),0);

  // Chip removal by keyboard moves focus to the next chip; Reset clears everything and keeps the stack.
  await press(page.locator('#active-filters .flt-chip').first(),'Enter');
  assert.deepEqual(await chips(page),['Method: Iterative','Setting: United States / Japanese']);
  assert.match(await page.evaluate(()=>document.activeElement.textContent),/Method: Iterative/);
  await page.locator('#active-filters .flt-reset').click();
  assert.equal(await page.locator('#active-filters .flt-chip').count(),0);
  assert.equal(await summary(page),'All 890 valid graphs · no filters set');
  assert.match((await page.locator('#study-ledger').textContent()).replace(/\s+/g,' '),/896 attempts.*890 valid graphs \(12 empty\).*6 failures.*6 supplemental graphs excluded/);
  assert.equal((await store(page,app=>app.getState().layers)).length,7,'filters never change the stack');

  // Eight seeds of one condition stack together (MAX_STACK_LAYERS); a ninth is refused with a message.
  await store(page,app=>app.update({filters:{model:['gpt-6-sol'],method:['local'],setting:['india-english'],seed:[],graph:'any',search:''}},'check'));
  await page.locator('#run-results .flt-stack-these').click();
  await page.waitForFunction(async()=>{const {app}=await window.__layersModule();return app.getState().layers.length===8;});
  await store(page,app=>app.update({filters:{model:['gpt-6-sol'],method:['global'],setting:['india-english'],seed:[44000],graph:'any',search:''}},'check'));
  await press(page.locator('#run-results .flt-row .flt-add').first(),'Enter');
  assert.equal((await store(page,app=>app.getState().layers)).length,8);
  assert.match(await page.locator('#action-status').innerText(),/stack is full/i);

  // Shortcut 8 replays the eighth stacked run; "/" focuses search.
  await page.locator('body').click({position:{x:2,y:2}}).catch(()=>{});
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.press('8');
  const afterEight=await store(page,app=>({active:app.getState().active,eighth:app.getState().layers[7],view:app.getState().view}));
  assert.equal(afterEight.active,afterEight.eighth);
  await reveal(page,'filters');
  await page.evaluate(()=>document.activeElement?.blur());
  if(await page.locator('#filter-root input[data-flt-search]').isVisible()){await page.keyboard.press('/');assert.equal(await page.evaluate(()=>document.activeElement.dataset.fltSearch),'true');await page.keyboard.press('Escape');}

  // Search: whole words; supplements and failures never appear.
  await store(page,app=>app.update({filters:{model:[],method:[],setting:[],seed:[],graph:'any',search:''}},'check'));
  for(const id of supplementIds){
    await store(page,(app,id)=>app.update({filters:{model:[],method:[],setting:[],seed:[],graph:'any',search:id}},'check'),id);
    assert.equal(await page.locator('#run-results').getAttribute('data-count'),'0',`supplement ${id} must not match`);
  }
  await reveal(page,'results');
  assert.equal(await page.locator('#run-results .flt-none').isVisible(),true);
  await store(page,app=>app.update({filters:{model:[],method:[],setting:[],seed:[],graph:'any',search:''}},'check'));

  // Walk the whole virtualised list: 890 unique primary run_ids, design order, nothing else.
  await reveal(page,'results');
  const walk=await page.evaluate(async()=>{
    const scroll=document.querySelector('#run-results .flt-scroll'),seen=new Map();
    for(let top=0;;top+=Math.max(200,scroll.clientHeight)){
      scroll.scrollTop=top;await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
      for(const row of scroll.querySelectorAll('.flt-row'))seen.set(row.dataset.runId,Number(row.getAttribute('aria-posinset')));
      if(top>=scroll.scrollHeight)break;
    }
    const rendered=scroll.querySelectorAll('.flt-row').length;scroll.scrollTop=0;
    return {ids:[...seen.keys()],positions:[...seen.values()],rendered};
  });
  assert.equal(walk.ids.length,890);assert.equal(new Set(walk.ids).size,890);
  assert.ok(walk.ids.every(id=>primaryIds.has(id)),'only primary graphs');
  assert.deepEqual([...walk.positions].sort((a,b)=>a-b),Array.from({length:890},(_,i)=>i+1));
  assert.ok(walk.rendered<80,`virtualised: ${walk.rendered} rows in the DOM`);

  // Row keyboard navigation and Replay of an empty graph.
  await press(option(page,'graph','empty'));
  await reveal(page,'results');
  assert.equal(await summary(page),'Showing 12 of 890 valid graphs · 12 empty');
  assert.equal(await page.locator('#run-results .flt-stack-these').isHidden(),true);
  assert.match(await page.locator('#run-results .flt-guide').innerText(),/narrow to 8 or fewer/);
  await page.locator('#run-results .flt-row').first().locator('.flt-replay').focus();
  await page.keyboard.press('ArrowDown');
  assert.equal(await page.evaluate(()=>document.activeElement.closest('.flt-row').getAttribute('aria-posinset')),'2');
  await store(page,app=>{const s=app.getState();app.update({layers:s.layers.slice(0,4),active:s.layers[0],view:'stack'},'check');});
  await page.keyboard.press('Enter');
  const replayed=await store(page,app=>({view:app.getState().view,active:app.getState().active,edges:app.data.runs.find(run=>run.run_id===app.getState().active).edges.length}));
  assert.equal(replayed.view,'replay');assert.equal(replayed.edges,0);
  assert.match(await page.locator('#active-filters .flt-view').innerText(),/^Replay · run 5 of 5$/);

  // Address bar writer (enabled by the integrator): short readable grammar, round trip.
  const written=await page.evaluate(async()=>{
    const {app}=await window.__layersModule(),{syncUrl,decodeState}=await window.__filtersModule();
    const stop=syncUrl(app,{delay:0});
    app.update({filters:{model:['gpt-4.1'],method:['local','global'],setting:['us-english'],seed:[],graph:'nonempty',search:''}},'check');
    await new Promise(r=>setTimeout(r,30));stop();
    return {search:location.search,state:decodeState(new URLSearchParams(location.search),app.data)};
  });
  assert.match(written.search,/^\?v=replay&m=gpt-4\.1&p=global,local&s=us-english&g=nonempty&l=[\d.]+&a=\d+/);
  assert.deepEqual(written.state.filters.method,['global','local']);
  assert.deepEqual(errors,[]);

  // Short links open filters, stack and active run.
  await page.goto(`${base}/${page_}?v=stack&m=gpt-4.1&p=local&s=us-english&l=8.9&a=9&c=religion`);await ready(page);
  await page.waitForFunction(async()=>{const {app}=await window.__layersModule();return app.getState().layers.length===2;});
  const opened=await store(page,app=>{const s=app.getState();return {layers:s.layers.map(id=>app.data.runs.find(run=>run.run_id===id).slot),active:app.data.runs.find(run=>run.run_id===s.active).slot,colorBy:s.colorBy,view:s.view};});
  assert.deepEqual(opened,{layers:[8,9],active:9,colorBy:'religion',view:'stack'});
  assert.deepEqual(await chips(page),['Model: GPT-4.1','Method: Local','Setting: United States / English']);
  // A failed slot gives a clear message and the default view, never a substitute.
  const failedSlot=await store(page,app=>app.data.slots.find(slot=>slot.status==='failed').slot);
  await page.goto(`${base}/${page_}?v=stack&l=${failedSlot}`);await ready(page);
  assert.equal(await page.locator('#active-filters .flt-notice').isVisible(),true);
  assert.match(await page.locator('#active-filters .flt-notice').innerText(),new RegExp(`Slot ${failedSlot} is a failed attempt.*Showing the default view`));
  assert.equal(await page.locator('#active-filters .flt-chip').count(),0);
  assert.ok(!(await store(page,app=>app.getState().layers.map(id=>app.data.runs.find(run=>run.run_id===id).slot))).includes(failedSlot));
  await page.locator('#active-filters .flt-notice-close').click();
  assert.equal(await page.locator('#active-filters .flt-notice').isHidden(),true);
  // Legacy ?view= links from the deployed site open the same runs; their single-select filters become chips.
  const legacyRuns=['global','local'].map(method=>`final_060dcd3a86d5_main_${method}_gpt-6-sol_india_english_s44002`);
  await page.goto(`${base}/${page_}?dataset=primary&view=${encodeURIComponent(JSON.stringify({runs:legacyRuns,filters:{collection:'primary',model:'gpt-6-sol',method:'',culture:'india',language:'english',seed:'44002'},mode:'compare',active:legacyRuns[1],person:'',color:'gender'}))}`);await ready(page);
  await page.waitForFunction(async()=>{const {app}=await window.__layersModule();return app.getState().layers.length===2;});
  assert.deepEqual(await store(page,app=>app.getState().layers),legacyRuns);
  assert.equal(await store(page,app=>app.getState().active),legacyRuns[1]);
  assert.deepEqual(await chips(page),['Model: GPT-6 Sol','Setting: India / English','Seed: 44002']);
  assert.match(await summary(page),/^Showing 4 of 890 valid graphs/);

  // Sticky bar stays on screen at every scroll position.
  for(const y of [600,2000,99999]){await page.evaluate(y=>scrollTo(0,y),y);const box=await page.locator('#active-filters').boundingBox();assert.ok(box&&box.y>=-1&&box.y<200,`bar visible at scrollY ${y}`);}
  await page.close();

  // Phone: no horizontal scroll, compact sticky bar, 44 px targets.
  const phone=await browser.newPage({viewport:{width:390,height:844},deviceScaleFactor:1,hasTouch:true,isMobile:true});watch(phone,'phone');
  await phone.goto(`${base}/${page_}?m=gpt-4.1&p=local`);await ready(phone);
  assert.ok(await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'no horizontal scroll on phone');
  const sizes=await phone.evaluate(()=>[...document.querySelectorAll('#active-filters button')].filter(b=>b.offsetParent).map(b=>{const r=b.getBoundingClientRect();return [b.className,Math.round(r.width),Math.round(r.height)];}));
  for(const [name,width,height] of sizes)assert.ok(width>=44&&height>=44,`${name} is ${width}x${height}`);
  const barHeight=(await phone.locator('#active-filters').boundingBox()).height;
  assert.ok(barHeight<=140,`phone bar is ${barHeight}px tall`);
  for(const y of [800,99999]){await phone.evaluate(y=>scrollTo(0,y),y);const box=await phone.locator('#active-filters').boundingBox();assert.ok(box&&box.y>=-1&&box.y<200,`phone bar visible at scrollY ${y}`);}
  await phone.close();
  assert.deepEqual(errors,[]);
  console.log('PASS: exact unfiltered facts, DOM facet counts = applyFilters x20, set/unset contrast, visible focus, arrow-key facets, chips remove themselves, reset keeps stack, stack 7/8 + 9th refused, keys 8 and /, supplements excluded, 890 unique rows virtualised, keyboard rows, empty-graph replay, URL writer, short/failed/legacy links, sticky bar, phone targets.');
}finally{await browser.close();}
