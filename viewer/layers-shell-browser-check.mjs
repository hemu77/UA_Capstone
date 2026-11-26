// Page-shell check (workstream 4, SPEC.md B.6 / C.6 / D "Shell").
// Run against a served dist: VIEWER_URL=http://127.0.0.1:8781 node layers-shell-browser-check.mjs
// Optional SHOT_DIR=<dir> writes screenshots there (never into viewer/).
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
import {mkdirSync} from 'node:fs';
import {join} from 'node:path';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const shots=process.env.SHOT_DIR||'';
if(shots)mkdirSync(shots,{recursive:true});
const launch={headless:true,args:['--use-gl=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']};
if(process.env.BROWSER_CHANNEL!=='chromium')launch.channel=process.env.BROWSER_CHANNEL||'msedge';
const browser=await chromium.launch(launch);

// Every id other modules and checks read (SPEC C.1): may move, never disappear.
const FROZEN=['help-open','share','dataset-summary','download-selection','study-ledger','active-filters','filter-root','run-results','retry-load','reset-view','evidence-notice','coverage-panel','coverage-table','match-note','atlas-mode','formation-mode','compare-mode','rewind','previous','play','next','step-label','playback-scope','timeline','speed','show-final','drag-mode','focus-person','clear-person','layer-count','layers','undo','restore','action-status','graph-interaction','color','node-labels','spacing-value','separation','ties','opacity','comparison-scope','active-badge','stage-title','event-delta','stage-state','viewport','plane-labels','stage-overlay','node-tooltip','status','expand-stage','zoom-in','zoom-out','orbit','top','front','rotate','person','person-prev','person-next','person-detail','play-persona','incident','quick-summary','person-degrees','event-heading','event-explanation','event-changes','legend','replay-note','insight-root','trajectory-title','trajectory','event-ledger','ledger-count','events','measurements','research-question','rq-explanation','rq-status','rq-issues','rq-pairs','rq-homophily','roster-limitations','metric','metric-definition','plot','metric-deltas','comparison','warning','rows','aggregate-note','aggregate-graph','source-gallery','provenance-summary','provenance-fields','source-png','provenance-hashes','primary-findings','saved-plots','help','help-title','help-close'];
// Retired at integration (legacy filter drawer, primary gallery and its dialog): these ids must not come back.
const RETIRED=['filter-form','model','method','culture','language','seed','apply','reset-filters','filter-preview','candidate','add','candidate-detail','filter-chips','coverage-summary','open-filters','legend-heading','plot-drawer','plot-filters','plot-cohort','plot-method','plot-search','plot-clear','plot-status','plot-retry','plot-grid','plot-page-back','plot-page-next','plot-dialog','plot-dialog-title','plot-close','plot-dialog-cohort','network-stage','network-person','network-zoom-in','network-zoom-out','network-reset','network-status','plot-back','plot-dialog-position','plot-forward','plot-original','plot-source','plot-replay','plot-record','legacy-controls'];
// Roots whose inner DOM belongs to other workstreams: their targets are reported, not asserted here.
const FOREIGN='#active-filters,#filter-root,#run-results,#study-ledger,#insight-root,#legend,#stage-overlay,#plane-labels,#node-tooltip,#coverage-table,#events,#source-gallery,#person-degrees,#rq-homophily,#trajectory,#plot,#aggregate-graph,#plot-grid';
const SIZES=[['desktop',1440,900,0.60],['tablet',1024,768,0.60],['phone',390,844,0.45]];
const failures=[];
const check=(name,fn)=>{try{fn();}catch(error){failures.push(`${name}: ${error.message.split('\n')[0]}`);}};

async function metrics(page) {
  return page.evaluate(({foreign})=>{
    const visible=e=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return r.width>0&&r.height>0&&s.visibility!=='hidden'&&s.display!=='none'&&!e.closest('[hidden],dialog:not([open])');};
    // Content inside a horizontal scroller (tables, wide charts) is scrollable, not page overflow.
    const clipped=e=>{for(let p=e.parentElement;p&&p!==document.body;p=p.parentElement)if(getComputedStyle(p).overflowX!=='visible')return true;return false;};
    const vp=document.querySelector('#viewport').getBoundingClientRect();
    const seen=Math.max(0,Math.min(vp.right,innerWidth)-Math.max(vp.left,0))*Math.max(0,Math.min(vp.bottom,innerHeight)-Math.max(vp.top,0));
    const overflow=[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return visible(e)&&r.right>document.documentElement.clientWidth+1&&getComputedStyle(e).position!=='fixed'&&!clipped(e);}).slice(0,8).map(e=>`${e.tagName.toLowerCase()}#${e.id}.${e.className}`);
    const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);
    const small=[...document.querySelectorAll('button,a[href],select,input,summary,[role=tab],[tabindex="0"]')].filter(e=>visible(e)&&!e.closest(foreign)&&!e.closest('.skip')).map(e=>{
      const box=(['checkbox','radio'].includes(e.type)&&e.closest('label')?e.closest('label'):e).getBoundingClientRect();
      const inline=e.tagName==='A'&&getComputedStyle(e).display==='inline';
      return {name:(e.id||e.getAttribute('aria-label')||e.textContent).trim().slice(0,32),w:Math.round(box.width),h:Math.round(box.height),inline};
    }).filter(t=>t.inline?t.h<20:(t.w<44||t.h<44));
    const statement=[...document.querySelectorAll('body *')].filter(e=>visible(e)&&!e.closest('#active-filters,#study-ledger,.ledger-fallback,.coverage-card,#saved-plots,dialog')&&[...e.childNodes].some(n=>n.nodeType===3&&/\b896\b/.test(n.textContent))).map(e=>e.tagName+'#'+e.id);
    return {scrollWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth,ratio:seen/(innerWidth*innerHeight),viewport:[vp.x,vp.y,vp.width,vp.height].map(Math.round),
      overflow,duplicates:ids.filter((id,i)=>ids.indexOf(id)!==i),missing:[],small,statement,shell:document.body.dataset.shell,view:document.body.dataset.view};
  },{foreign:FOREIGN});
}
async function focusVisible(page) {
  return page.evaluate(()=>{const e=document.activeElement;if(!e||e===document.body)return {ok:false,name:'body'};const s=getComputedStyle(e),r=e.getBoundingClientRect();
    const ring=(s.outlineStyle!=='none'&&parseFloat(s.outlineWidth)>=2)||/rgb/.test(s.boxShadow);
    return {ok:ring&&r.width>0&&r.height>0,name:(e.id||e.getAttribute('aria-label')||e.textContent||e.tagName).trim().slice(0,40)};});
}
const shot=async(page,name)=>{if(shots)await page.screenshot({path:join(shots,`${name}.png`)});};

try{
  for(const [size,width,height,minimum] of SIZES){
    const phone=size==='phone';
    const context=await browser.newContext({viewport:{width,height},deviceScaleFactor:1,hasTouch:phone,isMobile:phone});
    const page=await context.newPage(),errors=[];
    page.on('pageerror',error=>errors.push(`pageerror: ${error.message}`));
    page.on('console',message=>{if(['error','warning'].includes(message.type())&&!/GL Driver Message|GPU stall|WebGL/.test(message.text()))errors.push(`${message.type()}: ${message.text()}`);});
    await page.goto(base+'/layers.html');
    await page.waitForFunction(()=>document.body.dataset.ready==='true'&&document.body.dataset.shell);
    await page.waitForSelector('#layers .select-run',{state:'attached'});await page.waitForTimeout(800);
    await shot(page,`${size}-top`);
    const m=await metrics(page);
    const expectedShell={desktop:'desktop',tablet:'tabs',phone:'sheet'}[size];
    check(`${size} shell mode`,()=>assert.equal(m.shell,expectedShell));
    check(`${size} horizontal overflow`,()=>assert.ok(m.scrollWidth<=m.clientWidth&&!m.overflow.length,`scrollWidth ${m.scrollWidth} > ${m.clientWidth}; ${m.overflow.join(', ')}`));
    check(`${size} stage area`,()=>assert.ok(m.ratio>=minimum,`3D canvas covers ${(100*m.ratio).toFixed(1)}% of the first screen; need ${100*minimum}% (${m.viewport})`));
    check(`${size} duplicate ids`,()=>assert.deepEqual(m.duplicates,[]));
    check(`${size} touch targets`,()=>{if(phone)assert.deepEqual(m.small,[],'shell controls under 44x44 px');});
    check(`${size} population statement`,()=>assert.deepEqual(m.statement,[],'896 restated outside the ledger'));
    const present=await page.evaluate(ids=>ids.filter(id=>!document.getElementById(id)),FROZEN);
    check(`${size} frozen ids`,()=>assert.deepEqual(present,[]));
    const retired=await page.evaluate(ids=>ids.filter(id=>document.getElementById(id)),RETIRED);
    check(`${size} retired legacy controls stay deleted`,()=>assert.deepEqual(retired,[]));
    const roots=await page.evaluate(()=>['#stage-overlay','#viewport','#plane-labels','#node-tooltip'].every(sel=>document.querySelector('.stage').contains(document.querySelector(sel))));
    check(`${size} stage roots`,()=>assert.ok(roots));
    const landmarks=await page.evaluate(()=>({banner:document.querySelectorAll('body>header').length,main:document.querySelectorAll('main').length,h1:document.querySelectorAll('h1').length,contentinfo:document.querySelectorAll('body>footer').length,lives:document.querySelectorAll('#action-status[role=status]').length}));
    check(`${size} landmarks`,()=>assert.deepEqual(landmarks,{banner:1,main:1,h1:1,contentinfo:1,lives:1}));
    // One "Filters" entry point on screen.
    const filters=await page.evaluate(()=>[...document.querySelectorAll('button,a,summary')].filter(e=>e.getBoundingClientRect().width>0&&!e.closest('#active-filters,#filter-root,#run-results')&&/^\s*filters\b/i.test(e.textContent)).length);
    check(`${size} one Filters entry`,()=>assert.equal(filters,1));

    // Study ledger: visible without scrolling on desktop; within one tap elsewhere.
    if(size==='desktop'){
      const ledger=await page.evaluate(()=>{const e=document.querySelector('#panel-study');const r=e.getBoundingClientRect();return !e.hidden&&r.top>=0&&r.bottom<=innerHeight&&r.height>0;});
      check('desktop ledger visible',()=>assert.ok(ledger));
    }else{
      await page.locator('#tab-study').click();await page.waitForTimeout(150);
      const ledger=await page.evaluate(()=>{const e=document.querySelector('#panel-study'),r=e.getBoundingClientRect();return {shown:!e.hidden&&r.height>0&&r.top<innerHeight,focus:document.activeElement?.classList.contains('sheet-title'),below:r.top>=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--chrome-h'))-1};});
      check(`${size} ledger one tap`,()=>assert.ok(ledger.shown&&ledger.focus,JSON.stringify(ledger)));
      check(`${size} sheet keeps the filter bar`,()=>assert.ok(ledger.below));
      await shot(page,`${size}-sheet-study`);
      await page.keyboard.press('Escape');await page.waitForTimeout(100);
      const closed=await page.evaluate(()=>({hidden:document.querySelector('#panel-study').hidden,focus:document.activeElement?.id}));
      check(`${size} Escape closes and returns focus`,()=>assert.deepEqual(closed,{hidden:true,focus:'tab-study'}));
      await page.locator('#tab-filters').click();await page.waitForTimeout(200);await shot(page,`${size}-sheet-filters`);
      await page.locator('#dock-left .dock-close').click();
      await page.locator('#tab-stack').click();await page.waitForTimeout(200);await shot(page,`${size}-sheet-stack`);
      const stackSmall=phone?(await metrics(page)).small:[];
      check(`${size} stack panel targets`,()=>assert.deepEqual(stackSmall,[]));
      await page.locator('#dock-right .dock-close').click();
    }

    // Desktop docks: rails reopen, toggles collapse, focus returns to the toggle.
    if(size==='desktop'){
      const before=(await metrics(page)).viewport[2];
      await page.locator('#tab-persona').click();await page.waitForTimeout(250);
      const opened=await page.evaluate(()=>({open:document.querySelector('#dock-right').dataset.open,panel:!document.querySelector('#panel-persona').hidden,width:Math.round(document.querySelector('#viewport').getBoundingClientRect().width)}));
      check('desktop right dock opens on its tab',()=>assert.ok(opened.open==='true'&&opened.panel&&opened.width<before));
      await shot(page,'desktop-right-dock');
      await page.locator('#person').focus();await page.locator('#dock-right .dock-toggle').click();await page.waitForTimeout(150);
      const collapsed=await page.evaluate(()=>({open:document.querySelector('#dock-right').dataset.open,focus:document.activeElement?.classList.contains('dock-toggle')}));
      check('desktop right dock collapses',()=>assert.deepEqual(collapsed,{open:'false',focus:true}));
      await page.locator('#tab-results').click();
      const results=await page.evaluate(()=>!document.querySelector('#panel-results').hidden&&document.querySelector('#panel-filters').hidden);
      check('desktop left tabs switch panels',()=>assert.ok(results));
      await page.locator('#tab-filters').click();
      // Header link to the ledger focuses it.
      await page.locator('#site-menu a[href="#study-ledger"]').click();
      const ledgerFocus=await page.evaluate(()=>document.activeElement?.id);
      check('desktop ledger link focuses the ledger',()=>assert.equal(ledgerFocus,'panel-study'));
    }

    // View mirroring: transport only in replay.
    await page.evaluate(()=>scrollTo(0,0));
    await page.locator('#formation-mode').click();await page.waitForTimeout(400);
    const replay=await page.evaluate(()=>({view:document.body.dataset.view,transport:getComputedStyle(document.querySelector('.transport')).display!=='none'}));
    check(`${size} replay shows transport`,()=>assert.deepEqual(replay,{view:'replay',transport:true}));
    await shot(page,`${size}-replay`);
    const rm=await metrics(page);
    check(`${size} replay overflow`,()=>assert.ok(rm.scrollWidth<=rm.clientWidth&&!rm.overflow.length,rm.overflow.join(', ')));
    check(`${size} replay targets`,()=>{if(phone)assert.deepEqual(rm.small,[]);});
    await page.locator('#compare-mode').click();await page.waitForTimeout(300);
    const stack=await page.evaluate(()=>({view:document.body.dataset.view,transport:getComputedStyle(document.querySelector('.transport')).display!=='none'}));
    check(`${size} stack hides transport`,()=>assert.deepEqual(stack,{view:'stack',transport:false}));

    // Header menu (collapsed below 900 px) and dialog focus return.
    const menu=await page.locator('#menu-toggle').isVisible();
    if(menu){
      await page.locator('#menu-toggle').click();
      const open=await page.locator('#site-menu').isVisible();check(`${size} menu visible`,()=>assert.ok(open));
      await shot(page,`${size}-menu`);
      await page.keyboard.press('Escape');
      const after=await page.evaluate(()=>({menu:getComputedStyle(document.querySelector('#site-menu')).display,focus:document.activeElement?.id}));
      check(`${size} menu Escape`,()=>assert.deepEqual(after,{menu:'none',focus:'menu-toggle'}));
      await page.locator('#menu-toggle').click();
    }
    await page.locator('#help-open').click();await page.waitForTimeout(100);
    const helpOpen=await page.evaluate(()=>document.querySelector('#help').open);
    check(`${size} help opens`,()=>assert.ok(helpOpen));
    await page.keyboard.press('Escape');await page.waitForTimeout(100);
    const returned=await page.evaluate(()=>document.activeElement?.id);
    check(`${size} help returns focus`,()=>assert.equal(returned,menu?'menu-toggle':'help-open'));

    // The live region shows as a toast.
    await page.evaluate(()=>{document.getElementById('action-status').textContent='Shell check announcement';});
    await page.waitForTimeout(250);
    const toast=await page.evaluate(()=>{const e=document.getElementById('action-status'),r=e.getBoundingClientRect();return Number(getComputedStyle(e).opacity)>0.9&&r.bottom<=innerHeight&&r.top>=0;});
    check(`${size} toast visible`,()=>assert.ok(toast));

    // Keyboard pass: focus is always visible and never lands on an invisible element.
    if(!phone){
      await page.goto(base+'/layers.html');await page.waitForFunction(()=>document.body.dataset.ready==='true'&&document.body.dataset.shell);await page.waitForTimeout(500);
      const lost=[];
      for(let i=0;i<45;i++){await page.keyboard.press('Tab');const state=await focusVisible(page);if(!state.ok)lost.push(state.name);}
      check(`${size} keyboard focus visible`,()=>assert.deepEqual(lost,[]));
    }
    check(`${size} console`,()=>assert.deepEqual(errors,[]));
    if(m.small.length&&!phone)console.log(`${size}: dense controls (allowed above 700 px): ${m.small.length}`);
    await context.close();
  }
}finally{await browser.close();}
if(failures.length){console.error(`Shell check failed (${failures.length}):\n- ${failures.join('\n- ')}`);process.exit(1);}
console.log('Shell check passed: layout, stage area, overflow, targets, ids, docks, sheet, menu, dialogs, focus.');
