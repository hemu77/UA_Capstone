// Browser plugin absent: exercise the static build with installed Playwright.
// Current-study routing, exact population counts, failures, empty graphs, export, archived-link rejection,
// supplement isolation and phone layout, driven through the live filter bar and result list.
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
const errors=[],requests=[];
const SUMMARY='All 890 valid graphs · no filters set';
// The build appends ?v=<hash> to module URLs; reuse that URL so the page's own store is inspected.
const modules=()=>{window.__layersModule=()=>{const src=[...document.scripts].find(script=>/\/layers\.js(\?|$)/.test(script.src))?.src;return import(src);};};
const state=page=>page.evaluate(async()=>{const {app}=await window.__layersModule();await app.ready;const s=app.getState();return {...s,runs:s.layers.map(id=>app.data.runs.find(run=>run.run_id===id))};});
const ready=page=>page.waitForSelector('#active-filters[data-ready] .flt-summary',{state:'attached'});
const reveal=async(page,tab)=>{
  // Shell panels may sit in tabs inside a collapsible dock; open the dock, then the tab (as a user would).
  const button=page.locator(`#tab-${tab}`);if(!await button.count())return;
  if(!await button.isVisible()){const dock=await button.getAttribute('data-dock');const toggle=page.locator(`.dock-toggle[data-dock="${dock}"]`).first();if(dock&&await toggle.count()&&await toggle.isVisible()&&await toggle.getAttribute('aria-expanded')!=='true')await toggle.click();}
  if(await button.isVisible()&&await button.getAttribute('aria-selected')!=='true')await button.click();
};
const toggle=async(page,facet,value)=>{const option=page.locator(`#filter-root .flt-options[data-facet="${facet}"] .flt-option[data-value="${value}"]`);if(!await option.isVisible())await reveal(page,'filters');await option.focus();await option.press('Space');};
try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  await page.addInitScript(modules);
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await page.goto(base+'/');await ready(page);
  assert.equal(await page.locator('#active-filters .flt-summary').innerText(),SUMMARY);
  // The exact population (896 / 890 / 12 / 6) is stated once, by the study ledger, not restated by the filter bar.
  assert.match((await page.locator('#study-ledger').textContent()).replace(/\s+/g,' '),/896 attempts.*890 valid graphs \(12 empty\).*6 failures.*6 supplemental graphs excluded/);
  assert.doesNotMatch(await page.locator('#active-filters').innerText(),/\b896\b|failures|12 empty/);
  assert.equal(await page.locator('#dataset, #collection, #failure-list').count(),0);
  assert.equal((await state(page)).layers.length,4,'default comparison: four methods of one condition');

  // Coverage table (accessible equivalent of the atlas): 112 conditions; a cell puts its condition in the filters.
  await page.locator('#coverage-panel summary').click();
  assert.equal(await page.locator('#coverage-table button').count(),112);
  await page.locator('#coverage-table button').first().click();
  const cell=await state(page);
  assert.equal(cell.filters.model.length,1);assert.equal(cell.filters.method.length,1);assert.equal(cell.filters.setting.length,1);
  assert.equal(await page.locator('#active-filters .flt-chip').count(),3,'the bar shows the condition the cell selected');

  // Narrow to one seed and stack that single graph.
  await toggle(page,'seed','44000');
  await reveal(page,'results');
  assert.equal(await page.locator('#run-results').getAttribute('data-count'),'1');
  assert.match(await page.locator('#active-filters .flt-summary').innerText(),/^Showing 1 of 890 valid graphs/);
  await page.locator('#run-results .flt-stack-these').click();
  await page.waitForFunction(async()=>{const {app}=await window.__layersModule();return app.getState().layers.length===1;});
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  // The first coverage cell is a Global condition: its record is one response, so Replay states that and offers
  // the person-level run of the same model, setting and seed, which then steps event by event from event 0.
  await page.locator('#formation-mode').click();
  await page.waitForFunction(()=>document.body.dataset.replayRecord==='single');
  assert.equal(await page.locator('#next').isVisible(),false);
  await page.locator('#replay-single [data-replay-alt=local]').click();
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  assert.equal(await page.locator('#timeline').inputValue(),'0');
  await page.locator('#next').click();
  assert.equal(await page.locator('#timeline').inputValue(),'1');
  await reveal(page,'persona');
  await page.locator('#person').selectOption('12');
  assert.equal(await page.locator('#incident').isChecked(),true);
  await page.locator('#clear-person').click();assert.equal(await page.locator('#person').inputValue(),'');

  // A valid empty graph replays as "no ties"; the filter finds exactly the 12 of them.
  await page.locator('#active-filters .flt-reset').click();
  await toggle(page,'graph','empty');
  await reveal(page,'results');
  assert.equal(await page.locator('#active-filters .flt-summary').innerText(),'Showing 12 of 890 valid graphs · 12 empty');
  await page.locator('#run-results .flt-row .flt-replay').first().click();
  const replay=await state(page);
  assert.equal(replay.view,'replay');assert.equal(replay.runs.find(run=>run.run_id===replay.active).edges.length,0);
  // A Global record is one response: the explanatory state replaces the transport (no dead play button),
  // and a valid empty graph still says that it has no ties.
  await page.waitForFunction(()=>document.body.dataset.replayRecord==='single');
  assert.equal(await page.locator('#replay-single-text').innerText(),'Global generation returned the whole network in one response — there is no step-by-step record to replay.');
  assert.equal(await page.locator('#play').isVisible(),false);
  await page.waitForFunction(()=>/valid run has no ties/.test(document.querySelector('#replay-note').textContent));
  assert.match(await page.locator('#replay-note').innerText(),/valid run has no ties/);

  // Failures are counted, never rendered: the gpt-5.6-luna global US-Portuguese cell lost two of eight seeds.
  await page.locator('#active-filters .flt-reset').click();
  await page.evaluate(async()=>{const {app}=await window.__layersModule();app.update({filters:{model:['gpt-5.6-luna'],method:['global'],setting:['us-portuguese'],seed:[],graph:'any',search:''}},'check');});
  assert.equal(await page.locator('#active-filters .flt-summary').innerText(),'Showing 6 of 890 valid graphs · 2 failed attempts in scope (not rendered)');
  assert.equal(await page.locator('#run-results').getAttribute('data-count'),'6');
  // Supplements never enter the primary list, whatever is searched.
  for(const id of ['supplement_20261105_v1_final_060dcd3a86d5_main_global_gpt-5.6-luna_us_portuguese_s44000','supp_complete_20261105_v1_final_060dcd3a86d5_main_global_gpt-5.6-luna_us_portuguese_s44004_a1']){
    await page.evaluate(async id=>{const {app}=await window.__layersModule();app.update({filters:{model:[],method:[],setting:[],seed:[],graph:'any',search:id}},'check');},id);
    assert.equal(await page.locator('#run-results').getAttribute('data-count'),'0');
  }
  await page.locator('#active-filters .flt-reset').click();

  await page.locator('#research-question').selectOption('rq2');assert.match(await page.locator('#rq-homophily').innerText(),/NA/);
  await reveal(page,'stack');
  const download=page.waitForEvent('download');await page.locator('#download-selection').click();
  assert.match((await download).suggestedFilename(),/primary/);
  for(const query of ['?dataset=revised_calibration','?dataset=calibration','?dataset=legacy','?view='+encodeURIComponent(JSON.stringify({runs:['calibration_v6_old']}))]){
    await page.goto(base+'/layers.html'+query);await page.waitForSelector('body[data-ready=true]');
    assert.match(await page.locator('#status').innerText(),/archived research/);
    assert.equal(await page.locator('#layers li').count(),0);
    assert.equal(await page.locator('#run-results .flt-row').count(),0,'no primary graph is offered for an archived link');
    await page.locator('#reset-view').click();await ready(page);
    assert.equal(await page.locator('#active-filters .flt-summary').innerText(),SUMMARY);
  }
  assert.ok(!requests.some(u=>/data\/(?:calibration|revised-calibration|networks\.json|presentation)/.test(u)),'Legacy payload requested');
  for(const route of ['study.html','archive.html']){await page.goto(base+'/'+route);await ready(page);assert.equal(await page.locator('#active-filters .flt-summary').innerText(),SUMMARY);}
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile overflow');
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);await page.locator('#formation-mode').click();await page.locator('#next').click();
  assert.equal(await page.locator('#timeline').inputValue(),'1');
  await page.screenshot({path:process.env.VIEWER_SCREENSHOT||join(tmpdir(),'current-study-mobile.png')});
  assert.deepEqual(errors,[]);
  console.log('PASS: current-only routing, exact 896/890/12/6 counts, coverage cell -> filters, single-graph stack, empty graph, failures in scope, supplements excluded, export, legacy rejection/reset, aliases, mobile.');
}finally{await browser.close();}
