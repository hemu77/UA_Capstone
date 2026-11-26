// Interaction regressions for Claude's presentation on the final primary study.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
// The build appends ?v=<hash> to module URLs; reuse that URL so the page's own store is inspected.
const modules=()=>{window.__layersModule=()=>{const src=[...document.scripts].find(script=>/\/layers\.js(\?|$)/.test(script.src))?.src;return import(src);};};
const state=page=>page.evaluate(async()=>{const {app}=await window.__layersModule();await app.ready;return app.getState();});
const reveal=async(page,tab)=>{
  // Shell panels may sit in tabs inside a collapsible dock; open the dock, then the tab (as a user would).
  const button=page.locator(`#tab-${tab}`);if(!await button.count())return;
  if(!await button.isVisible()){const dock=await button.getAttribute('data-dock');const toggle=page.locator(`.dock-toggle[data-dock="${dock}"]`).first();if(dock&&await toggle.count()&&await toggle.isVisible()&&await toggle.getAttribute('aria-expanded')!=='true')await toggle.click();}
  if(await button.isVisible()&&await button.getAttribute('aria-selected')!=='true')await button.click();
};
try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  await page.addInitScript(modules);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/layers.html');await page.waitForSelector('#active-filters[data-ready]',{state:'attached'});
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  await page.keyboard.press('?');assert.equal(await page.locator('#help').evaluate(n=>n.open),true);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#help').evaluate(n=>n.open),false);
  await page.keyboard.press('3');
  await page.waitForFunction(()=>document.querySelectorAll('#events button').length===50);
  const third=await state(page);assert.equal(third.active,third.layers[2],'key 3 replays the third stacked run');
  await page.keyboard.press('Home');assert.equal(await page.locator('#timeline').inputValue(),'0');
  await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#timeline').inputValue(),'1');
  await page.locator('#viewport canvas').focus();
  await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#timeline').inputValue(),'1');
  await reveal(page,'persona');
  await page.locator('#person').selectOption('12');
  await page.locator('#person-next').click();assert.equal(await page.locator('#person').inputValue(),'13');
  await page.locator('#person-prev').click();assert.equal(await page.locator('#person').inputValue(),'12');
  await page.locator('#clear-person').click();
  await page.locator('#compare-mode').click();
  if(await page.locator('.stage-heading').isVisible()){
    const viewport=await page.locator('#viewport').boundingBox(),heading=await page.locator('.stage-heading').boundingBox();
    // The heading is either above the canvas or an overlay confined to its top band; it must never cover the graph area.
    assert.ok(viewport.y>=heading.y+70||heading.y+heading.height<=viewport.y+viewport.height*0.25,'Projection must reserve heading space');
  }

  // Filters are live and separate from the stack: a facet narrows the list at once, shows exactly one chip,
  // never changes the stacked layers, and unset facets never appear as chips.
  const before=(await state(page)).layers;
  const option=page.locator('#filter-root .flt-options[data-facet="model"] .flt-option[data-value="gpt-6-sol"]');
  if(!await option.isVisible())await reveal(page,'filters');
  await option.click();
  assert.equal(await option.getAttribute('aria-pressed'),'true');
  assert.deepEqual(await page.locator('#active-filters .flt-chip').allInnerTexts().then(list=>list.map(text=>text.replace(/\s*×\s*$/,''))),['Model: GPT-6 Sol']);
  assert.match(await page.locator('#active-filters .flt-summary').innerText(),/^Showing 223 of 890 valid graphs · 1 failed attempt in scope \(not rendered\)$/);
  assert.equal(await page.locator('#run-results').getAttribute('data-count'),'223');
  assert.deepEqual((await state(page)).layers,before,'Filters must not change the stacked layers');
  assert.doesNotMatch(await page.locator('#active-filters').innerText(),/All methods|All models|Apply/);
  await page.locator('#active-filters .flt-chip').click();
  assert.equal(await page.locator('#active-filters .flt-chip').count(),0);
  assert.equal(await option.getAttribute('aria-pressed'),'false');

  // Shortcuts reach all eight stack positions.
  await page.evaluate(async()=>{const {app}=await window.__layersModule();const ids=app.data.runs.filter(run=>run.model==='gpt-4.1'&&run.method==='local'&&run.setting==='us-english').sort((a,b)=>a.slot-b.slot).map(run=>run.run_id);app.update({layers:ids,active:ids[0],view:'stack'},'check');});
  await page.waitForFunction(async()=>{const {app}=await window.__layersModule();return app.getState().layers.length===8;});
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.press('8');
  const eighth=await state(page);assert.equal(eighth.active,eighth.layers[7],'key 8 replays the eighth stacked run');

  await page.setViewportSize({width:390,height:844});
  // Phone navigation may collapse behind a header menu button; the link must be one tap away.
  const research=page.getByRole('link',{name:'Research & files'});
  if(!await research.isVisible()){const menu=page.locator('header button[aria-expanded="false"]').first();if(await menu.count())await menu.click();}
  assert.equal(await research.isVisible(),true);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No mobile overflow');
  assert.doesNotMatch(await page.locator('body').innerText(),/Historical calibration replay/);
  assert.deepEqual(errors,[]);
  console.log('PASS: shortcuts/modal, keys 3 and 8, persona stepping, camera-arrow ownership, live filters separate from the stack, chips only for set values, overlay-safe projection, mobile navigation.');
}finally{await browser.close();}
