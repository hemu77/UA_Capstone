// Workstream 5 browser check: study ledger, attribute lens / legend / store wiring, and the supplement-only panel.
// Saved data only; no model or collector access. Run: node plot-gallery-browser-check.mjs (VIEWER_URL, BROWSER_CHANNEL).
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const {chromium}=createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
const store=page=>page.evaluate(async()=>{const m=await import(document.querySelector('script[src*="layers.js"]').src);return m.app.getState();});
const click=(page,selector,index=0)=>page.evaluate(([s,i])=>{const el=document.querySelectorAll(s)[i];el.scrollIntoView({block:'center'});el.click();},[selector,index]);
try{
  const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  page.on('console',m=>{if(m.type()==='error'&&!/Failed to load resource/.test(m.text()))errors.push(m.text());});
  await page.goto(base+'/layers.html');
  await page.waitForSelector('#study-ledger .ins-ledger[data-state=ready]',{state:'attached',timeout:30000});

  // One ledger with the exact population.
  const ledger=await page.locator('#study-ledger .ins-ledger-line').textContent();
  assert.match(ledger.replace(/\s+/g,' '),/896 attempts.*890 valid graphs \(12 empty\).*6 failures.*6 supplemental graphs excluded/);
  assert.equal(await page.locator('.ins-ledger-line').count(),1,'the population is stated once');

  // Lens and legend talk only through the store.
  await click(page,'#insight-root .ins-toggle[data-focus="color:gender"]');
  await page.waitForFunction(()=>document.querySelectorAll('#legend .ins-legend-item').length===3);
  assert.equal((await store(page)).colorBy,'gender');
  await click(page,'#legend .ins-legend-item',2);
  await page.waitForFunction(()=>document.querySelector('#legend .ins-legend-item[aria-pressed=true]'));
  assert.deepEqual((await store(page)).groups,['Nonbinary']);
  await click(page,'#insight-root .ins-toggle[data-focus="lens:mixed"]');
  assert.equal((await store(page)).lens,'mixed');
  await click(page,'#insight-root .ins-toggle[data-focus="color:religion"]');
  await page.waitForFunction(()=>document.querySelectorAll('#legend .ins-legend-item').length===5);
  assert.deepEqual((await store(page)).groups,[],'changing the attribute clears isolated groups');
  assert.match(await page.locator('#insight-root .ins-reference').textContent(),/18\.4%.*1,225 possible persona pairs/);
  await click(page,'#insight-root .ins-toggle[data-focus="scope:filters"]');
  await page.waitForFunction(()=>/Aggregate of 890 filtered graphs, not a run/.test(document.querySelector('#insight-root .ins-aggregate-label')?.textContent||''));

  // Supplements: nothing fetched before the panel opens; six graphs, never primary.
  assert.ok(!requests.some(u=>/supplements\.json|saved-networks\.json|saved-plots\.json|primary-viewer\.json/.test(u)),'no supplement or monolith request at start');
  await page.evaluate(()=>{const d=document.querySelector('.ins-supp-drawer');d.scrollIntoView();d.open=true;});
  await page.waitForFunction(()=>document.querySelectorAll('.ins-supp-item').length===6,null,{timeout:20000});
  assert.equal(requests.filter(u=>u.includes('supplements.json')).length,1);
  await click(page,'.ins-supp-item',0);
  await page.waitForSelector('.ins-supp-stage canvas[data-run^="supp"]',{state:'attached'});
  const canvas=page.locator('.ins-supp-stage canvas');
  assert.equal(await canvas.getAttribute('data-nodes'),'50');
  assert.match(await page.locator('.ins-supp-context').textContent(),/slot \d+/);
  const emptyIndex=await page.evaluate(()=>[...document.querySelectorAll('.ins-supp-item')].findIndex(b=>/no ties/.test(b.textContent)));
  await click(page,'.ins-supp-item',emptyIndex);
  await page.waitForFunction(()=>document.querySelector('.ins-supp-stage canvas').dataset.edges==='0');
  const primaryIds=await page.evaluate(async()=>{const m=await import(document.querySelector('script[src*="layers.js"]').src);return m.app.data.runs.map(r=>r.run_id);});
  const shown=await page.evaluate(()=>[...document.querySelectorAll('.ins-supp-meta .ins-mono')].map(n=>n.textContent));
  assert.ok(shown.every(id=>!primaryIds.includes(id)));
  assert.equal(primaryIds.length,890);
  assert.deepEqual(errors,[]);

  // No WebGL for the supplement canvas: the tie list remains the complete graph.
  const noGpu=await browser.newPage({viewport:{width:390,height:844}});
  await noGpu.route('**/network-stage.js*',r=>r.fulfill({status:200,contentType:'text/javascript',body:'export function createNetworkStage(){throw Error("No WebGL");}'}));
  await noGpu.goto(base+'/layers.html#saved-plots');
  await noGpu.waitForFunction(()=>document.querySelectorAll('.ins-supp-item').length===6,null,{timeout:30000});
  await click(noGpu,'.ins-supp-item',0);
  await noGpu.waitForSelector('.ins-supp-fallback',{state:'attached'});
  assert.match(await noGpu.locator('.ins-supp-status').textContent(),/generated ties: \d+–\d+/);
  console.log('Ledger, attribute lens and supplement-only panel checks passed.');
}finally{await browser.close();}
