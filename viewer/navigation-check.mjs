// Run actual pointer events: a rendered canvas alone does not prove navigation.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
try{
  const page=await browser.newPage({viewport:{width:1500,height:1050}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(`${process.env.VIEWER_URL||'http://127.0.0.1:8766'}/layers.html`);
  await page.waitForSelector('#layers button');
  assert.ok((await page.locator('#play').boundingBox()).y<(await page.locator('.laboratory').boundingBox()).y,'Playback must precede the graph');
  await page.locator('#open-filters').click();
  assert.equal(await page.locator('.run-drawer').getAttribute('open'),'');
  assert.equal(await page.locator('#model').evaluate(node=>node===document.activeElement),true);
  await page.locator('.run-drawer summary').click();
  await page.locator('#node-labels').selectOption('all');
  const positions=()=>page.locator('.persona-label').evaluateAll(nodes=>nodes.map(n=>n.style.cssText).join('|'));
  for(const mode of ['compare','formation']){
    await page.locator(`#${mode}-mode`).click();
    await page.locator('#viewport').scrollIntoViewIfNeeded();
    const box=await page.locator('#viewport').boundingBox();
    const x=box.x+box.width*.15,y=box.y+box.height*.2;
    await page.mouse.move(x,y);
    let before=await positions();
    await page.mouse.down();await page.mouse.move(x+90,y+40,{steps:12});await page.mouse.up();
    await page.waitForFunction(previous=>[...document.querySelectorAll('.persona-label')].map(n=>n.style.cssText).join('|')!==previous,before);
    console.log(`${mode}: drag rotation changes projected nodes`);
    await page.locator('#drag-mode').selectOption('pan');
    before=await positions();await page.mouse.move(x,y);await page.mouse.down();
    await page.mouse.move(x+70,y-45,{steps:12});await page.mouse.up();
    await page.waitForFunction(previous=>[...document.querySelectorAll('.persona-label')].map(n=>n.style.cssText).join('|')!==previous,before);
    assert.equal(await page.locator('#person').inputValue(),'','Dragging must not select a persona');
    await page.locator('#drag-mode').selectOption('orbit');
    await page.locator('#orbit').click();
    await page.locator('#viewport').scrollIntoViewIfNeeded();
    await page.mouse.move(x+box.width*.25,y+40);
    before=await positions();await page.mouse.wheel(0,-500);
    await page.waitForFunction(previous=>[...document.querySelectorAll('.persona-label')].map(n=>n.style.cssText).join('|')!==previous,before);
    before=await positions();await page.keyboard.down('Shift');await page.mouse.wheel(65,45);await page.keyboard.up('Shift');
    await page.waitForFunction(previous=>[...document.querySelectorAll('.persona-label')].map(n=>n.style.cssText).join('|')!==previous,before);
    await page.locator('#viewport canvas').focus();before=await positions();await page.keyboard.press('ArrowRight');
    await page.waitForFunction(previous=>[...document.querySelectorAll('.persona-label')].map(n=>n.style.cssText).join('|')!==previous,before);
    // Gestures disabled means the graph does not seize the user's page scrolling.
    await page.locator('#graph-interaction').uncheck();
    assert.equal(await page.locator('#viewport canvas').evaluate(node=>getComputedStyle(node).touchAction),'pan-y');
    await page.locator('#viewport').scrollIntoViewIfNeeded();
    const disabledBox=await page.locator('#viewport').boundingBox();
    const dx=disabledBox.x+disabledBox.width*.15,dy=disabledBox.y+disabledBox.height*.2;
    before=await positions();await page.mouse.move(dx,dy);await page.mouse.down();
    await page.mouse.move(dx+50,dy+20,{steps:6});await page.mouse.up();await page.mouse.wheel(0,-80);
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    assert.equal(await positions(),before,'Disabled gestures must leave the camera unchanged');
    await page.locator('#graph-interaction').check();
    assert.equal(await page.locator('#viewport canvas').evaluate(node=>getComputedStyle(node).touchAction),'none');
    await page.locator('#orbit').click();
  }
  await page.locator('#compare-mode').click();await page.locator('#viewport').scrollIntoViewIfNeeded();
  const point=await page.locator('.persona-label').evaluateAll(nodes=>{
    const viewport=document.querySelector('#viewport').getBoundingClientRect();
    for(const node of [...nodes].reverse())if(!node.hidden){
      const x=viewport.x+parseFloat(node.style.left)*viewport.width/100,y=viewport.y+parseFloat(node.style.top)*viewport.height/100;
      if(x>viewport.x+20&&x<viewport.right-20&&y>viewport.y+20&&y<viewport.bottom-20)return {x,y};
    }
  });
  assert.ok(point,'An actual node must be visible for hit testing');
  await page.mouse.click(point.x,point.y);
  const person=await page.locator('#person').inputValue();assert.notEqual(person,'','Click a rendered node, not only a dropdown');
  assert.equal(await page.locator('#incident').isChecked(),true);
  assert.match(await page.locator('.canvas-legend').innerText(),new RegExp(`persona ${person} across 4 runs`));
  const expected=await page.evaluate(async selected=>{
    const data=await (await fetch('./data/revised-calibration.json')).json();
    const view=JSON.parse(new URL(location.href).searchParams.get('view'));
    return view.runs.map(id=>{const run=data.runs.find(r=>r.run_id===id);return {id,people:[selected,...run.edges.filter(edge=>edge.includes(selected)).flat().filter(id=>id!==selected)].sort()};});
  },person);
  for(const run of expected){
    const actual=await page.locator(`.persona-label[data-run="${run.id}"]`).evaluateAll(nodes=>nodes.map(node=>node.dataset.person).sort());
    assert.deepEqual(actual,[...new Set(run.people)].sort());
  }
  await page.mouse.click(point.x,point.y);
  assert.equal(await page.locator('#person').inputValue(),'','Clicking the same node again clears selection');
  await page.mouse.click(point.x,point.y);assert.equal(await page.locator('#person').inputValue(),person);
  await page.locator('#focus-person').click();
  await page.locator('#node-labels').selectOption('none');
  assert.equal(await page.locator(`.persona-label[data-person="${person}"]`).count(),4,'Selected identity remains labelled when other labels are off');
  await page.locator('#clear-person').click();assert.equal(await page.locator('#person').inputValue(),'');
  await page.locator('#formation-mode').click();
  await page.locator('#play').click();
  await page.waitForFunction(()=>Number(document.querySelector('#timeline').value)>0);
  await page.locator('#play').click();assert.match(await page.locator('#play').innerText(),/Play/);
  // Native browser touch input verifies two-finger pan/pinch without node selection.
  const touch=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  touch.on('pageerror',error=>errors.push(error.message));
  await touch.goto(`${process.env.VIEWER_URL||'http://127.0.0.1:8766'}/layers.html`);
  await touch.waitForSelector('#layers button');await touch.locator('#node-labels').selectOption('all');
  const cdp=await touch.context().newCDPSession(touch);
  for(const mode of ['compare','formation']){
    await touch.locator(`#${mode}-mode`).click();
    await touch.locator('#viewport').scrollIntoViewIfNeeded();
    const rect=await touch.locator('#viewport').boundingBox(),cx=rect.x+rect.width/2,cy=rect.y+rect.height/2;
    const old=await touch.locator('.persona-label').evaluateAll(nodes=>nodes.map(n=>n.style.cssText).join('|'));
    await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:cx-35,y:cy,id:1},{x:cx+35,y:cy,id:2}]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:cx-45,y:cy+15,id:1},{x:cx+65,y:cy+15,id:2}]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
    await touch.waitForFunction(previous=>[...document.querySelectorAll('.persona-label')].map(n=>n.style.cssText).join('|')!==previous,old);
    assert.equal(await touch.locator('#person').inputValue(),'');
  }
  await touch.screenshot({path:'../outputs/qa/navigation-touch.png'});
  await touch.close();
  assert.deepEqual(errors,[]);
  console.log('PASS: both cameras rotate/pan/zoom/keyboard; real node selection matches all saved neighborhoods; identity labels; visible filters/playback');
}finally{await browser.close();}
