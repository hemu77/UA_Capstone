// Run actual pointer events: a rendered canvas alone does not prove navigation.
// Stack and Replay: drag rotates, Pan mode pans, wheel zooms toward the pointer, Shift-wheel pans, arrow keys pan the
// focused canvas, disabled gestures leave the camera alone; clicking a real node selects that persona and the labels
// equal its saved neighbourhood in every plane; touch pinch/pan works without selecting anyone.
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8766';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
// Controls may sit in a collapsed dock or sheet: drive those through the DOM, exactly as their own click/change would.
const setValue=(page,id,value)=>page.evaluate(([id,value])=>{const n=document.getElementById(id);if(n.type==='checkbox')n.checked=value;else n.value=value;n.dispatchEvent(new Event(n.type==='checkbox'?'change':'input',{bubbles:true}));n.dispatchEvent(new Event('change',{bubbles:true}));},[id,value]);
const press=(page,id)=>page.evaluate(id=>document.getElementById(id).click(),id);
const positions=page=>page.locator('.persona-label').evaluateAll(nodes=>nodes.map(n=>n.style.cssText).join('|'));
// Damped camera moves glide for a moment after input; wait until the projection is still.
// The stage renders on demand, so "still" means no frame was drawn for a while.
async function settle(page){const frames=()=>page.evaluate(()=>document.querySelector('#viewport canvas').dataset.frames);let last=await frames(),quiet=0;for(let i=0;i<80&&quiet<3;i++){await page.waitForTimeout(150);const next=await frames();quiet=next===last?quiet+1:0;last=next;}return positions(page);}
const changed=(page,before)=>page.waitForFunction(previous=>[...document.querySelectorAll('.persona-label')].map(n=>n.style.cssText).join('|')!==previous,before);
try{
  const page=await browser.newPage({viewport:{width:1500,height:1050}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(`${base}/layers.html`);
  await page.waitForSelector('body[data-ready=true]');
  assert.equal(await page.locator('#viewport canvas').getAttribute('data-view'),'atlas','The Atlas is the default view');
  await setValue(page,'node-labels','all');
  for(const mode of ['compare','formation']){
    await press(page,`${mode}-mode`);
    await page.locator('#viewport').scrollIntoViewIfNeeded();
    const box=await page.locator('#viewport').boundingBox();
    const x=box.x+box.width*.55,y=box.y+box.height*.45;
    await page.mouse.move(x,y);
    let before=await positions(page);
    await page.mouse.down();await page.mouse.move(x+90,y+40,{steps:12});await page.mouse.up();
    await changed(page,before);
    console.log(`${mode}: drag rotation changes projected nodes`);
    await page.locator('#drag-mode').selectOption('pan');
    before=await positions(page);await page.mouse.move(x,y);await page.mouse.down();
    await page.mouse.move(x+70,y-45,{steps:12});await page.mouse.up();
    await changed(page,before);
    assert.equal(await page.locator('#person').inputValue(),'','Dragging must not select a persona');
    await page.locator('#drag-mode').selectOption('orbit');
    await press(page,'orbit');await page.waitForTimeout(700);
    await page.mouse.move(x+box.width*.1,y+40);
    before=await positions(page);await page.mouse.wheel(0,-500);
    await changed(page,before);
    before=await positions(page);await page.keyboard.down('Shift');await page.mouse.wheel(65,45);await page.keyboard.up('Shift');
    await changed(page,before);
    await page.locator('#viewport canvas').focus();before=await positions(page);await page.keyboard.press('ArrowRight');
    await changed(page,before);
    // Gestures disabled means the graph does not seize the user's page scrolling.
    await setValue(page,'graph-interaction',false);
    assert.equal(await page.locator('#viewport canvas').evaluate(node=>getComputedStyle(node).touchAction),'pan-y');
    const disabledBox=await page.locator('#viewport').boundingBox();
    const dx=disabledBox.x+disabledBox.width*.55,dy=disabledBox.y+disabledBox.height*.45;
    before=await settle(page);await page.mouse.move(dx,dy);await page.mouse.down();
    await page.mouse.move(dx+50,dy+20,{steps:6});await page.mouse.up();await page.mouse.wheel(0,-80);
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    assert.equal(await positions(page),before,'Disabled gestures must leave the camera unchanged');
    await setValue(page,'graph-interaction',true);
    assert.equal(await page.locator('#viewport canvas').evaluate(node=>getComputedStyle(node).touchAction),'none');
    await press(page,'orbit');await page.waitForTimeout(700);
  }
  await press(page,'compare-mode');await page.waitForTimeout(800);
  // Click the screen point a visible persona label is anchored to (its node), inside the canvas and clear of chrome.
  const point=await page.locator('.persona-label').evaluateAll(nodes=>{
    const view=document.querySelector('#viewport').getBoundingClientRect();
    const blocked=[...document.querySelectorAll('.stage-heading,.stage-tools,.stage-key,.stage-footer,#stage-overlay .g3-hud,.g3-plane')].map(e=>e.getBoundingClientRect());
    for(const node of [...nodes].reverse())if(!node.hidden){
      const r=node.getBoundingClientRect(),x=r.left-6,y=r.top+r.height/2;
      if(x>view.left+20&&x<view.right-20&&y>view.top+20&&y<view.bottom-20&&!blocked.some(b=>x>=b.left-8&&x<=b.right+8&&y>=b.top-8&&y<=b.bottom+8))return {x,y};
    }
  });
  assert.ok(point,'An actual node must be visible for hit testing');
  await page.mouse.click(point.x,point.y);
  const person=await page.locator('#person').inputValue();assert.notEqual(person,'','Click a rendered node, not only a dropdown');
  assert.equal(await page.locator('#incident').isChecked(),true);
  // The identity thread is explained once, in the scene, next to the line itself.
  assert.equal(await page.locator('#plane-labels .g3-thread').isVisible(),true);
  assert.match(await page.locator('#plane-labels .g3-thread').innerText(),new RegExp(`Persona ${person}: same persona, not a tie`));
  // Labels shown per plane must equal the persona plus its saved neighbours in that run (table of final values).
  const expected=await page.locator('#rows tr').evaluateAll((rows,selected)=>rows.map(row=>{const cell=row.children[5].textContent,ids=cell.includes(':')?cell.split(':')[1].split(',').map(s=>s.trim()).filter(s=>s&&s!=='none'):[];return [selected,...ids].sort().join(',');}).sort(),person);
  const actual=await page.locator('.persona-label').evaluateAll(nodes=>{const groups=new Map();for(const n of nodes){if(!groups.has(n.dataset.run))groups.set(n.dataset.run,[]);groups.get(n.dataset.run).push(n.dataset.person);}return [...groups.values()].map(ids=>[...new Set(ids)].sort().join(',')).sort();});
  assert.equal(expected.length,4);assert.deepEqual(actual,expected);
  await page.mouse.click(point.x,point.y);
  assert.equal(await page.locator('#person').inputValue(),'','Clicking the same node again clears selection');
  await page.mouse.click(point.x,point.y);assert.equal(await page.locator('#person').inputValue(),person);
  await press(page,'focus-person');
  await setValue(page,'node-labels','none');
  assert.equal(await page.locator(`.persona-label[data-person="${person}"]`).count(),4,'Selected identity remains labelled when other labels are off');
  assert.equal(await page.locator('.g3-thread').count(),1,'The identity thread is labelled "same persona, not a tie"');
  await press(page,'clear-person');assert.equal(await page.locator('#person').inputValue(),'');
  await press(page,'formation-mode');
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  await page.locator('#play').click();
  await page.waitForFunction(()=>Number(document.querySelector('#timeline').value)>0);
  await page.locator('#play').click();assert.match(await page.locator('#play').innerText(),/Play/);
  // Native browser touch input verifies two-finger pan/pinch without node selection.
  const touch=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  touch.on('pageerror',error=>errors.push(error.message));
  await touch.goto(`${base}/layers.html`);
  await touch.waitForSelector('body[data-ready=true]');await setValue(touch,'node-labels','all');
  const cdp=await touch.context().newCDPSession(touch);
  for(const mode of ['compare','formation']){
    await press(touch,`${mode}-mode`);await touch.waitForTimeout(700);
    await touch.locator('#viewport').scrollIntoViewIfNeeded();
    const rect=await touch.locator('#viewport').boundingBox(),cx=rect.x+rect.width/2,cy=rect.y+rect.height/2;
    const old=await positions(touch);
    await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:cx-35,y:cy,id:1},{x:cx+35,y:cy,id:2}]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:cx-45,y:cy+15,id:1},{x:cx+65,y:cy+15,id:2}]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
    await changed(touch,old);
    assert.equal(await touch.locator('#person').inputValue(),'');
  }
  await touch.screenshot({path:process.env.SHOT_PATH||join(tmpdir(),'navigation-touch.png')});
  await touch.close();
  assert.deepEqual(errors,[]);
  console.log('PASS: Stack and Replay cameras rotate/pan/zoom/keyboard; gestures can be disabled; real node selection matches every saved neighbourhood; identity labels; touch pinch/pan');
}finally{await browser.close();}
