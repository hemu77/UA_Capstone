// WORKSTREAM 4: page shell for layers.html (docs/qa/viewer-3d-2026-11-25/SPEC.md, section C.6).
// Owns panel open / close state, the tab strip below 1150 px, the phone bottom sheet below 700 px,
// sticky offsets (--sticky-top, --chrome-h, --tabbar-h), focus return, the header menu, the toast that
// shows the one live region, and body[data-view] mirroring state.view.
// It never reads or writes study data and never renders inside another module's mount root.

export const BREAKPOINTS=Object.freeze({tabs:1150,sheet:700,bothDocks:1600});
export const PANELS=Object.freeze([
  {name:'study',dock:'left',title:'Study ledger'},
  {name:'filters',dock:'left',title:'Filters'},
  {name:'results',dock:'left',title:'Results'},
  {name:'stack',dock:'right',title:'Stacked runs'},
  {name:'persona',dock:'right',title:'Persona'},
  {name:'compare',dock:'right',title:'Compare'}
]);
const DOCK_TABS=Object.freeze({left:['filters','results'],right:['stack','persona','compare']});
const PREFS_KEY='layers-shell:v1';
const TOAST_MS=6000;

// Pure helpers (unit-tested in current-study.test.js).
export function layoutMode(width) {return width<BREAKPOINTS.sheet?'sheet':width<BREAKPOINTS.tabs?'tabs':'desktop';}
export function defaultDocks(width) {return {left:true,right:width>=BREAKPOINTS.bothDocks};}
export function panelDock(name) {const panel=PANELS.find(p=>p.name===name);if(!panel)throw new Error(`Unknown panel: ${name}`);return panel.dock;}
// Stored preferences are a per-viewer convenience: anything malformed falls back to the defaults.
export function parsePrefs(text,width) {
  const fallback={docks:defaultDocks(width),tabs:{left:'filters',right:'stack'}};
  let value;try{value=JSON.parse(text);}catch{return fallback;}
  if(!value||typeof value!=='object')return fallback;
  const docks={...fallback.docks},tabs={...fallback.tabs};
  for(const side of ['left','right']){
    if(typeof value.docks?.[side]==='boolean')docks[side]=value.docks[side];
    if(DOCK_TABS[side].includes(value.tabs?.[side]))tabs[side]=value.tabs[side];
  }
  return {docks,tabs};
}
const storage={
  read(){try{return globalThis.localStorage?.getItem(PREFS_KEY)??null;}catch{return null;}},
  write(value){try{globalThis.localStorage?.setItem(PREFS_KEY,JSON.stringify(value));}catch{/* private mode: keep in memory */}}
};

let mounted=null;

export function mountShell(app) {
  if(mounted)return mounted.handle; // layers.html and layers.js may both mount; the first wins.
  if(typeof document==='undefined')return {destroy(){}};
  const $=id=>document.getElementById(id);
  const body=document.body,root=document.documentElement;
  const header=$('shell-header'),filterBar=$('active-filters'),workspace=$('workspace'),bar=$('shell-tabs');
  if(!header||!workspace||!bar)return {destroy(){}};
  const docks={left:$('dock-left'),right:$('dock-right')};
  const tabs=new Map([...document.querySelectorAll('.shell-tab[data-panel]')].map(tab=>[tab.dataset.panel,tab]));
  const panels=new Map([...document.querySelectorAll('.shell-panel[data-panel]')].map(panel=>[panel.dataset.panel,panel]));
  const homes={left:docks.left.querySelector('.dock-tabs'),right:docks.right.querySelector('.dock-tabs')};
  const cleanup=[];
  const on=(target,type,fn,options)=>{target.addEventListener(type,fn,options);cleanup.push(()=>target.removeEventListener(type,fn,options));};

  const prefs=parsePrefs(storage.read(),innerWidth);
  const shell={mode:layoutMode(innerWidth),docks:prefs.docks,tabs:prefs.tabs,sheet:null};
  const save=()=>storage.write({docks:shell.docks,tabs:shell.tabs});

  // --- Sticky offsets ---------------------------------------------------------
  const measure=()=>{
    const headerHeight=header.offsetHeight,barHeight=filterBar&&getComputedStyle(filterBar).display!=='none'?filterBar.offsetHeight:0;
    root.style.setProperty('--sticky-top',`${headerHeight}px`);
    root.style.setProperty('--chrome-h',`${headerHeight+barHeight}px`);
    root.style.setProperty('--tabbar-h',shell.mode==='sheet'?`${bar.offsetHeight}px`:'0px');
  };
  const sizes=new ResizeObserver(measure);
  for(const element of [header,filterBar,bar])if(element)sizes.observe(element);
  // An empty mount root is hidden by CSS; a mutation can change that without a resize of an observed box.
  const contentWatch=new MutationObserver(measure);if(filterBar)contentWatch.observe(filterBar,{childList:true});

  // --- Tabs and panels -------------------------------------------------------
  const order=PANELS.map(p=>p.name);
  function placeTabs() {
    if(shell.mode==='desktop')for(const side of ['left','right'])for(const name of (side==='left'?['study',...DOCK_TABS.left]:DOCK_TABS.right))homes[side].append(tabs.get(name));
    else for(const name of order)bar.append(tabs.get(name));
  }
  const visiblePanel=name=>shell.mode==='desktop'?(name==='study'?shell.docks.left:shell.docks[panelDock(name)]&&shell.tabs[panelDock(name)]===name):shell.sheet===name;
  function apply() {
    body.dataset.shell=shell.mode;
    body.dataset.sheet=shell.mode!=='desktop'&&shell.sheet?'open':'closed';
    for(const [name,panel] of panels){
      const shown=visiblePanel(name);panel.hidden=!shown;
      const tab=tabs.get(name);if(!tab)continue;
      const selected=shell.mode==='desktop'?(name==='study'?false:shell.tabs[panelDock(name)]===name):shell.sheet===name;
      tab.setAttribute('aria-selected',String(selected));
      if(shown)delete tab.dataset.attention;
    }
    for(const side of ['left','right']){
      const open=shell.mode==='desktop'?shell.docks[side]:!!shell.sheet&&panelDock(shell.sheet)===side;
      docks[side].dataset.open=String(open);
      const toggle=docks[side].querySelector('.dock-toggle'),label=side==='left'?'filters and results':'stack, persona and comparison';
      toggle.setAttribute('aria-expanded',String(shell.docks[side]));
      toggle.setAttribute('aria-label',`${shell.docks[side]?'Hide':'Show'} ${label}`);
      toggle.title=toggle.getAttribute('aria-label');
      if(shell.mode!=='desktop'&&open){const title=docks[side].querySelector('.sheet-title');title.textContent=PANELS.find(p=>p.name===shell.sheet).title;}
    }
    // Roving tabindex: one tab stop per tablist, on the selected (or first) tab.
    for(const list of [bar,homes.left,homes.right]){
      const items=[...list.querySelectorAll('.shell-tab')].filter(tab=>getComputedStyle(tab).display!=='none');
      const current=items.find(tab=>tab.getAttribute('aria-selected')==='true')||items[0];
      for(const tab of list.querySelectorAll('.shell-tab'))tab.tabIndex=tab===current?0:-1;
    }
    measure();
    document.dispatchEvent(new CustomEvent('shell:layout',{detail:{mode:shell.mode,docks:{...shell.docks},sheet:shell.sheet}}));
  }
  function openPanel(name,{focus=false}={}) {
    if(!panels.has(name))return;
    if(shell.mode==='desktop'){
      const side=panelDock(name);shell.docks[side]=true;
      if(name!=='study')shell.tabs[side]=name;
      leaveExpanded();save();apply();
      if(focus)(name==='study'?panels.get(name):tabs.get(name)).focus({preventScroll:false});
    }else{
      shell.sheet=name;apply();
      if(focus)docks[panelDock(name)].querySelector('.sheet-title').focus({preventScroll:true});
    }
  }
  function closeSheet({returnFocus=true}={}) {
    if(shell.mode==='desktop'||!shell.sheet)return;
    const tab=tabs.get(shell.sheet);shell.sheet=null;apply();
    if(returnFocus)tab?.focus({preventScroll:true});
  }
  function toggleDock(side) {
    shell.docks[side]=!shell.docks[side];
    const leaving=!shell.docks[side]&&docks[side].contains(document.activeElement);
    leaveExpanded();save();apply();
    if(leaving)docks[side].querySelector('.dock-toggle').focus({preventScroll:true});
  }
  for(const tab of tabs.values()){
    on(tab,'click',()=>{
      const name=tab.dataset.panel;
      if(shell.mode!=='desktop'&&shell.sheet===name){closeSheet();return;}
      openPanel(name,{focus:shell.mode!=='desktop'});
      if(shell.mode==='desktop')tab.focus({preventScroll:true});
    });
  }
  // Arrow keys move between tabs of the same list (manual activation: Enter / Space opens).
  for(const list of [bar,homes.left,homes.right])on(list,'keydown',event=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key))return;
    const items=[...list.querySelectorAll('.shell-tab')].filter(tab=>getComputedStyle(tab).display!=='none'),index=items.indexOf(document.activeElement);
    if(index<0)return;
    const step=['ArrowLeft','ArrowUp'].includes(event.key)?-1:1;
    const next=event.key==='Home'?items[0]:event.key==='End'?items.at(-1):items[(index+step+items.length)%items.length];
    event.preventDefault();event.stopPropagation();
    for(const tab of items)tab.tabIndex=tab===next?0:-1;
    next.focus();
  });
  for(const side of ['left','right']){
    on(docks[side].querySelector('.dock-toggle'),'click',()=>toggleDock(side));
    on(docks[side].querySelector('.dock-close'),'click',()=>closeSheet());
    // Escape closes the drawer / sheet before anything else (layers-ui.js skips prevented events).
    on(docks[side],'keydown',event=>{if(event.key==='Escape'&&shell.mode!=='desktop'&&shell.sheet){event.preventDefault();event.stopPropagation();closeSheet();}});
  }
  // Swipe down on the sheet head closes the phone sheet.
  for(const side of ['left','right']){
    const head=docks[side].querySelector('.sheet-head');let start=null;
    on(head,'pointerdown',event=>{if(shell.mode==='sheet')start={y:event.clientY,id:event.pointerId};});
    on(head,'pointerup',event=>{if(start&&event.pointerId===start.id&&event.clientY-start.y>56)closeSheet();start=null;});
    on(head,'pointercancel',()=>{start=null;});
  }

  // Links to a panel (header "Study ledger", skip links) open it instead of jumping to a hidden box.
  on(document,'click',event=>{
    const link=event.target instanceof Element&&event.target.closest('a[href^="#"]');if(!link)return;
    const target=document.getElementById(decodeURIComponent(link.getAttribute('href').slice(1)));
    const panel=target?.closest('.shell-panel');if(!panel)return;
    event.preventDefault();closeMenu();
    openPanel(panel.dataset.panel,{focus:true});
  });

  // --- Header menu (phone) ----------------------------------------------------
  const menuToggle=$('menu-toggle'),menu=$('site-menu');
  function closeMenu({returnFocus=false}={}) {if(menuToggle?.getAttribute('aria-expanded')!=='true')return;menuToggle.setAttribute('aria-expanded','false');menu.dataset.open='false';if(returnFocus)menuToggle.focus();}
  if(menuToggle&&menu){
    on(menuToggle,'click',()=>{const open=menuToggle.getAttribute('aria-expanded')!=='true';menuToggle.setAttribute('aria-expanded',String(open));menu.dataset.open=String(open);if(open)menu.querySelector('a,button')?.focus();});
    on(menu,'keydown',event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();closeMenu({returnFocus:true});}});
    on(document,'pointerdown',event=>{if(!menu.contains(event.target)&&!menuToggle.contains(event.target))closeMenu();});
    on(menu,'click',event=>{if(event.target instanceof Element&&event.target.closest('a,button'))closeMenu();});
    on(menu,'focusout',event=>{if(getComputedStyle(menuToggle).display!=='none'&&!menu.contains(event.relatedTarget)&&event.relatedTarget!==menuToggle)closeMenu();});
  }

  // --- Dialog focus return ----------------------------------------------------
  // Native modal dialogs trap focus; this returns it to the control that opened them.
  let lastFocus=null;
  on(document,'focusin',event=>{if(!(event.target instanceof Element)||!event.target.closest('dialog'))lastFocus=event.target;});
  for(const dialog of document.querySelectorAll('dialog'))on(dialog,'close',()=>{
    if(document.querySelector('dialog[open]'))return;
    // An opener inside the closed phone menu is not focusable; its menu button stands in for it.
    const target=lastFocus?.isConnected&&lastFocus.getClientRects().length?lastFocus:menu?.contains(lastFocus)?menuToggle:null;
    target?.focus({preventScroll:true});
  });

  // --- Toast for the one live region -------------------------------------------
  const status=$('action-status');let toastTimer=0;
  const toastWatch=new MutationObserver(()=>{
    clearTimeout(toastTimer);
    if(!status.textContent.trim()){delete status.dataset.fresh;return;}
    status.dataset.fresh='true';toastTimer=setTimeout(()=>{delete status.dataset.fresh;},TOAST_MS);
  });
  if(status){toastWatch.observe(status,{childList:true,characterData:true,subtree:true});on(status,'click',()=>{clearTimeout(toastTimer);delete status.dataset.fresh;});}

  // --- Store mirror -----------------------------------------------------------
  const view=state=>{if(state?.view)body.dataset.view=state.view;};
  view(app?.getState?.());
  const unsubscribe=app?.subscribe?.((state,changed)=>{
    if(changed.has('view'))view(state);
    // A choice made inside the drawer / sheet (Stack these, Replay, a Match button) closes it so the stage shows the result.
    if(shell.mode!=='desktop'&&shell.sheet&&['layers','view','active'].some(key=>changed.has(key))&&docks[panelDock(shell.sheet)].contains(document.activeElement))closeSheet();
    // A persona picked on the canvas marks the Persona tab while its panel is out of sight.
    if(changed.has('persona')&&state.persona&&!visiblePanel('persona'))tabs.get('persona').dataset.attention='true';
    if(changed.has('persona')&&!state.persona)delete tabs.get('persona').dataset.attention;
  })||(()=>{});

  // --- Breakpoints ------------------------------------------------------------
  const queries=[matchMedia(`(max-width: ${BREAKPOINTS.sheet-0.02}px)`),matchMedia(`(max-width: ${BREAKPOINTS.tabs-0.02}px)`)];
  const relayout=()=>{
    const mode=queries[0].matches?'sheet':queries[1].matches?'tabs':'desktop';
    if(mode===shell.mode&&body.dataset.shell)return;
    const hadFocus=[...tabs.values()].includes(document.activeElement)?document.activeElement:null;
    shell.mode=mode;shell.sheet=null;closeMenu();placeTabs();apply();hadFocus?.focus({preventScroll:true});
  };
  for(const query of queries)on(query,'change',relayout);
  // The legacy "Expand graph" button collapses both docks on desktop; keep the toggles truthful.
  let beforeExpand=null;
  const expandWatch=new MutationObserver(()=>{
    const expanded=workspace.classList.contains('expanded');
    if(expanded&&!beforeExpand){beforeExpand={...shell.docks};shell.docks={left:false,right:false};apply();}
    else if(!expanded&&beforeExpand){shell.docks=beforeExpand;beforeExpand=null;apply();}
  });
  // Opening a dock while the graph is expanded un-expands through layers.js, so its button stays truthful.
  function leaveExpanded() {if(!workspace.classList.contains('expanded'))return;beforeExpand=null;$('expand-stage')?.click();workspace.classList.remove('expanded');}
  expandWatch.observe(workspace,{attributes:true,attributeFilter:['class']});

  body.dataset.shell='';relayout();

  const handle={destroy(){
    for(const fn of cleanup.splice(0))fn();
    sizes.disconnect();contentWatch.disconnect();toastWatch.disconnect();expandWatch.disconnect();clearTimeout(toastTimer);unsubscribe();
    shell.mode='desktop';placeTabs();
    for(const panel of panels.values())panel.hidden=false;
    for(const side of ['left','right'])docks[side].dataset.open='true';
    for(const key of ['shell','sheet'])delete body.dataset[key];
    for(const name of ['--sticky-top','--chrome-h','--tabbar-h'])root.style.removeProperty(name);
    mounted=null;
  }};
  mounted={handle};
  return handle;
}
