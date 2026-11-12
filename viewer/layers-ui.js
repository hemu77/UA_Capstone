// Presentation-only helpers for layers.html: keyboard shortcuts, the shortcut
// dialog, persona stepping and tooltip placement. Everything here drives the
// existing controls (click / change events), so replay, data loading, saved
// links and provenance stay owned by layers.js.
const $ = id => document.getElementById(id);
const press = id => {const el=$(id);if(el&&!el.disabled){el.click();return true;}return false;};

// --- Shortcut dialog -------------------------------------------------------
const help=$('help');
const openHelp=()=>{if(help.open)return;typeof help.showModal==='function'?help.showModal():help.setAttribute('open','');};
$('help-open').addEventListener('click',openHelp);
help.addEventListener('click',event=>{if(event.target===help)help.close();});

// --- Persona stepping ------------------------------------------------------
function stepPersona(direction){
  const select=$('person'),count=select.options.length;
  if(count<2)return;
  // Index 0 is "Whole network"; stepping cycles through personas only.
  let index=select.selectedIndex+direction;
  if(index<1)index=count-1;
  if(index>=count)index=1;
  select.selectedIndex=index;
  select.dispatchEvent(new Event('change',{bubbles:true}));
}
$('person-prev').addEventListener('click',()=>stepPersona(-1));
$('person-next').addEventListener('click',()=>stepPersona(1));

// --- Keyboard shortcuts ----------------------------------------------------
const typing=target=>target instanceof HTMLElement&&(target.isContentEditable||['INPUT','SELECT','TEXTAREA'].includes(target.tagName));
const activates=target=>target instanceof HTMLElement&&['BUTTON','A','SUMMARY'].includes(target.tagName);
document.addEventListener('keydown',event=>{
  if(event.defaultPrevented||event.ctrlKey||event.metaKey||event.altKey)return;
  const target=event.target,onCanvas=target instanceof HTMLCanvasElement;
  if(help.open)return; // The native dialog handles Escape and focus.
  if(event.key==='?'){if(!typing(target)||target.type==='range'||target.type==='checkbox'){event.preventDefault();openHelp();}return;}
  if(typing(target))return;
  let handled=true;
  switch(event.key){
    case ' ':if(activates(target))return;press('play');break;
    // Arrow keys belong to the camera while the canvas has focus.
    case 'ArrowLeft':if(onCanvas)return;press('previous');break;
    case 'ArrowRight':if(onCanvas)return;press('next');break;
    case 'Home':if($('events').contains(target))return;press('rewind');break;
    case 'End':if($('events').contains(target))return;press('show-final');break;
    case 'c':case 'C':press($('compare-mode').getAttribute('aria-pressed')==='true'?'formation-mode':'compare-mode');break;
    case '[':stepPersona(-1);break;
    case ']':stepPersona(1);break;
    case 'f':case 'F':press('focus-person');break;
    case 'Escape':if(!$('person').value)return;press('clear-person');break;
    case '+':case '=':press('zoom-in');break;
    case '-':case '_':press('zoom-out');break;
    case '0':press('orbit');break;
    default:
      if(/^[1-6]$/.test(event.key)){const run=$('layers').querySelectorAll('.select-run')[Number(event.key)-1];if(run)run.click();else handled=false;}
      else handled=false;
  }
  if(handled)event.preventDefault();
});

// --- Tooltip placement -----------------------------------------------------
// layers.js fills the tooltip; here it only follows the pointer inside the stage.
const stage=document.querySelector('.stage'),tooltip=$('node-tooltip');
stage.addEventListener('pointermove',event=>{
  if(event.pointerType==='touch')return;
  const box=stage.getBoundingClientRect(),width=tooltip.offsetWidth||260,height=tooltip.offsetHeight||90;
  let x=event.clientX-box.left+16,y=event.clientY-box.top+16;
  if(x+width>box.width-8)x=Math.max(8,event.clientX-box.left-width-16);
  if(y+height>box.height-8)y=Math.max(8,event.clientY-box.top-height-16);
  tooltip.style.setProperty('--x',`${x}px`);tooltip.style.setProperty('--y',`${y}px`);
  tooltip.dataset.follow='true';
  stage.classList.toggle('hovering-node',!tooltip.hidden);
});
stage.addEventListener('pointerleave',()=>stage.classList.remove('hovering-node'));

// --- Timeline fill ---------------------------------------------------------
// A filled track makes the cursor position readable at a glance. Values are
// read from the control that layers.js already maintains.
const timeline=$('timeline');
const paint=()=>{const max=Number(timeline.max)||0;timeline.style.setProperty('--progress',`${max?100*Number(timeline.value)/max:100}%`);};
timeline.addEventListener('input',paint);
new MutationObserver(paint).observe($('step-label'),{childList:true,characterData:true,subtree:true});
paint();

// --- Filter summary --------------------------------------------------------
// The six run filters stay readable while the drawer is closed. Each chip
// mirrors a select that layers.js owns and opens the drawer on that field.
const chipHost=$('filter-chips'),drawer=document.querySelector('.run-drawer');
const filterFields=[['model','Model'],['method','Method'],['culture','Country framing'],['language','Language'],['seed','Repetition'],['collection','Collection']];
function paintChips(){
  chipHost.replaceChildren();
  for(const [id,name] of filterFields){
    const select=$(id),chosen=select.selectedOptions[0];if(!chosen||select.options.length<2)continue; // A single-option field is not a choice.
    const chip=document.createElement('button'),key=document.createElement('span'),value=document.createElement('strong');
    chip.type='button';chip.className='filter-chip';chip.dataset.narrowed=String(!!select.value&&id!=='collection');
    key.textContent=name;value.textContent=chosen.textContent;chip.append(key,value);
    chip.title=`Change ${name.toLowerCase()} filter`;
    chip.addEventListener('click',()=>{drawer.open=true;select.focus();select.scrollIntoView({block:'nearest'});});
    chipHost.append(chip);
  }
  const preview=$('filter-preview').textContent.match(/^\d+ saved runs? match/);
  if(preview){const count=document.createElement('span');count.className='filter-count';count.textContent=preview[0];chipHost.append(count);}
}
new MutationObserver(paintChips).observe($('filter-preview'),{childList:true,characterData:true,subtree:true});
$('filter-form').addEventListener('change',paintChips);
paintChips();
