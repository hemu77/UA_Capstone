// Presentation-only helpers for layers.html: keyboard shortcuts, the shortcut
// dialog, persona stepping and tooltip placement. Camera and playback keys drive
// the existing controls (click / change events); run selection (1-8) goes
// through the shared store, so replay, data loading and provenance stay owned by
// layers.js and filters by layers-filters.js.
import {app} from './layers.js?v=d068babb8c4c';
import {MAX_STACK_LAYERS} from './layers-app.js?v=d068babb8c4c';
import {replayRun} from './layers-filters.js?v=d068babb8c4c';
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
  if(document.querySelector('dialog[open]'))return; // Native dialogs own Escape and focus.
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
    case '/':{const search=document.querySelector('input[data-flt-search]');if(search&&search.offsetParent!==null){search.focus();search.select();}else handled=false;break;}
    default:
      // 1-8: replay that stacked run (recorded generation order).
      if(new RegExp(`^[1-${MAX_STACK_LAYERS}]$`).test(event.key)&&app.data){
        const id=app.getState().layers[Number(event.key)-1];
        if(id)replayRun(app,id,'layers-ui');else handled=false;
      }
      else handled=false;
  }
  if(handled)event.preventDefault();
});

// --- Tooltip placement -----------------------------------------------------
// layers.js fills the tooltip; here it only follows the pointer inside the stage.
const stage=document.querySelector('.stage'),tooltip=$('node-tooltip');
stage.addEventListener('pointermove',event=>{
  if(event.pointerType==='touch')return;
  const box=stage.getBoundingClientRect(),view=$('viewport').getBoundingClientRect(),width=tooltip.offsetWidth||260,height=tooltip.offsetHeight||90;
  const left=view.left-box.left+8,right=view.right-box.left-8;
  let x=event.clientX-box.left+16,y=event.clientY-box.top+16;
  if(x+width>right)x=Math.max(left,event.clientX-box.left-width-16);
  x=Math.max(left,Math.min(x,right-width));
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
