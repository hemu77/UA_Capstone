// WORKSTREAM 2 (3D engine). One DOM label layer for every 3D view (SPEC.md B.3 "Labels", C.4).
// - Elements are pooled by id and reused across updates; text and size are measured once per set(), never per frame.
// - project() only writes transforms and the hidden flag; it never reads layout, so orbiting cannot force reflow.
// - Collision culling is greedy by priority: items with cull:false always show; others hide when they overlap.
import {Vector3} from 'three';

const scratch=new Vector3();
const overlaps=(a,b)=>a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;

// Pure: screen boxes in priority order -> Set of visible ids. Exported for unit tests.
export function cullBoxes(boxes) {
  const placed=[],visible=new Set();
  for(const box of [...boxes].sort((a,b)=>(b.priority-a.priority)||(a.order-b.order))){
    if(box.offscreen)continue;
    if(box.cull!==false&&placed.some(other=>overlaps(box,other)))continue;
    placed.push(box);visible.add(box.id);
  }
  return visible;
}

// Anchor offsets as fractions of the element box, applied after the projected point.
const ANCHORS={right:[0,-.5,6,0],left:[-1,-.5,-6,0],above:[-.5,-1,0,-6],below:[-.5,0,0,6],center:[-.5,-.5,0,0]};

export function createLabels(host,{onRemeasure=null}={}) {
  const pool=new Map();let entries=[],dirty=true,generation=0;
  function element(item) {
    let entry=pool.get(item.id);
    const tag=item.onClick?'button':'span';
    if(entry&&entry.el.tagName.toLowerCase()!==tag){entry.el.remove();pool.delete(item.id);entry=null;}
    if(!entry){
      const el=document.createElement(tag);
      if(tag==='button')el.type='button';
      el.hidden=true;host.append(el);
      entry={el,w:0,h:0,x:NaN,y:NaN,visible:false,text:null,className:null,leader:-1};
      pool.set(item.id,entry);
    }
    return entry;
  }
  // Text is measured when it changes. A web font that finishes loading later changes every width, so measure the
  // pooled labels once more then (one batched read, outside any animation frame) and re-project.
  function measureAll(){
    const list=[...pool.values()].filter(entry=>entry.el.isConnected);
    for(const entry of list){entry.el.hidden=false;entry.el.style.visibility='hidden';}
    for(const entry of list){entry.w=entry.el.offsetWidth;entry.h=entry.el.offsetHeight;}
    for(const entry of list){entry.el.style.visibility='';entry.el.hidden=!entry.visible;}
    dirty=true;
  }
  globalThis.document?.fonts?.ready?.then(()=>{if(pool.size)measureAll();onRemeasure?.();});
  return {
    // items: [{id,text,position:[x,y,z],priority,className,anchor,cull,title,ariaLabel,color,data:{},onClick,clamp,column}]
    // anchor: one name or a list in order of preference (the first whose box fits the canvas wins).
    // clamp: keep the label inside the canvas instead of hiding it when it would be clipped.
    // column: labels sharing a column name line up on the leftmost projected x of the group (axis tick column).
    // screenOffset: [dx,dy] in pixels applied after projection.
    // leader: draw a hairline from the label to its own anchor point (used with column, so aligned labels still point
    //         at what they name). The length is written as the --leader custom property; no layout is read.
    set(items) {
      const seen=new Set(),measure=[];generation++;
      entries=items.map((item,order)=>{
        seen.add(item.id);
        const entry=element(item),el=entry.el;
        const className=`g3-label ${item.className||''}`.trim();
        if(entry.className!==className){el.className=className;entry.className=className;}
        if(entry.text!==item.text){el.textContent=item.text;entry.text=item.text;measure.push(entry);}
        if(item.title!==undefined)el.title=item.title;else el.removeAttribute('title');
        if(item.ariaLabel)el.setAttribute('aria-label',item.ariaLabel);else el.removeAttribute('aria-label');
        el.style.borderColor=item.color||'';
        if(item.color)el.style.setProperty('--hue',item.color);else el.style.removeProperty('--hue');
        for(const key of Object.keys(el.dataset))if(!(item.data&&key in item.data))delete el.dataset[key];
        for(const [key,value] of Object.entries(item.data||{}))el.dataset[key]=value;
        el.onclick=item.onClick?event=>{event.stopPropagation();item.onClick(event);}:null;
        if(item.pressed!==undefined)el.setAttribute('aria-pressed',String(item.pressed));
        entry.item=item;entry.order=order;entry.position=item.position;entry.anchors=(Array.isArray(item.anchor)?item.anchor:[item.anchor||'right']).map(name=>ANCHORS[name]);
        return entry;
      });
      for(const [id,entry] of pool)if(!seen.has(id)){entry.el.remove();pool.delete(id);}
      // One batched read for new or changed text: elements must be visible (not display:none) to have a size.
      if(measure.length){
        for(const entry of measure){entry.el.hidden=false;entry.el.style.visibility='hidden';}
        for(const entry of measure){entry.w=entry.el.offsetWidth;entry.h=entry.el.offsetHeight;}
        for(const entry of measure){entry.el.style.visibility='';entry.el.hidden=!entry.visible;}
      }
      dirty=true;
    },
    get dirty(){return dirty;},
    markDirty(){dirty=true;},
    // size: {width,height,offsetX,offsetY} of the canvas relative to the label host. No layout reads here.
    project(camera,size) {
      dirty=false;
      const boxes=[],columns=new Map(),x0=size.offsetX||0,y0=size.offsetY||0;
      for(const entry of entries){
        const [x,y,z]=entry.position;
        scratch.set(x,y,z).project(camera);
        entry.behind=scratch.z>1||scratch.z<-1;
        entry.px=(scratch.x+1)*size.width/2+x0+(entry.item.screenOffset?.[0]||0);entry.py=(1-scratch.y)*size.height/2+y0+(entry.item.screenOffset?.[1]||0);
        const column=entry.item.column;
        if(column&&!entry.behind)columns.set(column,Math.min(columns.get(column)??Infinity,entry.px));
      }
      for(const entry of entries){
        const px=entry.item.column&&columns.has(entry.item.column)?columns.get(entry.item.column)-(entry.item.leader?14:0):entry.px,py=entry.py;
        const fits=(l,t)=>!(l<x0||t<y0||l+entry.w>x0+size.width||t+entry.h>y0+size.height);
        let left=0,top=0,outside=true;
        for(const [ax,ay,dx,dy] of entry.anchors){left=px+ax*entry.w+dx;top=py+ay*entry.h+dy;if(fits(left,top)){outside=false;break;}}
        // A label shows only when its whole box fits on the canvas: a clipped label reads as a different word.
        // Labels marked clamp (plane buttons) slide inside the canvas edge instead.
        if(outside&&entry.item.clamp&&entry.w<=size.width&&entry.h<=size.height){
          left=Math.min(Math.max(left,x0+2),x0+size.width-entry.w-2);top=Math.min(Math.max(top,y0+2),y0+size.height-entry.h-2);outside=false;
        }
        const offscreen=entry.behind;
        entry.left=left;entry.top=top;
        boxes.push({id:entry,x:left,y:top,w:entry.w+4,h:entry.h+2,priority:entry.item.priority||0,order:entry.order,cull:entry.item.cull,offscreen:offscreen||outside});
      }
      const visible=cullBoxes(boxes);
      for(const entry of entries){
        const show=visible.has(entry);
        if(show){
          const x=Math.round(entry.left*2)/2,y=Math.round(entry.top*2)/2;
          if(x!==entry.x||y!==entry.y){entry.el.style.transform=`translate3d(${x}px,${y}px,0)`;entry.x=x;entry.y=y;}
          if(entry.item.leader){const length=Math.max(0,Math.round(entry.px-(x+entry.w)));if(length!==entry.leader){entry.el.style.setProperty('--leader',`${length}px`);entry.leader=length;}}
        }
        if(show!==entry.visible){entry.el.hidden=!show;entry.visible=show;}
      }
      return generation;
    },
    // Screen position (relative to the label host) of a visible label, for tests and focus helpers.
    positionOf(id){const entry=pool.get(id);return entry&&entry.visible?{x:entry.x,y:entry.y,w:entry.w,h:entry.h}:null;},
    clear(){this.set([]);},
    dispose(){for(const entry of pool.values())entry.el.remove();pool.clear();entries=[];}
  };
}
