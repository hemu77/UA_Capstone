import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {layoutMode, defaultDocks, parsePrefs, panelDock, PANELS, BREAKPOINTS, mountShell} from './layers-shell.js';

const html=readFileSync(new URL('./layers.html',import.meta.url),'utf8');
const ids=[...html.matchAll(/\sid="([^"]+)"/g)].map(match=>match[1]);
// Visible static text: no tags, comments, scripts or the hidden legacy block.
const visibleText=html.replace(/<!--[\s\S]*?-->/g,' ').replace(/<script[\s\S]*?<\/script>/g,' ').replace(/<[^>]+>/g,' ').replace(/\s+/g,' ');

test('current project surfaces retain the established research name',()=>{
  const name='Cross-Cultural and Linguistic LLM Network Study';
  const readme=readFileSync(new URL('../README.md',import.meta.url),'utf8');
  const deployment=readFileSync(new URL('../docs/SIMULATOR_DEPLOYMENT.md',import.meta.url),'utf8');
  assert.equal(readme.split(/\r?\n/)[0],`# ${name}`);
  assert.ok(html.includes(`<title>${name} / Recorded network lab</title>`));
  assert.ok(html.includes(`aria-label="${name}"`));
  assert.ok(html.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').includes(name));
  for(const text of [readme,html,deployment])assert.doesNotMatch(text,/Network Observatory/i);
});

test('public workspace offers current study only and links the preserved audit',()=>{
  assert.doesNotMatch(html,/id="(?:dataset|collection|failure-panel|failure-list)"/);
  assert.doesNotMatch(html,/revised_calibration|Historical V5|engineering pilots/);
  assert.match(html,/NETWORKS\.md/);
  assert.match(html,/id="saved-plots"/);
});

// SPEC.md C.1: these ids are read by layers.js, layers-ui.js, plot-gallery.js or the browser checks.
const FROZEN=['help-open','share','dataset-summary','download-selection','study-ledger','active-filters','filter-root','run-results','retry-load','reset-view','evidence-notice','coverage-panel','coverage-table','match-note','atlas-mode','formation-mode','compare-mode','rewind','previous','play','next','step-label','playback-scope','timeline','speed','show-final','drag-mode','focus-person','clear-person','layer-count','layers','undo','restore','action-status','graph-interaction','color','node-labels','spacing-value','separation','ties','opacity','comparison-scope','active-badge','stage-title','event-delta','stage-state','viewport','plane-labels','stage-overlay','node-tooltip','status','expand-stage','zoom-in','zoom-out','orbit','top','front','rotate','person','person-prev','person-next','person-detail','play-persona','incident','quick-summary','person-degrees','event-heading','event-explanation','event-changes','legend','replay-note','insight-root','trajectory-title','trajectory','event-ledger','ledger-count','events','measurements','research-question','rq-explanation','rq-status','rq-issues','rq-pairs','rq-homophily','roster-limitations','metric','metric-definition','plot','metric-deltas','comparison','warning','rows','aggregate-note','aggregate-graph','source-gallery','provenance-summary','provenance-fields','source-png','provenance-hashes','primary-findings','saved-plots','help','help-title','help-close'];
// Retired at integration (legacy filter drawer, primary gallery and its dialog): these ids must not come back.
const RETIRED=['filter-form','model','method','culture','language','seed','apply','reset-filters','filter-preview','candidate','add','candidate-detail','filter-chips','coverage-summary','open-filters','legend-heading','plot-drawer','plot-filters','plot-cohort','plot-method','plot-search','plot-clear','plot-status','plot-retry','plot-grid','plot-page-back','plot-page-next','plot-dialog','plot-dialog-title','plot-close','plot-dialog-cohort','network-stage','network-person','network-zoom-in','network-zoom-out','network-reset','network-status','plot-back','plot-dialog-position','plot-forward','plot-original','plot-source','plot-replay','plot-record','legacy-controls'];

test('shell keeps every live id exactly once and the retired legacy controls stay deleted',()=>{
  assert.deepEqual(FROZEN.filter(id=>!ids.includes(id)),[]);
  assert.deepEqual(ids.filter((id,i)=>ids.indexOf(id)!==i),[]);
  assert.deepEqual(RETIRED.filter(id=>ids.includes(id)),[]);
  assert.doesNotMatch(html,/plot-gallery\.js/,'the supplement panel is loaded on demand by layers-insight.js');
  assert.match(html,/id="supplement-root"/);
  const stage=html.slice(html.indexOf('<div class="stage"'),html.indexOf('<aside class="dock dock-right"'));
  for(const id of ['stage-overlay','viewport','plane-labels','node-tooltip','atlas-mode','compare-mode','formation-mode'])assert.match(stage,new RegExp(`id="${id}"`),id);
});

test('static page states no population count, offers one Filters entry and no dead options',()=>{
  // 896 / 890 / 12 / 6 are stated once, by the study ledger (layers-insight.js), not by static copy.
  assert.doesNotMatch(visibleText,/\b(?:896|890)\b/);
  assert.equal((visibleText.match(/\bFilters\b/g)||[]).length,2,'the Filters tab and its visually hidden panel heading');
  assert.doesNotMatch(html,/political affiliation|race\/ethnicity/);
  assert.doesNotMatch(html,/<option value="rq3" selected>/);
  assert.doesNotMatch(visibleText,/\b0[1-6] \//,'no stale section numbering');
  assert.match(html,/<body data-view="stack">/);
  assert.equal((html.match(/role="status" aria-live="polite"/g)||[]).length,1,'one explicit live region');
});

test('shell layout helpers follow the SPEC breakpoints',()=>{
  assert.deepEqual(BREAKPOINTS,{tabs:1150,sheet:700,bothDocks:1600});
  assert.equal(layoutMode(390),'sheet');
  assert.equal(layoutMode(699),'sheet');
  assert.equal(layoutMode(700),'tabs');
  assert.equal(layoutMode(1149),'tabs');
  assert.equal(layoutMode(1150),'desktop');
  assert.deepEqual(defaultDocks(1440),{left:true,right:false});
  assert.deepEqual(defaultDocks(1920),{left:true,right:true});
  assert.deepEqual(PANELS.map(p=>p.name),['study','filters','results','stack','persona','compare']);
  assert.equal(panelDock('results'),'left');
  assert.equal(panelDock('compare'),'right');
  assert.throws(()=>panelDock('legend'),/Unknown panel/);
  for(const panel of PANELS){
    assert.match(html,new RegExp(`id="panel-${panel.name}" data-panel="${panel.name}"`));
    assert.match(html,new RegExp(`id="tab-${panel.name}"[^>]*data-panel="${panel.name}"[^>]*data-dock="${panel.dock}"[^>]*aria-controls="panel-${panel.name}"`));
  }
});

test('stored shell preferences are validated and never trusted blindly',()=>{
  const defaults={docks:{left:true,right:false},tabs:{left:'filters',right:'stack'}};
  assert.deepEqual(parsePrefs(null,1440),defaults);
  assert.deepEqual(parsePrefs('not json',1440),defaults);
  assert.deepEqual(parsePrefs('[1,2]',1440),defaults);
  assert.deepEqual(parsePrefs(JSON.stringify({docks:{left:false,right:'yes'},tabs:{left:'results',right:'study'}}),1440),{docks:{left:false,right:false},tabs:{left:'results',right:'stack'}});
  assert.deepEqual(parsePrefs(JSON.stringify({docks:{right:true},tabs:{right:'persona'}}),1200),{docks:{left:true,right:true},tabs:{left:'filters',right:'persona'}});
});

test('mountShell is inert without a document',()=>{
  const handle=mountShell({getState:()=>({view:'stack'}),subscribe:()=>()=>{}});
  assert.equal(typeof handle.destroy,'function');
  handle.destroy();
});
