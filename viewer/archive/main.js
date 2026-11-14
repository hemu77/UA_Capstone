import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {validateData, differences, edgeKey, neighbors, replayEdges, comparisonLayout} from './graph-data.js';
import {togglePersona,tapTracker} from './graph-data.js';
import {dimensions, topology, studyRuns, filterRuns, conditionRows, coverage, measurement, topologySvg, homophilySvg, selectionCsv} from './research-plots.js';

const $ = id => document.getElementById(id);
const format = value => value == null ? 'unavailable' : Number(value).toFixed(3);
const palette = ['#186e71', '#b94e20', '#65529a', '#477334', '#af3661', '#956710', '#376dac', '#715541'];
const labels = {model: 'Model', method: 'Method', study: 'Study', culture: 'Country framing', language: 'Instructions', seed: 'Seed'};
let data, panes = [], selected = '', sandboxEdges = null, syncing = false, layout;
let analysisRows = [], figureSvgs = {};
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
function options(select, values, previous) {
  select.replaceChildren(...values.map(value => new Option(String(value), String(value))));
  if (values.map(String).includes(String(previous))) select.value = previous;
}
function attribute(person) {
  const value = person.attributes[$('color').value];
  return $('color').value === 'age' ? `${Math.floor(value / 10) * 10}s (display only)` : String(value);
}
function displayed(pane) {
  if (pane === panes[0] && sandboxEdges) return {...pane.run, edges: sandboxEdges};
  if (pane === panes[0] && Array.isArray(pane.run.events)) return {...pane.run, edges: replayEdges(pane.run.events, Number($('replay').value))};
  return pane.run;
}
function updateUrl() {
  const query = new URLSearchParams({a: panes[0].run.run_id, b: panes[1].run.run_id, color: $('color').value});
  if (selected) query.set('node', selected);
  query.set('flat', $('flat').checked ? '1' : '0');
  query.set('f', JSON.stringify(analysisSelection()));
  query.set('metric', $('plot-metric').value);
  history.replaceState(null, '', `${location.pathname}?${query}`);
}
function analysisSelection() {
  return Object.fromEntries([['study', $('study').value], ...dimensions.map(key => [key,
    [...$(`filter-${key}`).querySelectorAll('input[data-value]:checked')].map(input => input.dataset.value)])]);
}
function buildAnalysisFilters(preferred) {
  const rows = studyRuns(data.runs, $('study').value);
  for (const key of dimensions) {
    const host = $(`filter-${key}`), title = labels[key];
    host.replaceChildren(element('legend', title));
    const values = [...new Set(rows.map(run => String(run[key])))].sort();
    const defaults = preferred?.[key] ?? (key === 'language' ? ['english'] : values);
    const allLabel = element('label', 'All'), all = element('input');
    all.type = 'checkbox'; all.setAttribute('aria-label', `All ${title}`); allLabel.prepend(all); host.append(allLabel);
    function refreshAll() {
      const checked = host.querySelectorAll('input[data-value]:checked').length;
      all.checked = checked === values.length && values.length > 0;
      all.indeterminate = checked > 0 && checked < values.length;
    }
    for (const value of values) {
      const label = element('label', value), input = element('input');
      input.type = 'checkbox'; input.dataset.value = value;
      input.setAttribute('aria-label', `Include ${title} ${value}`);
      input.checked = defaults.map(String).includes(value);
      input.addEventListener('change', () => { refreshAll(); renderAnalysis(); });
      label.prepend(input); host.append(label);
    }
    all.addEventListener('change', () => {
      for (const input of host.querySelectorAll('input[data-value]')) input.checked = all.checked;
      refreshAll(); renderAnalysis();
    });
    refreshAll();
  }
}
function renderAnalysis() {
  const selection = analysisSelection(), count = coverage(data.runs, selection);
  analysisRows = filterRuns(data.runs, selection);
  $('selection-count').textContent = `${analysisRows.length} saved runs / ${count.observed} of ${count.expected} selected combinations / ${count.missing} missing`;
  $('model-coverage').textContent = ['gpt-6-sol', 'gpt-6-luna', 'gpt-4.1-mini'].map(model =>
    `${model}: ${data.runs.filter(run => run.study === 'engineering_pilot' && run.model === model).length} pilot graphs`).join(' | ');
  const caption = `${$('study').selectedOptions[0].textContent}; ${analysisRows.length} runs; ${selection.culture.join('+') || 'no country'}; ${selection.language.join('+') || 'no language'}`;
  const rows = conditionRows(analysisRows);
  figureSvgs = {topology: topologySvg(rows, $('plot-metric').value, caption), mixing: homophilySvg(rows, caption)};
  // The SVG builder escapes every data label; no raw model output enters HTML.
  $('topology-plot').innerHTML = figureSvgs.topology;
  $('mixing-plot').innerHTML = figureSvgs.mixing;
  $('selected-runs').replaceChildren(...analysisRows.map(run => {
    const row = element('tr');
    for (const value of [run.model, run.method, run.culture, run.language, run.seed, format(measurement(run, $('plot-metric').value))]) row.append(element('td', value));
    const actions = element('td');
    for (const pane of panes) {
      const button = element('button', `Load ${pane.id.toUpperCase()}`);
      button.setAttribute('aria-label', `Load ${run.run_id} in ${pane.id.toUpperCase()}`);
      button.addEventListener('click', () => { updateFilters(pane, run); render(); $('networks').scrollIntoView({block: 'start'}); });
      actions.append(button);
    }
    row.append(actions); return row;
  }));
  $('selected-metric').textContent = topology[$('plot-metric').value];
  if (panes.length === 2 && panes.every(pane => pane.run)) updateUrl();
}
function updateFilters(pane, preferred) {
  let rows = data.runs;
  for (const key of Object.keys(labels)) {
    const control = pane.filters[key];
    const values = [...new Set(rows.map(row => row[key]))].sort();
    options(control, values, preferred?.[key] ?? control.value);
    rows = rows.filter(row => String(row[key]) === control.value);
  }
  pane.run = rows[0];
  pane.root.querySelector('.run-id').textContent = pane.run.run_id;
  if (pane === panes[0]) {
    sandboxEdges = null;
    $('sandbox').checked = false;
    const events = pane.run.events;
    $('replay').disabled = !Array.isArray(events);
    $('replay').max = events?.length ?? 0;
    $('replay').value = events?.length ?? 0;
    $('replay-note').textContent = events ? 'Step through actual recorded decisions. Intermediate graphs are not final research measurements.' : 'Historical graph: final edges only. Replay unavailable; no history has been invented.';
  }
}
function createScene(pane) {
  try {
    const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(45, 1, .1, 1000);
    camera.position.set(0, 0, 240);
    pane.host.append(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = true;controls.enableZoom=true;controls.zoomSpeed=.65;
    controls.minDistance = 120;
    controls.maxDistance = 550;
    controls.enableDamping = false; // Render only on interaction, not an endless idle animation.
    pane.visual = {renderer, scene, camera, controls, group: null, nodeMeshes: []};
    controls.addEventListener('change', () => {
      renderer.render(scene, camera);
      if (syncing) return;
      syncing = true;
      for (const other of panes) if (other !== pane && other.visual) {
        other.visual.camera.position.copy(camera.position);
        other.visual.camera.quaternion.copy(camera.quaternion);
        other.visual.controls.target.copy(controls.target);
        other.visual.controls.update();
        other.visual.renderer.render(other.visual.scene, other.visual.camera);
      }
      syncing = false;
    });
    new ResizeObserver(() => {
      const width = pane.host.clientWidth, height = pane.host.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      renderer.render(scene, camera);
    }).observe(pane.host);
    const tap=tapTracker();
    renderer.domElement.addEventListener('pointerdown', event => tap.down(event));
    renderer.domElement.addEventListener('pointermove', event => tap.move(event));
    renderer.domElement.addEventListener('pointercancel', () => tap.cancel());
    renderer.domElement.addEventListener('wheel', () => tap.cancel(),{passive:true});
    renderer.domElement.addEventListener('pointerup', event => {
      if (!tap.up(event)) return;
      const rect = renderer.domElement.getBoundingClientRect(), ray = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
      const hit = ray.intersectObjects(pane.visual.nodeMeshes)[0];
      selectPersona(hit?hit.object.userData.id:'',!!hit);
    });
    renderer.domElement.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      $('status').textContent = '3D context unavailable. The persona table and downloads remain usable; reload to restore rendering.';
    });
  } catch {
    pane.host.textContent = '3D rendering unavailable on this device. Use the full accessible table below.';
    pane.root.querySelector('[data-download="png"]').disabled = true;
  }
}
function draw(pane, colors, diff) {
  const run = displayed(pane), visual = pane.visual;
  pane.host.dataset.nodes = data.personas.length;
  pane.host.dataset.edges = run.edges.length;
  if (!visual) return;
  const {scene, camera, renderer, controls} = visual;
  if (visual.group) {
    visual.group.traverse(item => { item.geometry?.dispose(); item.material?.dispose(); });
    scene.remove(visual.group);
  }
  const group = new THREE.Group(), positions = new Map();
  visual.group = group;
  visual.nodeMeshes = [];
  const near = neighbors(run.edges, selected);
  for (const person of data.personas) {
    const point = new THREE.Vector3(...layout.get(person.id));
    if ($('flat').checked) point.z = 0;
    positions.set(person.id, point);
    const active = !selected || selected === person.id || near.has(person.id);
    const material = new THREE.MeshBasicMaterial({color: colors.get(attribute(person)), transparent: true, opacity: active ? 1 : .18});
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(person.id === selected ? 3.5 : 2.3, 12, 8), material);
    mesh.position.copy(point);
    mesh.userData.id = person.id;
    visual.nodeMeshes.push(mesh);
    group.add(mesh);
  }
  for (const edge of run.edges) {
    const shared = diff?.shared.has(edgeKey(edge));
    if (diff && $('edge-mode').value === 'shared' && !shared) continue;
    if (diff && $('edge-mode').value === 'different' && shared) continue;
    if ($('neighbors').checked && selected && !edge.includes(selected)) continue;
    const active = !selected || edge.includes(selected);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(edge.map(id => positions.get(id))),
      new THREE.LineBasicMaterial({color: shared ? '#167d78' : '#b94e20', transparent: true, opacity: active ? .45 : .06}));
    group.add(line);
  }
  scene.add(group);
  controls.enableRotate = !$('flat').checked;
  renderer.render(scene, camera);
}
function renderMetrics(pane) {
  const host = pane.root.querySelector('.metrics');
  host.replaceChildren();
  const editing = pane === panes[0] && sandboxEdges;
  const partial = pane === panes[0] && pane.run.events && Number($('replay').value) < pane.run.events.length;
  if (editing || partial) {
    host.append(element('p', `${editing ? 'Manual simulation' : 'Intermediate replay'}: ${displayed(pane).edges.length} edges. Final research metrics hidden.`));
  } else {
    for (const [key, label] of [['density', 'Density'], ['avg_clustering_coef', 'Clustering'], ['prop_nodes_lcc', 'Largest component'], ['modularity', 'Modularity']]) {
      const card = element('div', undefined, 'metric');
      card.append(element('strong', format(pane.run.metrics[key])), element('span', label));
      host.append(card);
    }
  }
  const metric = $('color').value === 'age' ? `Numeric age assortativity: ${format(pane.run.age_assortativity)}` : `Coleman (${$('color').value}): ${format(pane.run.homophily[$('color').value])}`;
  pane.root.querySelector('.provenance').textContent = `${editing || partial ? 'Not a final research network.' : metric + '.'} Source: ${pane.run.source}. ${pane.run.provenance}`;
}
function render() {
  const values = [...new Set(data.personas.map(attribute))].sort(), colors = new Map(values.map((value, i) => [value, palette[i % palette.length]]));
  $('legend').replaceChildren(...values.map(value => {
    const label = element('span', value), swatch = element('i', undefined, 'swatch');
    swatch.style.background = colors.get(value);
    label.prepend(swatch);
    return label;
  }));
  const a = displayed(panes[0]), b = displayed(panes[1]), diff = differences(a, b);
  layout = comparisonLayout(data.personas, a.edges, b.edges);
  $('edge-mode').disabled = !diff;
  $('comparison').textContent = diff ? `${data.personas.length} matched personas | ${diff.shared.size} shared ties | ${diff.onlyA.size} only A | ${diff.onlyB.size} only B${sandboxEdges ? ' | MANUAL SIMULATION A' : ''}` : 'Different rosters: persona-level edge comparison disabled. Compare summary measurements only.';
  for (const pane of panes) { draw(pane, colors, diff); renderMetrics(pane); }
  $('people').replaceChildren(...data.personas.map(person => {
    const row = element('tr', undefined, person.id === selected ? 'selected' : '');
    const cell = element('td'), button = element('button', `ID ${person.id}`);
    button.dataset.persona = person.id;
    button.addEventListener('click', () => selectPersona(person.id));
    cell.append(button);
    row.append(cell, element('td', attribute(person)), element('td', neighbors(a.edges, person.id).size), element('td', neighbors(b.edges, person.id).size));
    return row;
  }));
  if (selected) {
    const person = data.personas.find(p => p.id === selected);
    $('inspection').textContent = `ID ${selected}. ${Object.entries(person.attributes).map(([k, v]) => `${k}: ${v}`).join('; ')}. Neighbors A: ${[...neighbors(a.edges, selected)].join(', ') || 'none'}. Neighbors B: ${[...neighbors(b.edges, selected)].join(', ') || 'none'}.`;
  } else $('inspection').textContent = 'Choose a node or ID to inspect its attributes and connections in both graphs.';
  for (const id of ['from', 'to', 'toggle-edge', 'discard']) $(id).disabled = !sandboxEdges;
  $('sandbox-status').textContent = sandboxEdges ? 'Sandbox edits exist only in memory. Links share original research views, not manual edits.' : '';
  updateUrl();
}
function selectPersona(id,toggle=true) {
  const wasInTable = !!document.activeElement.closest('#people');
  selected = toggle?togglePersona(selected,id):id; $('persona').value = selected; render();
  // Rebuilding the table must not strand keyboard users at the top of the page.
  if (wasInTable && id) $('people').querySelector(`[data-persona="${CSS.escape(id)}"]`)?.focus();
}
function resetCameras() {
  for (const pane of panes) if (pane.visual) {
    pane.visual.camera.position.set(0, 0, 240);
    pane.visual.controls.target.set(0, 0, 0);
    pane.visual.controls.update();
  }
  render();
}
function save(blob, name) {
  const a = document.createElement('a'), url = URL.createObjectURL(blob);
  a.href = url; a.download = name; a.hidden = true;
  document.body.append(a); a.click(); a.remove();
  // Leave slow embedded-browser download handlers time to consume the blob.
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
function download(pane, kind) {
  const run = displayed(pane), modified = (pane === panes[0] && (sandboxEdges || (pane.run.events && Number($('replay').value) < pane.run.events.length)));
  const prefix = `${modified ? 'NOT-RESEARCH_' : ''}${run.run_id}`;
  if (kind === 'png') {
    // Label exported images too; screenshots without context are easy to misuse.
    const canvas = document.createElement('canvas'), original = pane.visual.renderer.domElement;
    canvas.width = original.width; canvas.height = original.height + 130;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fffefa'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(original, 0, 90); ctx.fillStyle = '#203631'; ctx.font = '16px sans-serif';
    ctx.fillText(modified ? 'MANUAL / INTERMEDIATE VIEW - not a research result' :
      run.study === 'engineering_pilot' ? 'Engineering pilot; not confirmatory research' : 'Historical descriptive network; not cultural validation', 12, 24);
    ctx.fillText(run.run_id, 12, 48, canvas.width - 24);
    ctx.fillText(`Color: ${$('color').value}; edge filter: ${$('edge-mode').value}; selected ID: ${selected || 'all'}`, 12, 72, canvas.width - 24);
    ctx.fillText('Shared union-force layout; coordinates are not measured social distance.', 12, canvas.height - 12, canvas.width - 24);
    canvas.toBlob(blob => { if (blob) save(blob, prefix + '.png'); });
  } else if (kind === 'csv') save(new Blob(['source,target\n' + run.edges.map(e => e.join(',')).join('\n')], {type: 'text/csv'}), prefix + '.csv');
  else save(new Blob([JSON.stringify({...run, status: modified ? 'manual_or_intermediate_not_research' : run.status,
    ...(modified ? {metrics: null, homophily: null, age_assortativity: null} : {}),
    layout: 'shared_union_force_algorithmic_coordinates',
    personas: data.personas.map(person => ({...person, position: layout.get(person.id)}))}, null, 2)], {type: 'application/json'}), prefix + '.json');
}

async function start() {
  const response = await fetch('./data/networks.json', {cache:'no-store'});
  if (!response.ok) throw new Error('Export not found. Run export_research_viewer.py, then npm run build.');
  data = validateData(await response.json());
  const historical = data.runs.filter(run => run.status === 'historical_reanalyzed').length;
  $('status').textContent = `${historical} historical / ${data.engineering_pilots || 0} pilot / ${data.quarantined_runs.length} quarantined / offline viewer`;
  const query = new URLSearchParams(location.search);
  let preference;
  try { preference = JSON.parse(query.get('f')); } catch { preference = null; }
  if (preference && dimensions.every(key => Array.isArray(preference[key]) && preference[key].every(v => typeof v === 'string' || typeof v === 'number')) &&
    [...$('study').options].some(option => option.value === preference.study)) $('study').value = preference.study;
  else preference = null;
  options($('plot-metric'), Object.keys(topology), query.get('metric') || 'density');
  for (const option of $('plot-metric').options) option.textContent = topology[option.value];
  buildAnalysisFilters(preference);
  if ([...$('color').options].some(o => o.value === query.get('color'))) $('color').value = query.get('color');
  $('flat').checked = query.get('flat') !== '0';
  selected = data.personas.some(p => p.id === query.get('node')) ? query.get('node') : '';
  for (const id of ['persona', 'from', 'to']) {
    const previous = id === 'persona' ? selected : data.personas[id === 'to' ? 1 : 0].id;
    options($(id), (id === 'persona' ? [''] : []).concat(data.personas.map(p => p.id)), previous);
    if (id === 'persona') $(id).options[0].textContent = 'All personas';
  }
  panes = ['a', 'b'].map(id => ({id, root: $(`panel-${id}`), filters: {}, visual: null}));
  for (const pane of panes) {
    pane.host = pane.root.querySelector('.scene');
    for (const [key, title] of Object.entries(labels)) {
      const label = element('label', title), select = element('select');
      select.setAttribute('aria-label', `${pane.id.toUpperCase()} ${title}`);
      label.append(select); pane.root.querySelector('.filters').append(label);
      pane.filters[key] = select;
      select.addEventListener('change', () => { updateFilters(pane); render(); });
    }
    const preferred = data.runs.find(r => r.run_id === query.get(pane.id)) ||
      data.runs.find(r => r.method === 'sequential' && r.model === (pane.id === 'a' ? 'gpt-6-sol' : 'gpt-6-luna') && r.study === 'engineering_pilot' && r.language === 'english') ||
      data.runs.find(r => r.method === 'sequential' && r.model === 'gpt-4.1-mini' && r.study === 'engineering_pilot' && r.language === 'english');
    updateFilters(pane, preferred);
    createScene(pane);
    for (const button of pane.root.querySelectorAll('[data-download]')) button.addEventListener('click', () => download(pane, button.dataset.download));
  }
  for (const id of ['color', 'neighbors', 'edge-mode', 'replay']) $(id).addEventListener('change', render);
  $('persona').addEventListener('change', () => selectPersona($('persona').value,false));
  $('flat').addEventListener('change', resetCameras);
  $('reset').addEventListener('click', resetCameras);
  $('sandbox').addEventListener('change', () => { sandboxEdges = $('sandbox').checked ? structuredClone(panes[0].run.edges) : null; render(); });
  $('discard').addEventListener('click', () => { sandboxEdges = null; $('sandbox').checked = false; render(); });
  $('toggle-edge').addEventListener('click', () => {
    const edge = [$('from').value, $('to').value];
    if (edge[0] === edge[1]) { $('sandbox-status').textContent = 'Self-links are not allowed. Choose two different people.'; return; }
    const index = sandboxEdges.findIndex(e => edgeKey(e) === edgeKey(edge));
    if (index < 0) sandboxEdges.push(edge); else sandboxEdges.splice(index, 1);
    render();
  });
  $('share').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(location.href); $('share').textContent = 'Link copied'; }
    catch { $('share').textContent = 'Copy the URL from your address bar'; }
  });
  $('study').addEventListener('change', () => { buildAnalysisFilters(); renderAnalysis(); });
  $('plot-metric').addEventListener('change', renderAnalysis);
  $('reset-filters').addEventListener('click', () => { $('study').value = 'engineering_pilot'; buildAnalysisFilters(); renderAnalysis(); });
  $('selection-csv').addEventListener('click', () => save(new Blob([selectionCsv(analysisRows)], {type: 'text/csv'}), 'selected-research-runs.csv'));
  $('selection-json').addEventListener('click', () => save(new Blob([JSON.stringify({analysis_version: data.analysis_version,
    selection: analysisSelection(), coverage: coverage(data.runs, analysisSelection()), runs: analysisRows}, null, 2)], {type: 'application/json'}), 'selected-research-runs.json'));
  for (const button of document.querySelectorAll('[data-figure]')) button.addEventListener('click', () =>
    save(new Blob([figureSvgs[button.dataset.figure]], {type: 'image/svg+xml'}), `selected-${button.dataset.figure}.svg`));
  renderAnalysis();
  render();
}
start().catch(error => { $('status').textContent = error.message; $('status').className = 'error'; });
