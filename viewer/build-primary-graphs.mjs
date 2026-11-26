// Generates the compact primary-study data under viewer/public/data/graphs/ (SPEC section C.3):
//   index.json        design axes, roster, atlas layout, 896 attempt rows in slot order, hashes
//   edges.bin         all 890 edge lists, one uint8 pair (a < b) per tie, runs in slot order
//   metrics.json      the 16 saved descriptive metrics of the 890 valid graphs, columnar, CSV precision
//   supplements.json  the 6 supplemental graphs, labelled as excluded from the primary study
// Sources are read only: the frozen attempts CSV, the .adj files it names, the roster, the recorded
// event logs, and the two public fixtures (primary-viewer.json, saved-networks.json) used as cross-checks.
// Every integrity check throws; nothing is repaired or substituted. Output is byte-deterministic.
// Run from anywhere: `node build-primary-graphs.mjs`. build.mjs imports generateGraphFiles() to prove
// the committed files still equal a fresh generation before it copies them.
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';
import {comparisonLayout, edgeKey} from './graph-data.js';

const viewer = fileURLToPath(new URL('.', import.meta.url));
const repo = resolve(viewer, '..');
export const OUTPUT_DIR = resolve(viewer, 'public/data/graphs');
export const OUTPUT_FILES = Object.freeze(['index.json', 'edges.bin', 'metrics.json', 'supplements.json']);
export const SOURCES = Object.freeze({
  attempts: 'outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv',
  supplements: 'outputs/analysis_ready/060dcd3a86d5/supplemental_valid_graphs.csv',
  roster: 'text-files/revision224_adults.json'
});
// The fixed 4 x 4 x 7 x 8 design, in the display order the published viewer has always used.
export const DESIGN = Object.freeze({
  models: ['gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol'],
  methods: ['global', 'local', 'sequential', 'iterative'],
  settings: [['us', 'english'], ['india', 'english'], ['japan', 'english'], ['brazil', 'english'], ['us', 'hindi'], ['us', 'japanese'], ['us', 'portuguese']],
  seeds: [44000, 44001, 44002, 44003, 44004, 44005, 44006, 44007]
});
export const METRIC_COLUMNS = Object.freeze(['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'global_efficiency', 'num_components', 'modularity',
  'radius_lcc', 'diameter_lcc', 'avg_shortest_path_lcc', 'degree_one_share', 'isolate_share', 'degree_gini',
  'coleman_gender', 'coleman_religion', 'coleman_political orientation', 'age_assortativity']);
export const INDEX_COLUMNS = Object.freeze(['slot', 'status', 'edges', 'density', 'avg_clustering_coef', 'adj_sha256', 'record_sha256', 'events_sha256', 'failure']);
const STUDY = Object.freeze({attempts: 896, valid: 890, empty: 12, failures: 6, supplements_excluded: 6, personas: 50, cells: 112, repetitions: 8});
const PROTOCOL = '060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f';
const SHARED = Object.freeze({
  study: 'primary',
  prompt_variant: 'frozen060d_reference',
  protocol_sha256: PROTOCOL,
  analysis_version: 'frozen primary; descriptive run metrics',
  replay_status: 'actual_recorded_graph_changes',
  run_id_template: 'final_060dcd3a86d5_main_{method}_{model}_{culture}_{language}_s{seed}',
  source_template: 'outputs/final_study/060dcd3a86d5/main/runs/{run_id}.adj'
});
const LAYOUT_BASIS = 'force layout on tie frequency across all 890 valid graphs; algorithmic, not geographic';
const SUPPLEMENT_NOTE = 'Excluded from the primary study: 6 supplemental graphs collected after the primary run. They never enter the primary atlas, stack, filters, counts or comparisons.';

const fail = message => { throw new Error(`Primary graph generation stopped: ${message}`); };
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const fill = (template, values) => template.replace(/\{(\w+)\}/g, (_, key) => String(values[key]));
const readRepo = path => readFile(resolve(repo, path));

// RFC 4180 subset: comma separated, optional double-quoted fields, LF or CRLF rows, header row required.
export function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"' && field === '') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (quoted) fail('unterminated quoted CSV field');
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const [header, ...body] = rows;
  if (!header || new Set(header).size !== header.length) fail('CSV header missing or duplicated');
  return body.map((cells, line) => {
    if (cells.length !== header.length) fail(`CSV row ${line + 2} has ${cells.length} fields, expected ${header.length}`);
    return Object.fromEntries(header.map((name, k) => [name, cells[k]]));
  });
}

// networkx adjacency list: '#' comment lines, then "node neighbour neighbour ..." with each tie written once.
// Returns ties as numeric [a, b] with a < b, sorted by a then b. Every roster id must appear as a row.
export function parseAdjacency(text, personaCount) {
  const seen = new Set(), ties = [];
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || line.trim() === '') continue;
    const ids = line.trim().split(/\s+/).map(token => {
      if (!/^(0|[1-9]\d*)$/.test(token) || Number(token) >= personaCount) fail(`adjacency id ${JSON.stringify(token)} is outside the roster`);
      return Number(token);
    });
    const [node, ...neighbours] = ids;
    if (seen.has(node)) fail(`adjacency lists node ${node} twice`);
    seen.add(node);
    for (const other of neighbours) {
      if (other === node) fail(`self tie on node ${node}`);
      ties.push(node < other ? [node, other] : [other, node]);
    }
  }
  if (seen.size !== personaCount) fail(`adjacency lists ${seen.size} nodes, expected ${personaCount}`);
  ties.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  for (let i = 1; i < ties.length; i++) if (ties[i][0] === ties[i - 1][0] && ties[i][1] === ties[i - 1][1]) fail(`duplicate tie ${ties[i].join('-')}`);
  return ties;
}

// Python json.dumps(value, sort_keys=True) for the roster's value types; reproduces the CSV roster_sha256.
export function pythonJson(value) {
  if (value === null) return 'null';
  if (typeof value === 'string') {
    if (!/^[\x20-\x7e]*$/.test(value)) fail('roster contains characters outside printable ASCII');
    return JSON.stringify(value);
  }
  if (Number.isInteger(value)) return String(value);
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(', ')}]`;
  if (typeof value === 'object') return `{${Object.keys(value).sort().map(key => `${pythonJson(key)}: ${pythonJson(value[key])}`).join(', ')}}`;
  fail(`roster value ${JSON.stringify(value)} has no canonical form here`);
}

const metricValue = (raw, column, runId) => {
  if (raw === '') return null;
  const value = Number(raw);
  if (!Number.isFinite(value)) fail(`metric ${column} of ${runId} is not a number`);
  return value;
};
const bool = (raw, name) => raw === 'True' ? true : raw === 'False' ? false : fail(`${name} is ${JSON.stringify(raw)}, expected True or False`);
const hex = (value, name) => /^[0-9a-f]{64}$/.test(value) ? value : fail(`${name} is not a SHA-256 hex digest`);
const settingIndex = (culture, language) => DESIGN.settings.findIndex(([c, l]) => c === culture && l === language);
export const slotIndex = ({model, method, culture, language, seed}) => {
  const m = DESIGN.models.indexOf(model), p = DESIGN.methods.indexOf(method), s = settingIndex(culture, language), b = DESIGN.seeds.indexOf(Number(seed));
  if ([m, p, s, b].some(i => i < 0)) fail(`attempt ${model}/${method}/${culture}/${language}/${seed} is outside the design`);
  return ((m * DESIGN.methods.length + p) * DESIGN.settings.length + s) * DESIGN.seeds.length + b;
};
const numericEdges = edges => edges.map(([a, b]) => [Number(a), Number(b)].sort((x, y) => x - y)).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
const sameTies = (left, right) => left.length === right.length && left.every(([a, b], i) => a === right[i][0] && b === right[i][1]);
const round1 = value => { const r = Math.round(value * 10) / 10; return r === 0 ? 0 : r; };

// One row per line keeps diffs of the committed JSON reviewable; the bytes depend only on the data.
function formatJson(object, rowsPerLine) {
  const entries = Object.entries(object).map(([key, value]) => {
    const body = rowsPerLine.includes(key) && Array.isArray(value)
      ? `[\n${value.map(row => '  ' + JSON.stringify(row)).join(',\n')}\n]`
      : JSON.stringify(value);
    return `${JSON.stringify(key)}: ${body}`;
  });
  return Buffer.from(`{\n${entries.join(',\n')}\n}\n`);
}

export async function generateGraphFiles() {
  const attemptsBytes = await readRepo(SOURCES.attempts), rosterBytes = await readRepo(SOURCES.roster), supplementBytes = await readRepo(SOURCES.supplements);
  const fixture = JSON.parse(await readFile(resolve(viewer, 'public/data/primary-viewer.json'), 'utf8'));
  const catalog = JSON.parse(await readFile(resolve(viewer, 'public/data/saved-networks.json'), 'utf8'));

  // Design and roster.
  for (const key of ['models', 'methods', 'settings']) if (JSON.stringify(fixture[key]) !== JSON.stringify(DESIGN[key])) fail(`design ${key} differs from primary-viewer.json`);
  const roster = JSON.parse(rosterBytes);
  const personaIds = Object.keys(roster).sort((a, b) => Number(a) - Number(b));
  if (personaIds.length !== STUDY.personas || personaIds.some((id, i) => id !== String(i))) fail('roster ids are not exactly 0-49');
  const rosterId = sha256(pythonJson(roster));
  if (rosterId !== fixture.roster_id) fail('canonical roster hash differs from primary-viewer.json roster_id');
  if (JSON.stringify(fixture.personas.map(p => ({id: p.id, ...p.attributes}))) !== JSON.stringify(personaIds.map(id => ({id, ...roster[id]})))) fail('roster differs from primary-viewer.json personas');

  // Attempts.
  const rows = parseCsv(attemptsBytes.toString('utf8'));
  if (rows.length !== STUDY.attempts) fail(`attempts CSV has ${rows.length} rows, expected ${STUDY.attempts}`);
  const header = Object.keys(rows[0]);
  for (const column of METRIC_COLUMNS) if (!header.includes(column)) fail(`attempts CSV lacks metric column ${column}`);
  const fixtureRuns = new Map(fixture.runs.map(run => [run.run_id, run])), fixtureFailures = new Map(fixture.failures.map(run => [run.run_id, run]));
  const supplementIds = new Set(catalog.graphs.map(g => g.run_id).filter(id => !fixtureRuns.has(id)));
  const bySlot = new Array(STUDY.attempts).fill(null);
  for (const row of rows) {
    const seed = Number(row.seed), slot = slotIndex({...row, seed});
    if (bySlot[slot]) fail(`two attempts share slot ${slot}`);
    if (row.cohort !== 'primary' || !bool(row.primary_member, 'primary_member') || row.run_id !== row.original_run_id || supplementIds.has(row.run_id)) fail(`${row.run_id} is not a primary attempt`);
    if (row.run_id !== fill(SHARED.run_id_template, {...row, seed})) fail(`${row.run_id} does not follow the run id template`);
    if (row.roster_sha256 !== rosterId) fail(`${row.run_id} used a different roster`);
    if (row.wording !== 'reference') fail(`${row.run_id} used wording ${row.wording}`);
    bySlot[slot] = row;
  }
  if (new Set(rows.map(r => r.run_id)).size !== STUDY.attempts) fail('run ids are not unique');

  const attempts = [], metricRows = [], edgeChunks = [], frequency = new Map();
  let valid = 0, empty = 0, failures = 0, totalTies = 0;
  for (let slot = 0; slot < STUDY.attempts; slot++) {
    const row = bySlot[slot], record = hex(row.record_sha256, `record_sha256 of ${row.run_id}`);
    if (!bool(row.valid_graph, 'valid_graph')) {
      const failure = fixtureFailures.get(row.run_id);
      if (!row.failure || row.adj_path || row.edges !== '' || !failure || failure.failure !== row.failure) fail(`failure ${row.run_id} is not recorded consistently`);
      attempts.push([slot, 'failed', 0, null, null, null, record, null, row.failure]);
      failures++;
      continue;
    }
    if (row.failure) fail(`${row.run_id} is valid but names a failure`);
    const source = fill(SHARED.source_template, row);
    if (row.adj_path !== source) fail(`${row.run_id} adjacency path is unexpected`);
    const adjBytes = await readRepo(source), adjHash = sha256(adjBytes);
    if (adjHash !== hex(row.adj_sha256, 'adj_sha256')) fail(`${source} does not hash to adj_sha256`);
    const ties = parseAdjacency(adjBytes.toString('utf8'), STUDY.personas);
    if (Number(row.nodes) !== STUDY.personas || Number(row.edges) !== ties.length || bool(row.empty_graph, 'empty_graph') !== (ties.length === 0)) fail(`${row.run_id} node, tie or empty count differs from its .adj`);
    const published = fixtureRuns.get(row.run_id);
    if (!published) fail(`${row.run_id} is missing from primary-viewer.json`);
    if (published.source !== source || published.source_sha256 !== adjHash || published.record_sha256 !== record) fail(`${row.run_id} provenance differs from primary-viewer.json`);
    if (!sameTies(numericEdges(published.edges), ties)) fail(`${row.run_id} ties differ from primary-viewer.json`);
    const eventBytes = await readFile(resolve(viewer, 'public/data/primary-events', `${row.run_id}.json`));
    const events = sha256(eventBytes);
    if (events !== published.events_sha256) fail(`${row.run_id} recorded events differ from primary-viewer.json`);
    for (const key of Object.keys(SHARED).filter(k => !k.endsWith('_template'))) if (published[key] !== SHARED[key]) fail(`${row.run_id} ${key} is unexpected`);
    const metrics = METRIC_COLUMNS.map(column => metricValue(row[column], column, row.run_id));
    if (metrics[0] === null || metrics[1] === null) fail(`${row.run_id} lacks density or clustering`);
    const status = ties.length ? 'valid' : 'empty';
    attempts.push([slot, status, ties.length, metrics[0], metrics[1], adjHash, record, events, null]);
    metricRows.push(metrics);
    edgeChunks.push(Uint8Array.from(ties.flat()));
    for (const tie of ties) { const key = edgeKey(tie.map(String)); frequency.set(key, (frequency.get(key) || 0) + 1); }
    totalTies += ties.length; valid++; if (!ties.length) empty++;
  }
  const counts = {attempts: attempts.length, valid, empty, failures, supplements_excluded: supplementIds.size, personas: personaIds.length,
    cells: DESIGN.models.length * DESIGN.methods.length * DESIGN.settings.length, repetitions: DESIGN.seeds.length};
  if (JSON.stringify(counts) !== JSON.stringify(STUDY)) fail(`population ${JSON.stringify(counts)} differs from the study facts`);
  if (fixture.runs.length !== valid || fixture.failures.length !== failures) fail('primary-viewer.json population differs');

  const edges = Buffer.concat(edgeChunks);
  if (edges.length !== totalTies * 2) fail('edge buffer length differs from the tie total');

  // Atlas layout: one planar force layout over every tie, springs weighted by recurrence across the 890 graphs.
  const most = Math.max(...frequency.values());
  const union = [...frequency.keys()].map(key => JSON.parse(key));
  const points = comparisonLayout(personaIds.map(id => ({id})), union, [], {weights: new Map([...frequency].map(([key, n]) => [key, n / most])), planar: true});
  const xy = personaIds.flatMap(id => points.get(id).slice(0, 2).map(round1));

  // Supplements: exactly the six non-primary graphs of the saved catalogue, re-read from their .adj sources.
  const catalogById = new Map(catalog.graphs.map(g => [g.run_id, g]));
  const supplementRows = parseCsv(supplementBytes.toString('utf8'));
  const supplements = [];
  for (const row of supplementRows) {
    const graph = catalogById.get(row.run_id), seed = Number(row.seed);
    if (!supplementIds.has(row.run_id) || !graph || row.cohort === 'primary' || bool(row.primary_member, 'primary_member') || !bool(row.valid_graph, 'valid_graph')) fail(`${row.run_id} is not one of the six excluded supplements`);
    if (row.roster_sha256 !== rosterId) fail(`${row.run_id} used a different roster`);
    slotIndex({...row, seed});
    const adjBytes = await readRepo(row.adj_path), adjHash = sha256(adjBytes);
    if (adjHash !== hex(row.adj_sha256, 'adj_sha256') || graph.source !== row.adj_path || graph.source_sha256 !== adjHash) fail(`${row.adj_path} does not match its recorded hash`);
    const ties = parseAdjacency(adjBytes.toString('utf8'), STUDY.personas);
    if (Number(row.edges) !== ties.length || !sameTies(numericEdges(graph.edges), ties)) fail(`${row.run_id} ties differ from saved-networks.json`);
    supplements.push({run_id: row.run_id, cohort: row.cohort, model: row.model, method: row.method, culture: row.culture, language: row.language, seed,
      edges: ties.map(tie => tie.map(String)), source: row.adj_path, source_sha256: adjHash});
  }
  if (supplements.length !== STUDY.supplements_excluded || new Set(supplements.map(g => g.run_id)).size !== supplementIds.size) fail('supplement set differs from the six excluded graphs');
  supplements.sort((a, b) => slotIndex(a) - slotIndex(b) || (a.run_id < b.run_id ? -1 : 1));

  const metricsFile = formatJson({schema_version: 2, columns: METRIC_COLUMNS, rows: metricRows}, ['rows']);
  const supplementsFile = formatJson({schema_version: 1, note: SUPPLEMENT_NOTE, graphs: supplements}, ['graphs']);
  const index = formatJson({
    schema_version: 2,
    dataset: 'primary',
    source: {attempts_csv: SOURCES.attempts, attempts_csv_sha256: sha256(attemptsBytes), roster: SOURCES.roster, roster_file_sha256: sha256(rosterBytes),
      supplements_csv: SOURCES.supplements, supplements_csv_sha256: sha256(supplementBytes)},
    roster_id: rosterId,
    counts,
    design: DESIGN,
    personas: personaIds.map(id => ({id, ...roster[id]})),
    layout: {basis: LAYOUT_BASIS, xy},
    edges: {file: 'edges.bin', sha256: sha256(edges), bytes: edges.length, encoding: 'uint8 pairs a<b, runs in slot order'},
    metrics: {file: 'metrics.json', sha256: sha256(metricsFile)},
    supplements: {file: 'supplements.json', sha256: sha256(supplementsFile)},
    columns: INDEX_COLUMNS,
    attempts,
    shared: SHARED,
    notes: {analysis_version: fixture.analysis_version, evidence_notice: fixture.evidence_notice, calibration_note: fixture.calibration_note,
      limitations: fixture.limitations, verification: fixture.verification}
  }, ['personas', 'attempts']);
  return new Map([['index.json', index], ['edges.bin', edges], ['metrics.json', metricsFile], ['supplements.json', supplementsFile]]);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const files = await generateGraphFiles();
  await mkdir(OUTPUT_DIR, {recursive: true});
  for (const [name, bytes] of files) await writeFile(resolve(OUTPUT_DIR, name), bytes);
  console.log(JSON.stringify(Object.fromEntries([...files].map(([name, bytes]) => [name, {bytes: bytes.length, sha256: sha256(bytes)}]))));
}
