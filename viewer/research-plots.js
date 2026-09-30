// Filtering and figure preparation use exported Python measurements only.
// Keep each source/prompt/roster condition separate; seeds are individual dots.
export const dimensions = ['model', 'method', 'culture', 'language', 'seed'];
export const topology = {
  density: 'Density', avg_clustering_coef: 'Mean clustering',
  prop_nodes_lcc: 'Largest component share', modularity: 'Modularity',
  global_efficiency: 'Global efficiency', num_components: 'Connected components',
};
export const demographics = ['gender', 'race/ethnicity', 'religion', 'political affiliation', 'age'];

export function studyRuns(runs, study) {
  return runs.filter(run => study === 'historical' ? run.status === 'historical_reanalyzed' : run.study === study);
}
export function filterRuns(runs, selection) {
  return dimensions.reduce((rows, key) => rows.filter(row =>
    selection[key]?.map(String).includes(String(row[key]))), studyRuns(runs, selection.study));
}
export function conditionRows(runs) {
  const groups = new Map();
  for (const run of runs) {
    const key = JSON.stringify(['study', 'model', 'method', 'culture', 'language', 'roster_id', 'prompt_variant', 'language_confound'].map(k => run[k] ?? null));
    if (!groups.has(key)) groups.set(key, {key, run, runs: []});
    groups.get(key).runs.push(run);
  }
  return [...groups.values()].sort((a, b) => a.key.localeCompare(b.key));
}
export function measurement(run, metric) {
  const value = metric === 'age' ? run.age_assortativity :
    demographics.includes(metric) ? run.homophily[metric] : run.metrics[metric];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
export function coverage(runs, selection) {
  const expected = dimensions.reduce((n, key) => n * (selection[key]?.length ?? 0), 1);
  const observed = new Set(filterRuns(runs, selection).map(run => JSON.stringify(dimensions.map(k => String(run[k]))))).size;
  return {expected, observed, missing: expected - observed};
}

const escape = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;'}[char]));
const number = value => value == null ? 'NA' : value.toFixed(3);
const shortModel = model => model.replace('gpt-', '');
function rowLabel(row, variants) {
  const run = row.run;
  return `${shortModel(run.model)} / ${run.method} / ${run.culture.toUpperCase()} / ${run.language} / ${run.study} / v${variants.indexOf(run.prompt_variant ?? 'legacy_pre_source_hash') + 1}`;
}
function shell(title, caption, width, height, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(title)}"><title>${escape(title)}</title><desc>${escape(caption)}</desc><rect width="100%" height="100%" fill="#fff"/><g font-family="sans-serif" font-size="12" fill="#172c3d"><text x="16" y="24" font-size="16" font-weight="bold">${escape(title)}</text><text x="16" y="45" fill="#526372">${escape(caption)}</text>${body}</g></svg>`;
}

export function topologySvg(rows, metric, caption) {
  const variants = [...new Set(rows.map(row => row.run.prompt_variant ?? 'legacy_pre_source_hash'))];
  const values = rows.flatMap(row => row.runs.map(run => measurement(run, metric))).filter(v => v !== null);
  const low = Math.min(0, ...values), high = Math.max(1, ...values);
  const x = value => 540 + (value - low) / (high - low) * 320;
  let body = '';
  for (let i = 0; i <= 4; i++) {
    const value = low + (high - low) * i / 4, px = x(value);
    body += `<path d="M${px} 72 V${104 + rows.length * 30}" stroke="#e4e9ed"/><text x="${px}" y="72" text-anchor="middle">${number(value)}</text>`;
  }
  rows.forEach((row, i) => {
    const y = 96 + i * 30;
    body += `<text x="16" y="${y + 4}">${escape(rowLabel(row, variants))}</text>`;
    const points = row.runs.map(run => measurement(run, metric));
    points.forEach((value, j) => {
      if (value === null) return;
      // Deterministic vertical separation reveals coincident seed measurements.
      const offset = (j - (points.length - 1) / 2) * 8;
      body += `<circle cx="${x(value)}" cy="${y + offset}" r="4" fill="#186e71" stroke="#172c3d"><title>${escape(row.runs[j].run_id)}: ${number(value)}</title></circle>`;
    });
    body += `<text x="880" y="${y + 4}">n=${points.filter(v => v !== null).length}/${points.length}</text>`;
  });
  if (!rows.length) body += '<text x="16" y="95">No saved runs for this selection.</text>';
  return shell(topology[metric], caption + ' | Individual runs; no confidence intervals. v = prompt variant.', 970, 130 + rows.length * 30, body);
}

export function homophilySvg(rows, caption) {
  const variants = [...new Set(rows.map(row => row.run.prompt_variant ?? 'legacy_pre_source_hash'))];
  let body = demographics.map((demo, j) => `<text x="${580 + j * 112}" y="72" text-anchor="middle">${escape(demo === 'political affiliation' ? 'Politics' : demo === 'race/ethnicity' ? 'Race/ethnicity' : demo === 'age' ? 'Age (numeric)' : demo)}</text>`).join('');
  rows.forEach((row, i) => {
    const y = 86 + i * 34;
    body += `<text x="16" y="${y + 20}">${escape(rowLabel(row, variants))}</text>`;
    demographics.forEach((demo, j) => {
      const values = row.runs.map(run => measurement(run, demo)).filter(v => v !== null);
      const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
      const opacity = mean === null ? 0 : Math.min(.75, Math.abs(mean) * .75);
      const left = 530 + j * 112;
      body += `<rect x="${left}" y="${y}" width="100" height="28" fill="${mean !== null && mean < 0 ? '#b94e20' : '#186e71'}" fill-opacity="${opacity}" stroke="#ccd5dc"/><text x="${left + 50}" y="${y + 18}" text-anchor="middle" fill="${opacity > .45 ? '#fff' : '#172c3d'}">${number(mean)} (${values.length})<title>${escape(demo)}; valid n=${values.length}/${row.runs.length}</title></text>`;
    });
  });
  if (!rows.length) body += '<text x="16" y="105">No saved runs for this selection.</text>';
  return shell('Demographic mixing', caption + ' | Mean (valid n). Teal: positive; orange: negative; NA: undefined. Age uses a different metric.', 1110, 130 + rows.length * 34, body);
}

export function selectionCsv(runs) {
  const fields = ['run_id', 'study', ...dimensions, 'roster_id', 'prompt_variant', 'status', 'source', 'source_sha256', 'nodes', 'edges', ...Object.keys(topology), ...demographics];
  const cell = value => `"${String(value ?? '').replace(/"/g, '""')}"`;
  const lines = runs.map(run => fields.map(key => cell(key === 'nodes' ? 50 : key === 'edges' ? run.edges.length :
    Object.hasOwn(topology, key) || demographics.includes(key) ? measurement(run, key) : run[key])).join(','));
  return [fields.map(cell).join(','), ...lines].join('\n');
}
