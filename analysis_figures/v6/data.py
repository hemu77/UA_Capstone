"""Verified, read-only access to the sealed tables behind release v6."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis_figures import publication as v2, publication_v3 as v3, review_v5 as v5

EXTENSION = Path('outputs/research_extension/20261114_v2')
EXTENSION_MANIFEST_SHA256 = 'a4104ca4d67289e44767ce9390ac8b017d3657ee24a915db364da8296745dae9'
V5_MANIFEST_SHA256 = 'b3738aa39afdfc462d680158e220b3771f03495cf70af8e8ca18003663ad60c2'
PRIMARY_IDS = Path('outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv')
SUPPLEMENT_IDS = Path('outputs/analysis_ready/060dcd3a86d5/supplemental_attempts.csv')
ROSTER = Path('text-files/revision224_adults.json')
EXTENSION_TABLES = ('mixing_scores', 'mixing_matrices', 'mixing_paired_cells', 'mixing_method_sensitivity',
                    'mixing_rank_sensitivity', 'mixing_coverage', 'full_network_metrics',
                    'observed_sample_metrics', 'sample_overlap', 'candidate_size_metrics',
                    'transfer_distances', 'transfer_record_summary', 'target_er_distances',
                    'target_er_summary')
COUNTS = dict(primary_attempts=896, primary_valid=890, primary_empty=12, primary_failed=6,
              primary_nonempty=878, supplemental_graphs=0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roster_sizes(root):
    """Category sizes of the one fixed fictional roster (parsed JSON, newline independent)."""
    roster = json.loads((Path(root) / ROSTER).read_text(encoding='utf8'))
    if set(roster) != set(map(str, range(50))):
        raise ValueError('Unexpected roster IDs')
    labels = pd.DataFrame(roster).T
    return {a: labels[a].value_counts().sort_index() for a in ('political orientation', 'religion', 'gender')}, labels


def assert_primary_only(root, run_ids, expect=896):
    """Refuse any table that is not exactly the original primary cohort."""
    root = Path(root)
    primary = pd.read_csv(root / PRIMARY_IDS)
    supplement = pd.read_csv(root / SUPPLEMENT_IDS)
    ids = set(run_ids)
    if (len(primary) != 896 or int(primary.valid_graph.sum()) != 890 or not primary.cohort.eq('primary').all()
            or ids & set(supplement.run_id) or ids & set(supplement.original_run_id) - set(primary.run_id)
            or not ids <= set(primary.run_id) or len(ids) != expect):
        raise ValueError('Table is not restricted to the 896 original primary attempts')
    return primary


def load(root):
    """Verify every historical seal first; any changed byte stops the release."""
    root = Path(root).resolve()
    a, primary, tables, held, protected = v3.load_tables(root)
    if sha(root / v5.OUT / 'manifest.json') != V5_MANIFEST_SHA256:
        raise ValueError('Unexpected v5 manifest')
    v5.verify(root)
    v3_manifest = json.loads((root / v3.OUTPUT / 'figure_manifest.json').read_bytes())
    v2.verify_hashes({root / v3.OUTPUT / p: h for p, h in v3_manifest['artifacts'].items()})
    seal_path = root / EXTENSION / 'manifest.json'
    if sha(seal_path) != EXTENSION_MANIFEST_SHA256:
        raise ValueError('Unexpected extension manifest')
    seal = json.loads(seal_path.read_text())
    v2.verify_hashes({root / EXTENSION / n: h for n, h in seal['artifacts'].items()})
    if [seal[k] for k in ('primary_attempts', 'primary_valid', 'primary_empty', 'primary_failed',
                          'supplemental_graphs', 'paid_calls', 'ledger_queries')] != [896, 890, 12, 6, 0, 0, 0]:
        raise ValueError('Extension cohort differs')
    ext = {n: pd.read_csv(root / EXTENSION / (n + '.csv')) for n in EXTENSION_TABLES}
    ext['copenhagen_source'] = json.loads((root / EXTENSION / 'copenhagen_source.json').read_text())
    for name in ('model_contrasts_descriptive', 'coverage_by_cell'):
        tables[name] = pd.read_csv(root / v2.PRIMARY / (name + '.csv'))
    v5_tables = {n: v5.table(root, n) for n in ('efficiency_by_density', 'attribute_cells')}
    sizes, labels = roster_sizes(root)
    # Cohort guards: every run-level table must be the 896 primary attempts and nothing else.
    assert_primary_only(root, primary.run_id)
    assert_primary_only(root, ext['mixing_scores'].run_id)
    assert_primary_only(root, held['candidate_metrics'].query("kind == 'LLM'").run_id)
    assert_primary_only(root, tables['synthetic_reference_per_run'].run_id)
    llm = ext['candidate_size_metrics'].query("kind == 'LLM'")
    assert_primary_only(root, llm.run_id)
    if (int(primary.valid_graph.sum()), int(primary.empty_graph.fillna(False).astype(bool).sum())) != (890, 12):
        raise ValueError('Primary outcome counts differ')
    inputs = {p.relative_to(root).as_posix(): h for p, h in protected.items()}
    inputs.update({(EXTENSION / n).as_posix(): h for n, h in seal['artifacts'].items()})
    inputs[(EXTENSION / 'manifest.json').as_posix()] = EXTENSION_MANIFEST_SHA256
    inputs[(v5.OUT / 'manifest.json').as_posix()] = V5_MANIFEST_SHA256
    for folder in (v3.OUTPUT, Path('outputs/publication_figures/20261111_v4'), v5.OUT):
        for path in sorted((root / folder).iterdir()):
            if path.is_file():
                inputs[path.relative_to(root).as_posix()] = sha(path)
    for rel in (PRIMARY_IDS, SUPPLEMENT_IDS):
        inputs[rel.as_posix()] = sha(root / rel)
    return dict(root=root, spec=a, primary=primary, tables=tables, held=held, ext=ext, v5=v5_tables,
                roster_sizes=sizes, roster=labels, inputs=inputs)
