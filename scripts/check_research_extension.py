"""Build or verify the executed extension notebook against its sealed figures."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures import reviewed_v62 as release
from scripts.check_figure_release_v6 import BANNER, EXPECTED, embedded
OUT = ROOT/'outputs/research_extension/20261114_v2'
NOTEBOOK = ROOT/'analysis/06_rq2_sampling_copenhagen.ipynb'


def notebook():
    seal = json.loads((OUT/'manifest.json').read_text())
    pin = hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()
    cells = []
    def md(text, name):
        cells.append(nbformat.v4.new_markdown_cell(text, id=name))
    def code(text, name):
        cells.append(nbformat.v4.new_code_cell(text, id=name))
    def figure(name):
        code(f"show(root, ['{name}'])", name)
    md('''# RQ2 mixing, sampling and Copenhagen

**Exploratory extension, not new confirmatory tests.** Only the original 896 attempts enter:
890 valid graphs (including 12 empty), six failures, **zero supplements**. The locked 24-test
family is unchanged. No paid calls or ledger queries. This notebook reads versioned aggregate
tables; no public node-level data or provider credentials are required.

[Plain-language findings](../docs/research/EXTENSION_RESULTS_20261114.md) |
[Protocol](../analysis_extension/protocol.json) | [Original RQ2 figure](05_rq2_demographic_mixing.ipynb)

The Copenhagen protocol was written after study collection, before comparison outcomes.
A raw-inventory check found 11 self-links; the recorded pre-comparison cleaning amendment
removes those only. This is **not preregistered** and not proof of held-out model training data.''', 'scope')
    md(BANNER, 'figure-release-v6')
    code(f'''from pathlib import Path
import hashlib, json, sys, warnings
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
warnings.filterwarnings('ignore', category=DeprecationWarning)
root = Path.cwd()
if not (root/'analysis_extension').is_dir():
    root = root.parent
sys.path.insert(0, str(root))
from analysis_figures.reviewed_v62 import show
folder = root/'outputs/research_extension/20261114_v2'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(folder/'manifest.json') == '{pin}'
seal = json.loads((folder/'manifest.json').read_text())
assert sys.version_info[:3] == (3, 11, 9)
for name, expected in seal['artifacts'].items():
    assert digest(folder/name) == expected, name
for name, expected in seal['code_sha256'].items():
    assert digest(root/name) == expected, name
assert [seal[k] for k in ['primary_attempts','primary_valid','primary_empty','primary_failed','supplemental_graphs','paid_calls','ledger_queries']] == [896,890,12,6,0,0,0]
scores = pd.read_csv(folder/'mixing_scores.csv')
primary = pd.read_csv(root/'outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv')
assert set(scores.run_id) == set(primary.original_run_id)
assert scores.groupby('run_id').size().eq(3).all()
assert scores.status.value_counts().to_dict() == {{'nonempty':2634, 'empty':36, 'failed':18}}
print('Sealed primary-only extension verified; original failures and empty graphs retained.')''', 'integrity')
    md('''## 1. RQ2: compare procedures on the same cases

Excess edge share = observed same-attribute edge fraction minus finite-roster random-pair
chance. A difference of 0.10 is ten percentage points of excess edge share, **not** ten percent
more accurate. Each procedure pair first matches model, setting and seed. We average paired
seed differences within each cell, then give the 28 cells equal weight. Missing a whole cell
suppresses the overall estimate. Empty graphs have undefined mixing scores, not zero scores.

The archived Coleman-type score uses roster proportions; Newman uses edge-end proportions. Different denominators
answer different questions. Newman does not remove roster or prompt-order confounding.''', 'mixing-intro')
    code("""coverage = pd.read_csv(folder/'mixing_coverage.csv')
display(coverage)
contrasts = pd.read_csv(folder/'mixing_method_sensitivity.csv')
main = contrasts[contrasts.omitted_seed.eq(-1) & contrasts.score.eq('excess') & contrasts.support.eq('score_specific')]
assert main.planned_cells.eq(28).all() and main.defined_cells.eq(28).all()
display(main[['attribute','first','second','mean_difference','defined_pairs','defined_cells','planned_cells']].round(4))""", 'coverage')
    figure('16_rq2_paired_mixing')
    md('''## 2. RQ2: inspect mixing, not just attribute winners

The diagonal contains within-category ties. Each undirected edge counts in both directions.
The archived edge fractions sum to one; Figure 17 divides each mean fraction by its
finite-roster chance fraction. The displayed ratios do NOT sum to one: 1 means chance,
above 1 means more ties than roster group sizes alone predict. Do not rank attributes by
panel darkness. Numerators, denominators and ratios are in the released plotted CSV.

Rank comparisons below use the same defined seeds across attributes **and all three scores**.
Seed omission is a sensitivity check, not eight replications or a confidence interval.''', 'matrices-intro')
    figure('17_rq2_mixing_matrices')
    code("""ranks = pd.read_csv(folder/'mixing_rank_sensitivity.csv')
shared = ranks[ranks.support.eq('all_three_scores')]
main_ranks = shared[shared.omitted_seed.eq(-1)]
assert main_ranks.groupby(['method','score']).size().eq(28).all()
display(main_ranks.groupby(['method','score','winner']).size().rename('cells').unstack('winner',fill_value=0))
stability = shared.assign(politics_first=shared.winner.eq('political orientation')).groupby(['method','score','omitted_seed']).agg(politics_first=('politics_first','sum'),defined_cells=('winner',lambda x: x.ne('undefined').sum()))
display(stability.reset_index().query('omitted_seed != -1').groupby(['method','score']).agg(min_politics_first=('politics_first','min'),max_politics_first=('politics_first','max'),min_defined_cells=('defined_cells','min')))
common = contrasts[contrasts.omitted_seed.eq(-1) & contrasts.support.eq('all_three_scores')]
display(common[['attribute','score','first','second','mean_difference','defined_pairs','defined_cells']].round(4))""", 'rank-sensitivity')
    md('''## 3. Sampling: a small induced graph is not a scaled copy

We recompute all six metrics at 25, 40 and 50 nodes. The smaller draws are nested inside the
same 50-node draw; isolates remain. Overlap uses original public IDs before relabeling.
Full records are descriptive references, not complete population truth. Ranges below are
overlapping-draw variation, **not CIs**. Historical n=50 metrics/W1 and the older three-metric
n=25/40/50 sensitivity are reproduced to absolute tolerance 1e-12.''', 'sampling-intro')
    figure('18_sampling_distortion')
    code("""full = pd.read_csv(folder/'full_network_metrics.csv')
samples = pd.read_csv(folder/'observed_sample_metrics.csv')
assert len(samples) == 12*32*3
assert samples.groupby(['source','cohort','n']).size().eq(32).all()
sample_means = samples[samples.n.eq(50)].groupby(['source','cohort'])[['density','avg_clustering_coef','global_efficiency','isolates']].mean().add_prefix('sample50_')
comparison = full.merge(sample_means,on=['source','cohort'],validate='one_to_one')
display(comparison[['cohort','nodes','avg_clustering_coef','sample50_avg_clustering_coef','global_efficiency','sample50_global_efficiency','sample50_isolates']].round(4))
overlap = pd.read_csv(folder/'sample_overlap.csv')
assert len(overlap) == 12*3*496
display(overlap[overlap.n.eq(50)].groupby('cohort').shared_fraction.agg(['mean','min','max']).round(4))""", 'sampling-tables')
    md('''## 4. Copenhagen: one additional structural test, with a restricted roster

The versioned MIT-licensed release has 800 friendship endpoints and 6,429 raw edge records.
Eleven self-links are excluded: **800 nodes, 6,418 simple edges**, zero observed isolates.
This does not prove that unobserved people had no friends. The released gender file intersects
747 friendship endpoints; gender-only IDs are not assumed to be observed friendship isolates.
Politics/religion are unavailable, so this does **not validate RQ2 against human behavior**.

ER/BA/WS settings stay fixed from the old development panel. W1 compares valid-output
distributions, gives seven settings equal weight within each model, then averages four
model-specific distances. It is **not W1 of pooled models**. The separate target-density ER
has target information and cannot be called a fair transfer winner. All six metrics and all
three sample sizes remain available, including largest-component fraction.''', 'copenhagen-intro')
    code("""source = json.loads((folder/'copenhagen_source.json').read_text())
assert (source['raw_edge_rows'],source['removed_self_links'],source['observed_edges'],source['observed_roster_nodes']) == (6429,11,6418,800)
assert not source['full_network_isolates_observable']
display(pd.Series({k:source[k] for k in ['doi','observed_roster_nodes','observed_edges','removed_self_links','friendship_nodes_with_gender','gender_only_ids','expected_edges_in_50','suitability']}))
transfer = pd.read_csv(folder/'transfer_record_summary.csv')
diagnostic = pd.read_csv(folder/'target_er_summary.csv')
selected = pd.concat([transfer[transfer.source.eq('copenhagen_friendship')],diagnostic],ignore_index=True)
display(selected.pivot(index=['n','kind','procedure'],columns='metric',values='mean_w1').round(4))""", 'copenhagen-tables')
    figure('19_copenhagen_transfer')
    md('''## What to say, and what not to say

- We can describe how the four procedures mix the fixed roster differently, and show whether
  the answer survives scoring and paired-support changes. We cannot infer independent causes
  or human demographic preferences from these prompts.
- We can show how induced sampling changes observed structure and candidate comparisons.
  Overlapping draws, overlapping Facebook records and one Copenhagen cohort are not many
  independent social populations.
- We can compare against Copenhagen without fitting our transfer baselines to it. A close
  W1 on one metric is not network accuracy, predictive utility or a universal ranking.
- All original failures remain visible. Six selectively completed supplements never enter.

### Method references
[Newman: Mixing patterns](https://arxiv.org/abs/cond-mat/0209450) motivates the edge-end sensitivity;
[Leskovec and Faloutsos: Sampling from Large Graphs](https://cs.stanford.edu/~jure/pubs/sampling-kdd06.pdf)
motivates explicit size/sampling checks. Neither makes our sampling representative.
[Copenhagen descriptor](https://doi.org/10.1038/s41597-019-0325-x) and
[versioned dataset](https://doi.org/10.6084/m9.figshare.7267433.v1) establish source provenance.
[Next research steps](../docs/research/NEXT_STEPS_20261114.md) relate this extension to the five
professor-supplied papers. This is AI-assisted analysis, not peer review or publication acceptance.''', 'limits')
    return nbformat.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(
        display_name='UA Capstone (frozen Python 3.11)', language='python', name='ua-capstone-final060d')))


def check_outputs(nb):
    embedded(nb, release.verify(ROOT), EXPECTED['06'])


def check(write=False):
    expected = notebook()
    nb = expected if write else nbformat.read(NOTEBOOK, as_version=4)
    if not write:
        if [(c.cell_type,c.source) for c in nb.cells] != [(c.cell_type,c.source) for c in expected.cells]:
            raise ValueError('Notebook source differs from generator')
        check_outputs(nb)
    NotebookClient(nb, timeout=300, kernel_name='ua-capstone-final060d',
                   resources={'metadata':{'path':str(ROOT / 'analysis')}}, allow_errors=False).execute()
    check_outputs(nb)
    if write:
        with NOTEBOOK.open('w', encoding='utf8', newline='\n') as stream:
            nbformat.write(nb, stream)
        check_outputs(nbformat.read(NOTEBOOK, as_version=4))
    print('06: frozen-kernel execution passed; four exact PNGs/captions; primary-only checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Generate and execute notebook 06 only.')
    check(parser.parse_args().write)
