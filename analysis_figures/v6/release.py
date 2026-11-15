"""Build, seal, verify and display figure release v6 (all 19 current figures)."""
import hashlib
import importlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

OUT = Path('outputs/publication_figures/20261115_v6')
MODULES = ('primary', 'heldout', 'sampling', 'mixing')
STATUS = 'COMPLETE_FIGURE_RELEASE_V6_PRESENTATION_ONLY'
COHORT = (' Primary cohort only: 896 attempts, 890 valid graphs including 12 empty, six failures; '
          'the six supplementary graphs are excluded.')
CAPTIONS = {
    '01_primary_contrasts': 'All 24 planned instruction contrasts on one axis, grouped by procedure (Global: density; others: mean clustering). Circle: mean of complete paired seed blocks. Line: archived Bonferroni simultaneous 95% t interval for the 24-contrast family. Holm p: step-down adjusted p-value for the same family; the interval and the Holm p-value are different procedures and agree only for the smallest p. Small dots: the individual paired blocks (four-model means). A cross marks the contrast with fewer than six complete blocks, which was not tested. A non-significant contrast is not evidence of no difference.',
    '02_primary_robustness': 'The one contrast that passed the planned correction (Iterative, US-Hindi minus US-English, mean clustering). a: the eight paired blocks; band: archived simultaneous interval. b: Holm-adjusted p after refitting without each block in turn (3 of 8 stay below .05; the refits share seven blocks and are not replications). c: every p-value an exact eight-block sign-flip test can return; its smallest value, 2/256, is above the first Holm threshold .05/24, so that test cannot reject any contrast in this family and its adjusted p of .1875 is the floor, not evidence against the effect.',
    '03_model_profiles': 'Model-specific mean differences for the same 24 contrasts, over the seed blocks common to all four models. Vertical bar: four-model mean. Right columns: how many of the four fixed models share the sign of the difference, and the range across models. Descriptive only: no intervals or tests; four fixed models are not a sample of models. Bold marks all four in the same direction, not statistical significance.',
    '04_coverage_missingness': 'a: every one of the 896 primary attempts as one square (seeds left to right within each instruction setting). b, c: the three planned contrasts that lost blocks to failures. Dot: mean of complete blocks. Bar: range the block mean could take if every missing endpoint lay anywhere in [0, 1]; these are identification bounds, not confidence intervals. Failures are never zero-filled or replaced.',
    '05_resource_use': 'Requests, estimated received-usage cost and serial provider seconds per attempt, on log axes, for each procedure and model (56 attempts each, failures included). The multiplier is the median ratio to Global across the four models. Procedures were not given equal computation, so differences between procedures are not compute-matched comparisons.',
    '06_gnm_context': 'For every valid primary graph: observed mean clustering minus the mean clustering of eight random graphs with the same 50 nodes and the same edge count. Tick: one graph. Circle: median. Thick line: interquartile range. Thin line: 5th to 95th percentile. n: graphs with a defined reference. Descriptive context, not a test and not a comparison with human networks.',
    '07_legacy_comparison': 'Valid primary graphs (50 nodes) beside the eight archived small friendship networks used to fix the baselines (31 to 70 nodes; Tailor 1 and 2 are the same workers). Shaded band: range of the eight networks. Sizes and tie definitions differ, so position inside the band is context, not validation.',
    '08_heldout_distances': 'Wasserstein-1 distance between generated graphs and 50-node draws from 11 evaluation-held-out records, one panel per metric. Tick: one of ten overlapping Facebook records. Circle: their mean. Diamond: the single school record. LLM rows average four model-specific distances; ER, BA and WS average eight simulation banks with settings fixed on the earlier development networks. Smaller is closer on that metric only; panels have different units and are not combined into a score. Record spread is not a confidence interval.',
    '09_heldout_distributions': 'The distributions behind figure 08: density, clustering and efficiency of observed 50-node draws and of every generator. Tick: one graph. Circle: weighted median. Thick line: interquartile range. Thin line: 5th to 95th percentile. Records weighted equally; LLM graphs weighted equally by model and instruction setting, conditional on a valid graph.',
    '10_sampling_effect': 'Mean of 32 overlapping 50-node induced draws divided by the value of the full observed record, against the share of the record contained in a draw. Line: range of the 32 draws, not a confidence interval. Density is retained on average; clustering and efficiency fall as the draw becomes a smaller share of the record. Twelve records: ten Facebook, one school, one Copenhagen (endpoint-conditioned).',
    '11_conditional_controls': 'Observed 50-node draw minus the mean of eight controls rebuilt from that same draw: random graphs with the same edge count (squares) and degree-preserving edge swaps (circles). Ticks: the 32 overlapping draws. Controls use target information and are diagnostics, not transfer competitors. Positive values in every record describe closure these controls do not reproduce; no test is implied.',
    '12_cohort_sensitivity': 'a: clustering W1 of each procedure for each held-out record, records ordered by full-record density. b: gap between the closest and second-closest procedure, with the closest named. The ordering of procedures changes from record to record and several gaps are near ties; no procedure is closest throughout. Lines join separate records only to show changes of order.',
    '13_appendix_lcc': 'Appendix. a: share of the 50 nodes in the largest connected component for observed draws and every generator. b: the resulting W1 distances. Almost every ER, BA and WS graph is fully connected, so their distances are nearly identical: this metric mainly reflects fragmentation of sampled observed draws, not differences between baselines.',
    '14_demographic_mixing': 'a: share of ties within a group minus the share expected from roster group sizes, for each procedure and attribute; tick: one model-by-instruction cell (28 per row); marker: equal-weight mean. b: politics score minus the larger of the religion and gender scores for each cell, under three scores, with the number of cells in which politics is first. Political orientation is first in 28 of 28 cells for each person-level procedure and in 1 to 6 of 28 for Global. Descriptive: one fixed roster, correlated attributes (Cramer V 0.32 politics-religion) and one fixed prompt field order; not causal importance or human preference. 878 non-empty graphs.',
    '15_efficiency_density': 'a: efficiency W1 for each held-out record, ordered by full-record density: Global LLM, the fixed WS baseline, and a diagnostic ER graph given that record density. b: Global minus the density-informed ER against record density. The diagnostic holds target information and is not a ranked competitor. Global is closer than the informed ER mainly on denser records and farther on four of the five sparsest; post hoc description of 11 overlapping records, not a fitted trend or test.',
    '16_rq2_paired_mixing': 'Second procedure minus first, in excess same-group tie share, paired on model, instruction setting and seed before averaging. Tick: mean paired difference in one of 28 cells. Circle: equal-weight mean of the 28 cells. The count gives how many cells share the sign of the mean. Pairs with an empty or failed graph are missing, not replaced (206 to 224 pairs). Cell spread is not a confidence interval; no new test.',
    '17_rq2_mixing_matrices': 'Mean share of tie ends joining each pair of groups, divided by the share expected from roster group sizes alone (group sizes in parentheses); 1.0 is chance. Each undirected tie counts in both directions. Color is centered on chance in every matrix, so it compares group pairs within one attribute and procedure; a darker attribute is not a more important attribute. Non-empty graphs averaged within cells, 28 cells weighted equally.',
    '18_sampling_distortion': 'Full observed record (open diamond) against the mean and range of 32 overlapping induced draws at 25, 40 and 50 nodes, for 12 records ordered by size. Smaller draws are nested in the 50-node draw; nodes isolated by sampling are kept. Ranges are sampling variation, not confidence intervals. The Copenhagen release lists only people with at least one friendship, so its full-record isolate count of zero is a property of the release, not of the cohort.',
    '19_copenhagen_transfer': 'Copenhagen (800 endpoints, 6,418 friendships after removing 11 self-links from 6,429 raw rows) at 50 nodes: W1 for six metrics. Tick: one model (LLM rows) or one simulation bank. Marker: mean. The shaded bottom row is an ER graph given Copenhagen density: a target-informed diagnostic, never a transfer competitor. No procedure is closest on every metric; Global is closer than the fixed baselines on density, efficiency and transitivity, fixed ER is closer on clustering. One endpoint-conditioned record; no political or religious labels; not validation of RQ2.',
}
MAIN = ('01_primary_contrasts', '02_primary_robustness', '04_coverage_missingness', '08_heldout_distances',
        '15_efficiency_density', '14_demographic_mixing', '18_sampling_distortion')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figures():
    table = {}
    for module in MODULES:
        table.update(importlib.import_module('analysis_figures.v6.' + module).FIGURES)
    if sorted(table) != sorted(CAPTIONS) or len(table) != 19:
        raise ValueError('Release must contain exactly the 19 current figures')
    return dict(sorted(table.items()))


def code_files(root):
    return sorted((Path(root) / 'analysis_figures/v6').glob('*.py'))


def build(root, destination):
    """Render all 19 figures into a NEW directory; never overwrite a release."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis_figures.v6 import data, style
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(root / 'outputs/publication_figures'):
        raise ValueError('Use a NEW directory inside outputs/publication_figures; no overwrites')
    D = data.load(root)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        folder = Path(temporary) / 'release'
        folder.mkdir()
        rows = {}
        with plt.rc_context(style.STYLE):
            for name, function in figures().items():
                fig, table = function(D)
                fig.savefig(folder / (name + '.png'), dpi=300, metadata={'Software': None})
                fig.savefig(folder / (name + '.svg'), metadata={'Date': None})
                fig.savefig(folder / (name + '.pdf'), metadata={'CreationDate': None, 'ModDate': None, 'Producer': None,
                                                                 'Creator': None})
                plt.close(fig)
                table.to_csv(folder / (name + '_plotted.csv'), index=False, lineterminator='\n')
                rows[name] = len(table)
        D2 = data.load(root)  # every input must be byte-identical after rendering
        if D2['inputs'] != D['inputs']:
            raise ValueError('A sealed input changed during rendering')
        manifest = dict(status=STATUS, release='20261115_v6', python_version=platform.python_version(),
                        matplotlib_version=matplotlib.__version__, figures=19, main_text_proposal=list(MAIN),
                        plotted_rows=rows, captions=CAPTIONS, paid_calls=0, ledger_queries=0, new_generation=0,
                        new_hypothesis_tests=0, refitted_baselines=0, **data.COUNTS,
                        artifacts={p.name: sha(p) for p in sorted(folder.iterdir())},
                        code_sha256={p.relative_to(root).as_posix(): sha(p) for p in code_files(root)},
                        input_sha256=D['inputs'])
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf8')
        folder.rename(destination)
    return manifest


def verify(root, folder=OUT, inputs=True):
    """Check the release bytes, its code pins and (optionally) every sealed input."""
    root = Path(root).resolve()
    manifest = json.loads((root / folder / 'manifest.json').read_text(encoding='utf8'))
    counts = [manifest[k] for k in ('primary_attempts', 'primary_valid', 'primary_empty', 'primary_failed',
                                    'supplemental_graphs', 'paid_calls', 'ledger_queries', 'new_generation')]
    if manifest['status'] != STATUS or counts != [896, 890, 12, 6, 0, 0, 0, 0] or manifest['captions'] != CAPTIONS:
        raise ValueError('Unexpected v6 manifest')
    expected = {n + e for n in CAPTIONS for e in ('.png', '.svg', '.pdf', '_plotted.csv')}
    present = {p.name for p in (root / folder).iterdir()} - {'manifest.json'}
    if set(manifest['artifacts']) != expected or present != expected:
        raise ValueError('Release files differ from the manifest')
    for name, digest in manifest['artifacts'].items():
        if sha(root / folder / name) != digest:
            raise ValueError('Release artifact differs: ' + name)
    for rel, digest in manifest['code_sha256'].items():
        if sha(root / rel) != digest:
            raise ValueError('Renderer differs from the release: ' + rel)
    if inputs:
        from analysis_figures.v6 import data
        if data.load(root)['inputs'] != manifest['input_sha256']:
            raise ValueError('A sealed input differs from the release record')
    return manifest


def show(root, names):
    """Display released PNG bytes and their captions in a notebook; nothing is redrawn."""
    from IPython.display import Image, Markdown, display
    manifest = verify(root, inputs=False)
    for name in names:
        path = Path(root) / OUT / (name + '.png')
        if sha(path) != manifest['artifacts'][name + '.png']:
            raise ValueError('Figure differs from the release: ' + name)
        display(Image(filename=str(path)))
        display(Markdown(CAPTIONS[name] + COHORT))
