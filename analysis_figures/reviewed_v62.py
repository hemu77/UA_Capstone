"""Presentation-only successor to v6.1; plotted data are copied, never refitted."""
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis_figures import reviewed_v6 as prior

OUT = Path('outputs/publication_figures/20261118_v6_2')
PIN = '42cd5632f176b03eac472eea62434db62d8409508fa1a1b5c68cf723f70aa8fb'
sha = prior.original.sha
CAPTIONS = {k: prior.caption(k) for k in prior.CAPTIONS}
CAPTIONS['02_primary_robustness'] = (
    'Iterative US-Hindi minus US-English clustering: all eight paired differences are negative. '
    'a: archived simultaneous interval. b: raw and Holm p after omitting each block; the direction '
    'never changes, but only 3/8 dependent refits pass Holm. c: the exact two-sided sign-flip '
    'floor (2/256) exceeds .05/24. It cannot confirm familywise significance; this is not a reversal.'
    + prior.original.COHORT)
CAPTIONS['12_cohort_sensitivity'] += (
    ' Global clustering distance rises from 0.096 at FB09 (density 0.032) to 0.417 at FB06 '
    '(0.126); these are descriptive endpoints, not a monotone or causal law.')
CAPTIONS['19_copenhagen_transfer'] = (
    'Sampling stress test, not a ranking of friendship realism. The 32 overlapping draws of 50 '
    'endpoints average about 25 edges and 47% isolates. Five structural distances are shown; '
    'largest-component distances remain in the accompanying CSV. Ticks are models or banks, '
    'not confidence intervals. Density-informed ER is a diagnostic, not a transfer competitor.'
    + prior.original.COHORT)
caption = CAPTIONS.__getitem__
CHANGED = {'02', '10', '15', '18', '19'}


def revise(name, fig, table):
    axes = fig.axes
    if name == '02':
        fig.texts[-1].set_text('a: band = simultaneous 95% interval. b: overlapping refits are not replications.\n'
                               'c: ticks show attainable two-sided sign-flip p-values up to 0.25; '
                               'the minimum exceeds the first Holm cutoff.')
        ax = axes[1]
        fig.set_size_inches(9.4, 3.45)
        ax.set_title('Refits after omitting one block', loc='left')
        ax.plot(table.loo_raw_p, range(8), 'x', ms=4, color='#253444')
        ax.set_xlim(.00004, .2)
        ax.set_xlabel('Cross: raw p    Circle: Holm p\n(log scale)')
        for tick in axes[2].get_yticklabels():
            tick.set_visible(False)
        axes[2].set_yticks([0, 1], [])
        for label in axes[2].texts:
            if label.get_text().startswith('raw p'):
                label.set_text(('Paired t: ' if label.get_position()[1] == 0 else 'Exact: ') + label.get_text())
                label.set_bbox(dict(facecolor='white', edgecolor='none', pad=.4))
    if name == '10':
        axes[0].set_ylim(-.06, table.max_ratio.max() * 1.06)
        fig.texts[-1].set_text('Full range shown for every record; bars span 32 overlapping draws (not a CI). '
                               'Ratio 1 means the draw retains the full-record value.')
    if name == '15':
        for t in axes[1].texts:
            if t.get_text() == 'School':
                t.set_visible(False)  # The diamond is uniquely identified in the legend.
        for line in axes[0].lines:
            if line.get_marker() in ('o', 'D'):
                shift = .13 if line.get_marker() == 'o' else -.13
                line.set_ydata([y + shift for y in line.get_ydata()])
    if name == '18':
        fig.set_size_inches(8.5, 5.2)
        fig.subplots_adjust(bottom=.29)
        fig.legends[0].set_bbox_to_anchor((.5, .12))
        for ax in axes:
            ax.set_xlim(-.025 * ax.get_xlim()[1], None)
    if name == '19':
        ax = axes[-1]
        if 'component' not in ax.get_title(loc='left').lower():
            raise ValueError('Unexpected panel f')
        ax.clear()
        ax.set_axis_off()
        ax.text(.02, .85, 'What is being matched?', weight='bold', transform=ax.transAxes)
        ax.text(.02, .7, '50 sampled endpoints\n~25 observed edges per draw\n47% isolated nodes\n\n'
                'Fit to a sparse sample,\nnot a whole community.',
                va='top', linespacing=1.5, transform=ax.transAxes)
        for ax in axes[:-1]:
            ax.set_xlim(-.025 * ax.get_xlim()[1], None)


def build(root, dest):
    import matplotlib.pyplot as plt
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest.exists() or dest.parent != root / OUT.parent:
        raise ValueError('Use a new direct child of outputs/publication_figures')
    source = prior.verify(root)
    if sha(root / prior.OUT / 'manifest.json') != PIN:
        raise ValueError('Unexpected parent release')
    with TemporaryDirectory(dir=dest.parent) as temp:
        folder = Path(temp) / 'release'
        shutil.copytree(root / prior.OUT, folder)
        data = prior.data.load(root)
        with plt.rc_context({**prior.style.STYLE, 'svg.fonttype': 'path'}):
            for stem, draw in prior.original.figures().items():
                if stem[:2] not in CHANGED:
                    continue
                fig, table = draw(data)
                try:
                    revise(stem[:2], fig, table)
                    for ext, meta in [('png', {'Software': None}), ('svg', {'Date': None}),
                                      ('pdf', {'CreationDate': None, 'ModDate': None})]:
                        fig.savefig(folder / f'{stem}.{ext}', dpi=300, metadata=meta)
                finally:
                    plt.close(fig)
        manifest = dict(source, release=OUT.name, status='COMPLETE_FIGURE_RELEASE_V6_2',
                        source_manifest_sha256=PIN, captions=CAPTIONS,
                        code_sha256={'analysis_figures/reviewed_v62.py': sha(Path(__file__))},
                        artifacts={n: sha(folder / n) for n in source['artifacts']})
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        folder.rename(dest)


def verify(root, inputs=True):
    root = Path(root)
    source = prior.verify(root, inputs=inputs)
    folder = root / OUT
    manifest = json.loads((folder / 'manifest.json').read_text())
    expected = dict(source, release=OUT.name, status='COMPLETE_FIGURE_RELEASE_V6_2',
                    source_manifest_sha256=PIN, captions=CAPTIONS,
                    code_sha256={'analysis_figures/reviewed_v62.py': sha(Path(__file__))},
                    artifacts={n: sha(folder / n) for n in source['artifacts']})
    if (manifest != expected or sha(root / prior.OUT / 'manifest.json') != PIN or
            {p.name for p in folder.iterdir()} != set(source['artifacts']) | {'manifest.json'} or
            any(expected['artifacts'][n] != digest for n, digest in source['artifacts'].items()
                if n[:2] not in CHANGED or n.endswith('.csv'))):
        raise ValueError('Unexpected v6.2 release or modified historical data')
    return manifest


def show(root, names):
    from IPython.display import Image, Markdown, display
    verify(root, inputs=False)
    for name in names:
        text = caption(name)
        display(Image(filename=str(Path(root) / OUT / (name + '.png'))))
        display(Markdown(text))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(Path(__file__).resolve().parents[1], parser.parse_args().output)
