"""Review errata over immutable v6: corrected figure 16 and notebook captions."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from analysis_figures.v6 import data, release as original, style
from analysis_figures.v6.mixing import fig16 as original_fig16, fig17 as original_fig17

OUT = Path('outputs/publication_figures/20261115_v6_1')
SOURCE_SHA = '9b30c090f1a8997cb9d4c52cfee3ddf29cdb00f9d164b2cae8ed638e85379de1'
CHANGED = ('16_rq2_paired_mixing', '17_rq2_mixing_matrices')
STATUS = 'COMPLETE_FIGURE_RELEASE_V6_1_REVIEWED'
CAPTIONS = dict(original.CAPTIONS)
CAPTIONS['01_primary_contrasts'] = CAPTIONS['01_primary_contrasts'].replace(
    'different procedures and agree only for the smallest p',
    'different procedures: they use the same threshold for the smallest p but can differ for later ranks')
CAPTIONS['03_model_profiles'] = CAPTIONS['03_model_profiles'].replace(
    'how many of the four fixed models share the sign of the difference',
    'the larger of the positive and negative model counts (not agreement with the mean)')
CAPTIONS['08_heldout_distances'] += (
    ' Efficiency rankings partly reflect density mismatch: read with Figure 15, which separately shows'
    ' the target-density ER diagnostic. That diagnostic is not a transfer competitor.')


def caption(name):
    if name in ('10_sampling_effect', '18_sampling_distortion'):
        context = ' Observed public networks only; no generated primary or supplemental graphs enter this figure.'
    elif name == '11_conditional_controls':
        context = ' Observed public networks and synthetic conditional controls only; no LLM graphs or supplements.'
    else:
        context = original.COHORT
    return CAPTIONS[name] + context


def verify_history(root):
    """Include the v2 and extension source pins omitted by the original v6 loader."""
    root = Path(root)
    source = root / original.OUT / 'manifest.json'
    if original.sha(source) != SOURCE_SHA:
        raise ValueError('Original v6 manifest differs')
    original.verify(root)
    extension = json.loads((root / data.EXTENSION / 'manifest.json').read_text())
    old = json.loads((root / 'outputs/publication_figures/20261105_v2/figure_manifest.json').read_text())
    pins = {}
    for mapping in (extension['preserved_sha256'], extension['code_sha256'],
                    old['protected_sha256'], old['code_sha256']):
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError('Conflicting historical pin: ' + name)
            pins[name] = digest
    for name, digest in pins.items():
        if original.sha(root / name) != digest:
            raise ValueError('Historical input differs: ' + name)
    return pins


def fig16(D):
    # Retain the sealed layout; only the sign-agreement annotation and footnote change.
    fig, table = original_fig16(D)
    cells = D['ext']['mixing_paired_cells']
    cells = cells[cells.score.eq('excess') & cells.support.eq('score_specific')]
    for ax, attribute in zip(fig.axes, style.ATTRIBUTES):
        annotations = [t for t in ax.texts if t.get_text().endswith('/28 same sign')]
        rows = table[table.attribute.eq(attribute)]
        if len(annotations) != len(rows):
            raise ValueError('Unexpected figure 16 annotation layout')
        for text, (index, row) in zip(annotations, rows.iterrows()):
            values = cells[cells.attribute.eq(attribute) & cells['first'].eq(row['first']) &
                           cells.second.eq(row.second)].difference
            mean = values.mean()
            same = int(((values > 0) if mean > 0 else (values < 0) if mean < 0 else (values == 0)).sum())
            table.loc[index, 'cells_same_sign'] = same
            text.set_text(f'{same}/28 same sign')
            text.set_fontweight('bold' if same == 28 else 'normal')
            text.set_color(style.INK if same == 28 else style.MUTED)
    for text in fig.texts:
        text.set_text(text.get_text().replace('failed Global graph', 'failed graph'))
    return fig, table


def fig17(D):
    import numpy as np
    # Vector cells avoid low-resolution image decoding/interpolation artifacts in PDF viewers.
    fig, table = original_fig17(D)
    images = [image for ax in fig.axes for image in ax.images]
    if len(images) != 12:
        raise ValueError('Unexpected figure 17 matrix inventory')
    for image in images:
        ax = image.axes
        limits = ax.get_xlim(), ax.get_ylim()
        values = image.get_array()
        mesh = ax.pcolormesh(np.arange(values.shape[1] + 1) - .5,
                             np.arange(values.shape[0] + 1) - .5, values,
                             cmap=image.get_cmap(), norm=image.norm, shading='flat',
                             rasterized=False, antialiased=False, zorder=image.get_zorder())
        mesh.set_gid('categorical-mixing-cells')
        image.remove()
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    return fig, table


def build(root, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(root / 'outputs/publication_figures'):
        raise ValueError('Use a NEW directory inside outputs/publication_figures; no overwrites')
    pins = verify_history(root)
    source = original.verify(root)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        folder = Path(temporary) / 'release'
        folder.mkdir()
        for name in source['artifacts']:
            if not name.startswith(CHANGED) and not name.endswith('.svg'):
                shutil.copyfile(root / original.OUT / name, folder / name)
        D = data.load(root)
        drawers = original.figures()
        drawers.update(zip(CHANGED, (fig16, fig17)))
        with plt.rc_context({**style.STYLE, 'svg.fonttype': 'path'}):
            for name, function in drawers.items():
                fig, table = function(D)
                try:
                    fig.savefig(folder / (name + '.svg'), metadata={'Date': None})
                    if name in CHANGED:
                        fig.savefig(folder / (name + '.png'), dpi=300, metadata={'Software': None})
                        fig.savefig(folder / (name + '.pdf'), metadata={
                            'CreationDate': None, 'ModDate': None, 'Producer': None, 'Creator': None})
                finally:
                    plt.close(fig)
                if name in CHANGED:
                    table.to_csv(folder / (name + '_plotted.csv'), index=False, lineterminator='\n')
        if verify_history(root) != pins:
            raise ValueError('Historical inputs changed during build')
        manifest = dict(source, status=STATUS, release=OUT.name, source_manifest_sha256=SOURCE_SHA,
                        captions={name: caption(name) for name in CAPTIONS},
                        artifacts={p.name: original.sha(p) for p in sorted(folder.iterdir())},
                        code_sha256={'analysis_figures/reviewed_v6.py': original.sha(Path(__file__))},
                        historical_sha256=pins, svg_fonts='outlined')
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf8')
        folder.rename(destination)
    return manifest


def verify(root, inputs=True):
    root = Path(root).resolve()
    if original.sha(root / original.OUT / 'manifest.json') != SOURCE_SHA:
        raise ValueError('Original v6 manifest differs')
    manifest = json.loads((root / OUT / 'manifest.json').read_text())
    source = original.verify(root, inputs=inputs)
    expected = {n + e for n in CAPTIONS for e in ('.png', '.svg', '.pdf', '_plotted.csv')}
    if (manifest['status'] != STATUS or manifest['source_manifest_sha256'] != SOURCE_SHA or
            manifest.get('svg_fonts') != 'outlined' or
            manifest['captions'] != {name: caption(name) for name in CAPTIONS} or
            set(manifest['artifacts']) != expected or
            {p.name for p in (root / OUT).iterdir()} != expected | {'manifest.json'}):
        raise ValueError('Unexpected reviewed v6.1 release')
    for key in (*data.COUNTS, 'paid_calls', 'ledger_queries', 'new_generation', 'new_hypothesis_tests', 'refitted_baselines'):
        if manifest[key] != source[key]:
            raise ValueError('Release scope differs: ' + key)
    if set(manifest['code_sha256']) != {'analysis_figures/reviewed_v6.py'}:
        raise ValueError('Review code pin inventory differs')
    for name, digest in manifest['code_sha256'].items():
        if original.sha(root / name) != digest:
            raise ValueError('Review code differs: ' + name)
    for name, digest in manifest['artifacts'].items():
        if original.sha(root / OUT / name) != digest:
            raise ValueError('Reviewed artifact differs: ' + name)
        if name.endswith('.svg') and '<text' in (root / OUT / name).read_text(encoding='utf8'):
            raise ValueError('SVG depends on a reader-installed font: ' + name)
        if not name.startswith(CHANGED) and not name.endswith('.svg') and digest != source['artifacts'][name]:
            raise ValueError('An uncorrected v6 figure changed: ' + name)
    if inputs and verify_history(root) != manifest['historical_sha256']:
        raise ValueError('Historical pin inventory differs')
    return manifest


def show(root, names):
    from IPython.display import Image, Markdown, display
    verify(root, inputs=False)
    for name in names:
        if name not in CAPTIONS:
            raise ValueError('Unknown figure: ' + name)
        display(Image(filename=str(Path(root) / OUT / (name + '.png'))))
        display(Markdown(caption(name)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory under outputs/publication_figures.')
    root = Path(__file__).resolve().parents[1]
    build(root, parser.parse_args().output)
    print('REVIEWED_RELEASE_BUILT figures=19 corrected_figures=2 supplements=0 paid_calls=0')
