"""Verify reviewed v6.1 and the six notebooks displaying it. Offline; no provider calls.

Historical seals (primary, held-out, manuscript, v2-v5 figure releases, extension v2) are
re-verified byte for byte through the unchanged historical validators. Only the check that
notebooks embed the OLD figure bytes is superseded here by the v6 embedding check.
"""
import argparse
import base64
import hashlib
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures import reviewed_v6 as release

EXPECTED = {
    '01': ['01_primary_contrasts', '02_primary_robustness', '07_legacy_comparison'],
    '02': ['01_primary_contrasts', '02_primary_robustness', '03_model_profiles', '04_coverage_missingness',
           '05_resource_use', '06_gnm_context', '07_legacy_comparison'],
    '03': ['08_heldout_distances', '09_heldout_distributions', '10_sampling_effect', '11_conditional_controls',
           '12_cohort_sensitivity', '13_appendix_lcc', '15_efficiency_density'],
    '04': sorted(release.CAPTIONS),
    '05': ['14_demographic_mixing'],
    '06': ['16_rq2_paired_mixing', '17_rq2_mixing_matrices', '18_sampling_distortion', '19_copenhagen_transfer'],
}
BANNER = ('**Reviewed figure release v6.1 (2026-11-15).** All 19 figures are in '
          '`outputs/publication_figures/20261115_v6_1/`. This corrects Figure 16 sign-agreement counts and '
          'captions, Figure 17 vector cells and portable SVG fonts, without changing primary estimates, weights or tests. The original v6 release '
          'and all earlier sealed evidence remain unchanged. See `docs/figure_redesign/CODEX_REVIEW.md`.')


def embedded(nb, manifest, figures):
    nbformat.validate(nb)
    outputs = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs]
    if any(o.output_type == 'error' for o in outputs) or any(c.get('attachments') for c in nb.cells):
        raise ValueError('Notebook has an error output or an image attachment')
    images = [hashlib.sha256(base64.b64decode(o.data['image/png'])).hexdigest()
              for o in outputs if 'image/png' in o.get('data', {})]
    captions = [o.data['text/markdown'] for o in outputs if 'text/markdown' in o.get('data', {})]
    wanted = [release.caption(k) for k in figures]
    if images != [manifest['artifacts'][k + '.png'] for k in figures] or \
            [c for c in captions if c in set(wanted)] != wanted:
        raise ValueError('Embedded figures or captions are not exactly reviewed v6.1: ' + str(figures))


def check(write=False):
    manifest = release.verify(ROOT)
    for number, figures in EXPECTED.items():
        path = next((ROOT / 'analysis').glob(number + '_*.ipynb'))
        nb = nbformat.read(path, as_version=4)
        if write:
            banners = [c for c in nb.cells if c.get('id') == 'figure-release-v6']
            if banners:
                banners[0].source = BANNER
            else:
                cell = nbformat.v4.new_markdown_cell(BANNER)
                cell['id'] = 'figure-release-v6'
                nb.cells.insert(1, cell)
        else:
            embedded(nb, manifest, figures)
        NotebookClient(nb, timeout=600, kernel_name='ua-capstone-final060d',
                       resources={'metadata': {'path': str(ROOT / 'analysis')}}, allow_errors=False).execute()
        embedded(nb, manifest, figures)
        if write:
            with path.open('w', encoding='utf8', newline='\n') as stream:
                nbformat.write(nb, stream)
            embedded(nbformat.read(path, as_version=4), manifest, figures)
        print(f'{path.name}: executed; {len(figures)} v6.1 figures and captions verified', flush=True)
    release.verify(ROOT)
    print('RELEASE_V6_1_VERIFIED figures=19 supplements=0 paid_calls=0 historical_seals=unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Refresh notebook outputs after execution.')
    check(parser.parse_args().write)
