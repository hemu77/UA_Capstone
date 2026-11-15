"""Render figure release v6 into a NEW directory. Offline; reads sealed tables only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures.v6 import release

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / release.OUT)
    manifest = release.build(ROOT, parser.parse_args().output)
    print(f"FIGURE_RELEASE_V6={len(manifest['artifacts'])} artifacts; supplements=0; paid_calls=0")
