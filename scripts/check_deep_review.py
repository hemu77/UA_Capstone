"""Compatibility entry point for current notebooks; historical v2-v6 seals remain checked."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_figure_release_v6 import check

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Refresh current outputs in all six notebooks.')
    args = parser.parse_args()
    print('Current entry point: scripts/check_figure_release_v6.py; verifying reviewed v6.1 and historical seals.')
    check(args.write)
