"""Execute the maintained notebook workflow without changing the source notebook."""
import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
notebook = nbformat.read(ROOT / 'analyze_networks.ipynb', as_version=4)
nbformat.validate(notebook)
NotebookClient(notebook, timeout=120, kernel_name='python3',
               resources={'metadata': {'path': str(ROOT)}}).execute()
destination = ROOT / 'outputs/qa/analyze_networks.executed.ipynb'
destination.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, destination)
code = [cell for cell in notebook.cells if cell.cell_type == 'code']
archival = [cell for cell in code if cell.source.startswith('%%legacy_archive')]
print(f'Notebook: {len(code)} code cells executed, {len(archival)} explicitly skipped archival bodies; no API calls.')
print(f'Executed copy: {destination}')
