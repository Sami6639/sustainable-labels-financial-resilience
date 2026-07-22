"""One-command reproduction of the manuscript's core tables and figures.
Run from the repository root with: python code/run_all.py
"""
from pathlib import Path
import runpy, shutil
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'reproduced'
backup = ROOT / 'reproduced_previous'
if backup.exists(): shutil.rmtree(backup)
if REP.exists(): REP.rename(backup)
(REP/'results').mkdir(parents=True); (REP/'tables').mkdir(); (REP/'figures').mkdir()
steps=[
 ROOT/'code'/'validate_package.py',
 ROOT/'code'/'reproduction'/'reproduce_persistence.py',
 ROOT/'code'/'reproduction'/'reproduce_persistent_models.py',
 ROOT/'code'/'reproduction'/'build_publication_outputs.py',
 ROOT/'code'/'reproduction'/'validate_publication_outputs.py']
for i,s in enumerate(steps,1):
    print(f'\n[{i}/{len(steps)}] Running {s.name}...')
    runpy.run_path(str(s),run_name='__main__')
print('\nAll 8 manuscript tables (Tables 1–6 and A1–A2) and all 3 manuscript figures reproduced successfully in reproduced/.')
