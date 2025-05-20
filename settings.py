from __future__ import annotations
import os
import shutil
import sys
from typing import List, Optional

SUFFIX: Optional[str] = 'randinit_run0' # suffix for the output files
FIT_SHOW_FIGS: bool = True # show fit figures
FIT_SAVE_FIGS: bool = False # save fit figures

COUNTS_DIR = 'keocounts'
MODEL_DIR = 'keomodel'
FITPROPS_DIR = 'fitprops'
VERTPROPS_DIR = 'fitpropsvert'
KEOGRAMS_DIR = 'keograms'

# strip and remove whitespaces
if SUFFIX is not None:
    SUFFIX = SUFFIX.strip()
    SUFFIX = ''.join(SUFFIX.split())

if SUFFIX is not None and len(SUFFIX) > 0:
    MODEL_DIR += f'_{SUFFIX}'
    FITPROPS_DIR += f'_{SUFFIX}'
    VERTPROPS_DIR += f'_{SUFFIX}'
    KEOGRAMS_DIR += f'_{SUFFIX}'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FITPROPS_DIR, exist_ok=True)
os.makedirs(VERTPROPS_DIR, exist_ok=True)
os.makedirs(KEOGRAMS_DIR, exist_ok=True)

def delete_directories(dirs: List[str] | str)->None:
    if isinstance(dirs, str):
        dirs = [dirs]
    print('Deleting directories:')
    for d in dirs:
        if not os.path.exists(d):
            continue
        print(f'  {d}')
    while True:
        inp = input(f'Delete directories? (y/n): ')
        inp = ''.join(inp.split()).lower()
        if inp in ['y', 'yes']:
            break
        elif inp in ['n', 'no']:
            return
        else:
            print('Invalid input. Try again.')
    for d in dirs:
        print(f'Deleting directory {d}...', end=' ')
        sys.stdout.flush()
        shutil.rmtree(d, ignore_errors=True)
        print('Done.')
        sys.stdout.flush()

if __name__ == '__main__':
    delete_directories([FITPROPS_DIR, VERTPROPS_DIR, KEOGRAMS_DIR])
    delete_directories(MODEL_DIR)
