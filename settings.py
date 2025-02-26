from __future__ import annotations
import os
import shutil
import sys
from typing import Optional

SUFFIX: Optional[str] = None # suffix for the output files
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

if __name__ == '__main__':
    while True:
        inp = input('Delete all files in directories? (y/n): ')
        inp = ''.join(inp.split()).lower()
        if inp in ['y', 'yes']:
            break
        elif inp in ['n', 'no']:
            exit()
        else:
            print('Invalid input. Try again.')
    print('Deleting files...', end=' ')
    sys.stdout.flush()
    print(f'{FITPROPS_DIR}...', end=' ')
    sys.stdout.flush()
    shutil.rmtree(FITPROPS_DIR, ignore_errors=True)
    print(f'{VERTPROPS_DIR}...', end=' ')
    sys.stdout.flush()
    shutil.rmtree(VERTPROPS_DIR, ignore_errors=True)
    print(f'{KEOGRAMS_DIR}...', end=' ')
    sys.stdout.flush()
    shutil.rmtree(KEOGRAMS_DIR, ignore_errors=True)
    print('Done.')