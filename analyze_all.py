# %%
from __future__ import annotations
import lzma
import os
import pickle
from typing import List

import numpy as np
import scipy
import tqdm
import xarray
from common_funcs import get_date
from settings import ROOT_DIR, Directories
# %%


def init(run=False) -> List[str]:
    """Populate the directories with the required results.

    Returns:
        List[str]: A list of valid suffixes for the directories.
    """
    valid_suffixes = []
    for idx in range(16):
        suffix = f'randinit_run{idx}'
        dirname = ROOT_DIR / f'keomodel_{suffix}'
        if dirname.exists():
            if run:
                os.system(f'python fit_den.py {suffix}')
                os.system(f'python fit_loc.py {suffix}')
                os.system(f'python fit_tec.py {suffix}')
            valid_suffixes.append(suffix)
    return valid_suffixes


# %%
suffixes = init()
# %%


def compile_tec_corr(suffixes: List[str]) -> None:
    digi_base = dict()
    gps_base = dict()
    digi_gmean = dict()
    gps_gmean = dict()
    digi_corrs = dict()
    gps_corrs = dict()

    fname = ROOT_DIR / 'fitprops' / 'tec_correlation.csv'
    if fname.exists():
        with open(fname, 'r') as f:
            lines = f.readlines()[1:]  # skip header
            lines = [line.strip().split(',') for line in lines]
            for line in lines:
                key, digi, gps = line[0], float(line[1]), float(line[2])
                digi_base[key] = digi
                gps_base[key] = gps

    keys = digi_base.keys()
    for key in keys:
        digi_corrs[key] = []
        gps_corrs[key] = []

    for suffix in suffixes:
        fname = ROOT_DIR / f'fitprops_{suffix}' / 'tec_correlation.csv'
        if not fname.exists():
            print(f'File {fname} does not exist. Skipping.')
            continue
        with open(fname, 'r') as f:
            lines = f.readlines()[1:]  # skip header
            lines = [line.strip().split(',') for line in lines]
            for line in lines:
                if line[0] not in keys:
                    print(f'Key {line[0]} not in base keys. Skipping.')
                digi_corrs[line[0]].append(float(line[1]))
                gps_corrs[line[0]].append(float(line[2]))
    if keys is not None:
        for key in keys:
            digi_gmean[key] = scipy.stats.mstats.gmean(digi_corrs[key])
            gps_gmean[key] = scipy.stats.mstats.gmean(gps_corrs[key])

    header = ['Date', 'Baseline'] + \
        [f'Run {i}' for i in range(1, len(suffixes) + 1)] + ['Geomean']
    with open('tec_correlation_digi.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for key in keys:
            line = [key, digi_base[key]] + [digi_corrs[key][i]
                                            for i in range(len(suffixes))] + [f'{digi_gmean[key]:.2f}']
            f.write(','.join(map(str, line)) + '\n')
    with open('tec_correlation_gps.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for key in keys:
            line = [key, gps_base[key]] + [gps_corrs[key][i]
                                           for i in range(len(suffixes))] + [f'{gps_gmean[key]:.2f}']
            f.write(','.join(map(str, line)) + '\n')


# %%
compile_tec_corr(suffixes)
# %%
def compile_density_stats(suffixes: List[str]):
    msuffixes = [None] + suffixes  # Add None for the base case
    dates = list(map(get_date, (ROOT_DIR / 'keomodel').glob('fitres*.xz')))
    dates.sort()
    stats = []
    for date in tqdm.tqdm(dates, dynamic_ncols=True):
        dss = []
        for suffix in msuffixes:
            dirs = Directories(suffix)
            with lzma.open(dirs.model_dir / f'fitres_{date}.xz', 'rb') as f:
                fitres = pickle.load(f)
                tstamps = []
                scales = []
                for vals in fitres:
                    tstamp, pert = vals
                    tstamps.append(tstamp)
                    if pert is not None:
                        scales.append((pert.x[0], pert.x[1], pert.x[2], pert.x[3], pert.x[4], pert.x[5]))
                    else:
                        scales.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                scales = np.array(scales)
                ds = xarray.DataArray(scales, dims=['tstamp', 'species'], coords={'tstamp': tstamps, 'species': ['O', 'O2', 'N2', 'NO', 'N4S', 'e-']})
                dss.append(ds)
        dss = xarray.concat(dss, dim='suffix')
        ds = xarray.Dataset({'density': dss})
        ds['minval'] = ds.density.min(dim='suffix')
        ds['maxval'] = ds.density.max(dim='suffix')
        ds['meanval'] = ds.density.mean(dim='suffix')
        ds['stdval'] = ds.density.std(dim='suffix')
        ds['geomean'] = ds.density.std(dim='suffix')
        ds['geomean'].values = scipy.stats.mstats.gmean(ds.density, axis=0)
        stats.append(ds)
    return stats

compiled_stats = compile_density_stats(suffixes)

# %%
