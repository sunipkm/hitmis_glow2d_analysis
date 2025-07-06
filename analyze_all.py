# %%
from __future__ import annotations
from datetime import datetime, timedelta
import lzma
import os
import pickle
from typing import List, SupportsFloat as Numeric

from matplotlib import pyplot as plt, ticker
import matplotlib
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import pytz
import scipy
import tqdm
import xarray
from common_funcs import LINESTYLE_DICT, fill_array_1d, get_date
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
    stats = {}
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
                        scales.append(
                            (pert.x[0], pert.x[1], pert.x[2], pert.x[3], pert.x[4], pert.x[5]))
                    else:
                        scales.append(
                            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                scales = np.array(scales)
                ds = xarray.DataArray(scales, dims=['tstamp', 'species'], coords={
                                      'tstamp': tstamps, 'species': ['O', 'O2', 'N2', 'NO', 'N4S', 'e-']})
                dss.append(ds)
        dss = xarray.concat(dss, dim='suffix')
        ds = xarray.Dataset({'density': dss})
        ds['minval'] = ds.density.min(dim='suffix')
        ds['maxval'] = ds.density.max(dim='suffix')
        ds['meanval'] = ds.density.mean(dim='suffix')
        ds['stdval'] = ds.density.std(dim='suffix')
        ds['geomean'] = ds.density.std(dim='suffix')
        ds['geomean'].values = scipy.stats.mstats.gmean(ds.density, axis=0)
        stats[date] = ds
    return stats


compiled_stats = compile_density_stats(suffixes)

# %%

def plot_density_stat(stats):
    num_rows = len(stats) // 2
    gspec = GridSpec(num_rows + 1, 2, hspace=0, wspace=0.05,
                     height_ratios=[0.1] + [1] * num_rows)

    fig = plt.figure(figsize=(4.8, 2*num_rows), dpi=300)
    lax = fig.add_subplot(gspec[0, :])
    lax.set_axis_off()
    axes = []
    for i in range(num_rows):
        axes.append([])
        for j in range(2):
            if i == 0:
                ax = fig.add_subplot(gspec[i + 1, j])
            else:
                ax = fig.add_subplot(
                    gspec[i + 1, j], sharex=axes[0][j], sharey=axes[0][j])
            axes[i].append(ax)
    axes = np.asarray(axes, dtype=Axes)

    # fig, axes = plt.subplots(num_rows, 2, figsize=(
    #     4.8, 2*num_rows), sharex=True, sharey=True, dpi=300)
    # fig.subplots_adjust(hspace=0, wspace=0.05)
    lprops = {
        'O': {'color': 'blue', 'linestyle': LINESTYLE_DICT['dotted'], 'label': 'O', 'lw': 0.75},
        'O2': {'color': 'red', 'linestyle': LINESTYLE_DICT['loosely dashed'], 'label': 'O$_2$', 'lw': 0.75},
        'N2': {'color': 'green', 'linestyle': LINESTYLE_DICT['dashdot'], 'label': 'N$_2$', 'lw': 0.75},
        'NO': {'color': 'purple', 'linestyle': LINESTYLE_DICT['densely dashdotted'], 'label': 'NO', 'lw': 0.75},
        'N4S': {'color': 'orange', 'linestyle': LINESTYLE_DICT['dashdotdotted'], 'label': 'N$(^4S)$', 'lw': 0.75},
        'e-': {'color': 'black', 'linestyle': '-', 'label': 'e$^-$', 'lw': 0.75},
    }
    species = ['O', 'O2', 'N2', 'NO', 'N4S', 'e-']  # 'N2', 'NO', 'N4S',

    ax_xlim = []
    ax_ylim = []
    datagaps: dict[int, tuple[Numeric]] = {}

    matplotlib.rcParams.update({'font.size': 10})
    matplotlib.rcParams.update({'axes.titlesize': 10})
    matplotlib.rcParams.update({'axes.labelsize': 10})

    dates = list(stats.keys())
    dates.sort()
    for idx, (date, ax) in enumerate(zip(dates, axes.flatten())):
        ax: Axes = ax
        ds = stats[date]
        ttstamps = ds.density.tstamp.values.copy()
        tstamps = [pd.to_datetime(t).to_pydatetime().astimezone(
            pytz.timezone('US/Eastern')) for t in ttstamps]
        start = tstamps[0]
        start = datetime(start.year, start.month,
                         start.day, start.hour)
        legends = []
        ltexts = []
        baseval = ds.loc[dict(suffix=0)]
        for sp in species:
            bss = baseval.loc[dict(species=sp)]
            dss = ds.loc[dict(species=sp)]
            ttstamps = dss.density.tstamp.values.copy()
            tstamps = [pd.to_datetime(t).to_pydatetime().astimezone(
                pytz.timezone('US/Eastern')) for t in ttstamps]
            base = bss['density'].values
            meanval = dss['meanval'].values
            stdval = dss['stdval'].values
            minval = dss['minval'].values
            maxval = dss['maxval'].values
            geomean = dss['geomean'].values
            _, base, _ = fill_array_1d(base, tstamps)
            _, meanval, _ = fill_array_1d(meanval, tstamps)
            _, stdval, _ = fill_array_1d(stdval, tstamps)
            _, minval, _ = fill_array_1d(minval, tstamps)
            _, maxval, _ = fill_array_1d(maxval, tstamps)
            tstamps, geomean, nanfill = fill_array_1d(
                geomean, tstamps)  # type: ignore
            ttstamps = np.asarray([t.timestamp()
                                  for t in tstamps], dtype=float)
            ttstamps -= start.timestamp()
            ttstamps /= 3600  # convert to hours
            assert len(ttstamps) == len(
                meanval), f'{date} {sp} length mismatch: {len(ttstamps)} != {len(meanval)}'
            # line, = ax.plot(ttstamps, base, **lprops[sp])
            line, = ax.plot(ttstamps, meanval, **lprops[sp])
            fill1 = ax.fill_between(ttstamps, meanval - stdval, meanval + stdval,
                                    alpha=0.2, color=lprops[sp]['color'], edgecolor=None)
            # fill2 = ax.fill_between(ttstamps, minval, maxval,
            #                         alpha=0.2, color=lprops[sp]['color'])
            ax_xlim.append((ttstamps[0], ttstamps[-1]))
            ax_ylim.append((np.nanmin(minval), np.nanmax(maxval)))
            ax_ylim.append((np.nanmin(meanval - stdval),
                           np.nanmax(meanval + stdval)))
            legends.append((line, fill1))
            ltexts.append(fr'[{lprops[sp]["label"]}]$\pm 1\sigma$')
        if not idx % 2 == 0:
            ax.yaxis.set_ticks_position('none')
        ylim = ax.get_ylim()
        if nanfill is not None:  # type: ignore
            nanfill: np.ndarray = nanfill  # type: ignore
            tmin = nanfill[0]
            tmax = nanfill[-1]
            trange = np.asarray(ttstamps)[tmin:tmax + 1]
            ax.fill_between(trange, -10, 10, color='k',
                            alpha=0.2, edgecolor=None, hatch='//')
            datagaps[idx] = (ttstamps[tmin:tmax + 1].mean(),)
        ax.text(0.5, 0.99, f'{start:%Y-%m-%d}',
                ha='center', va='top', transform=ax.transAxes)

    ax_xlim = np.asarray(ax_xlim, dtype=float)
    ax_ylim = np.asarray(ax_ylim, dtype=float)
    formatter = ticker.ScalarFormatter()
    formatter.set_scientific(False)
    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlim(ax_xlim[:, 0].min(), ax_xlim[:, 1].max())
        ax.set_ylim(0, ax_ylim[:, 1].max())
        # ax.set_ylim(-0.05, 1.05)
        if idx % 2 == 0:
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.yaxis.set_ticks([])

    for k, v in datagaps.items():
        ax = axes.flatten()[k]
        ylim = ax.get_ylim()
        ax.text(v[0], np.asarray(ylim).mean(), 'Data Unavailable', ha='center',  # type: ignore
                va='center', fontsize=8, color='r', rotation='vertical')

    def fmt_time(x: Numeric, ofst: datetime) -> str:
        x = timedelta(hours=x)  # type: ignore
        res = ofst + x  # type: ignore
        return res.strftime('%H:%M')

    fig.text(0.055, 0.5, 'Density Perturbation',
             va='center', rotation='vertical')

    for ax in axes.flatten()[-2:]:
        xticks = np.asarray(ax.get_xticks())
        xticks = np.round(xticks, decimals=1)
        # type: ignore
        xticks = list(map(lambda x: fmt_time(x, start), xticks)) # type: ignore
        ax.set_xticklabels(xticks, rotation=45)
        ax.set_xlabel("Local Time (UTC$-$05:00)")

    # for (line, text) in zip(legends, ltexts):
    #     print(line, text)
    lax.legend(legends, ltexts, loc='center', fontsize=6, frameon=False, ncol=len(ltexts), mode='expand') # type: ignore
    # draw_vertical_legend(lax, items=legends, texts=ltexts, fontsize=6)
    fig.savefig(ROOT_DIR / 'density_stats_multirun.png',
                dpi=600, bbox_inches='tight')
    fig.show()


# %%
plot_density_stat(compiled_stats)
# %%
