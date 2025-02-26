# %% Imports
from __future__ import annotations
from collections.abc import Iterable
import datetime as dt
from functools import partial
import gc
from io import TextIOWrapper
import lzma
import multiprocessing
import pickle
from typing import Any, Dict, List, Sequence, SupportsFloat as Numeric, Tuple
from tzlocal import get_localzone
import uncertainties
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from scipy.optimize import curve_fit
from pysolar import solar
import pytz
from matplotlib.pyplot import cm
from glowpython import no_precipitation

import geomagdata as gi
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import least_squares, OptimizeResult
from skmpython import GenericFit, staticvars

import glow2d
from glowpython import generic
from tqdm import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc, ticker
import matplotlib
import pandas as pd
from dateutil.parser import parse

usetex = False
if not usetex:
    # computer modern math text
    matplotlib.rcParams.update({'mathtext.fontset': 'cm'})

rc('font', **{'family': 'serif',
   'serif': ['Times' if usetex else 'Times New Roman']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=usetex)

from settings import MODEL_DIR, VERTPROPS_DIR, FIT_SAVE_FIGS
print(f'Loaded settings: {MODEL_DIR}, {VERTPROPS_DIR}')
# %% Functions

def do_interp_smoothing(x: np.ndarray, xp: np.ndarray, yp: np.ndarray, sigma: int | float = 22.5, round: int = None):
    y = interp1d(xp, yp, kind='nearest-up', fill_value='extrapolate')(x)
    y = gaussian_filter1d(y, sigma=sigma)
    if round is not None:
        y = np.round(y, decimals=round)
    return y


def get_smoothed_geomag(tstamps: np.ndarray, tzaware: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tdtime = list(map(lambda t: pd.to_datetime(
        t).to_pydatetime().astimezone(pytz.utc), tstamps))
    tdtime_in = [tdtime[0] - dt.timedelta(hours=6), tdtime[0] - dt.timedelta(hours=3)] + tdtime + [
        tdtime[-1] + dt.timedelta(hours=3), tdtime[-1] + dt.timedelta(hours=6)]
    ttidx = np.asarray(list(map(lambda t: t.timestamp(), tdtime)))
    pdtime = []
    f107a = []
    f107 = []
    f107p = []
    ap = []
    for td in tdtime_in:
        ip = gi.get_indices(
            [td - dt.timedelta(days=1), td], 81, tzaware=tzaware)
        f107a.append(ip["f107s"][1])
        f107.append(ip['f107'][1])
        f107p.append(ip['f107'][0])
        ap.append(ip["Ap"][1])
        pdtime.append(pd.to_datetime(
            ip.index[1].value).to_pydatetime().timestamp())
    pdtime = np.asarray(pdtime)
    ap = np.asarray(ap)
    f107a = np.asarray(f107a)
    f107 = np.asarray(f107)
    f107p = np.asarray(f107p)

    ap = do_interp_smoothing(ttidx, pdtime, ap, round=0)  # rounds to integer
    f107 = do_interp_smoothing(ttidx, pdtime, f107)  # does not round
    f107a = do_interp_smoothing(ttidx, pdtime, f107a)  # does not round
    f107p = do_interp_smoothing(ttidx, pdtime, f107p)  # does not round

    return tdtime, ap, f107, f107a, f107p
# %% Functions


def fmt_time(x: Numeric, ofst: dt.datetime) -> str:
    x = dt.timedelta(hours=x)
    res = ofst + x
    return res.strftime('%H:%M')


def get_date(filename: str) -> str:
    """Get the date from a filename."""
    return filename.rsplit('.')[0].rsplit('_')[-1]

# %%
def fill_array(arr: np.ndarray, tstamps: List[Any], axis: int=0)->Tuple[List[Any], np.ndarray]:
    if arr.ndim < 1:
        raise ValueError('Array must be 1 dim')
    if axis >= arr.ndim or axis < 0:
        raise ValueError('Axis invalid')
    if arr.shape[axis] != len(tstamps):
        raise ValueError('Length of tstamps must be equal to the size of the array')
    ts = np.asarray(tstamps, dtype=float)
    dts = np.diff(ts)
    t_delta = dts.min()
    gaps = dts[np.where(dts > t_delta)[0]]
    gaps = np.asarray(gaps // t_delta, dtype=int)
    dts = np.diff(dts)
    oidx = np.where(dts < 0)[0]
    if len(oidx) == 0:
        return tstamps, arr
    tstamps = []
    tlen = int((ts[-1] - ts[0]) // t_delta) + 1
    for idx in range(tlen):
        tstamps.append(ts[0] + t_delta*idx)
    shape = list(arr.shape)
    shape[axis] = tlen
    out = np.full(shape, dtype=arr.dtype, fill_value=np.nan)
    start = 0
    dstart = 0
    for idx, oi in enumerate(oidx):
        if axis == 0:
            out[start:oi+1] = arr[dstart:oi+1]
        else:
            out[:, start:oi+1] = arr[:, dstart:oi+1]
        start = oi + gaps[idx]
        dstart = oi + 1
        if idx == len(oidx) - 1: # end
            if axis == 0:
                out[start:] = arr[dstart:]
            else:
                out[:, start:] = arr[:, dstart:]
    return (tstamps, out)


# %% Line styles
linestyle_str = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
    ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),    # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_dict = {
    'loosely dotted':      (0, (1, 10)),
    'dotted':              (0, (1, 1)),
    'densely dotted':      (0, (1, 1)),
    'long dash with offset': (5, (10, 3)),
    'loosely dashed':      (0, (5, 10)),
    'dashed':              (0, (5, 5)),
    'densely dashed':      (0, (5, 1)),
    'dashdot':             (0, (3, 5, 1, 5)),
    'loosely dashdotted':  (0, (3, 10, 1, 10)),
    'dashdotted':          (0, (3, 5, 1, 5)),
    'densely dashdotted':  (0, (3, 1, 1, 1)),
    'dashdotdotted':       (0, (3, 5, 1, 5, 1, 5)),
    'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}
# %% For each day


def generate_vert(*param):
    if len(param) == 1:
        param = param[0]
    date, file, ofile, keys, tfile = param
    date: str = date
    file: str = file
    ofile: TextIOWrapper = ofile
    tfile: TextIOWrapper = tfile
    keys: List[str] = keys

    with lzma.open(file, 'rb') as f:
        fitres = pickle.load(f)
    tstamps = [x[0] for x in fitres]
    start = pd.to_datetime(
        tstamps[0]).to_pydatetime()
    end = pd.to_datetime(
        tstamps[-1]).to_pydatetime()
    print(f'Processing {start:%Y-%m-%d}')
    # Get the model data
    # pbar = tqdm(range(len(tstamps)))
    pbar = range(len(tstamps))
    den_part = np.full((len(tstamps), 6), np.nan)
    for idx in pbar:
        res = fitres[idx][1]
        den_part[idx, :] = (res.x[0], res.x[1], res.x[2],
                            res.x[3], res.x[4], res.x[5])
    den_o = np.array(den_part[:, 0])
    den_o2 = np.array(den_part[:, 1])
    den_n2 = np.array(den_part[:, 2])
    den_no = np.array(den_part[:, 3])
    den_n4s = np.array(den_part[:, 4])
    den_e = np.array(den_part[:, 5])
    tstamps = np.asarray(tstamps, dtype=int)
    tstamps = tstamps.astype(float)
    tstamps *= 1e-9  # convert to seconds
    sstart = dt.datetime.fromtimestamp(tstamps[0]) # start
    sstart = dt.datetime(sstart.year, sstart.month, sstart.day, sstart.hour, 0, 0)
    tstamps -= sstart.timestamp()
    sstart = sstart.astimezone(pytz.utc)
    """Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-
    """
    stats = {}
    stats['O'] = (np.nanmean(den_o), np.nanstd(den_o), np.nanmedian(den_o), np.nanmin(den_o), np.nanmax(den_o))
    stats['O2'] = (np.nanmean(den_o2), np.nanstd(den_o2), np.nanmedian(den_o2), np.nanmin(den_o2), np.nanmax(den_o2))
    stats['N2'] = (np.nanmean(den_n2), np.nanstd(den_n2), np.nanmedian(den_n2), np.nanmin(den_n2), np.nanmax(den_n2))
    stats['NO'] = (np.nanmean(den_no), np.nanstd(den_no), np.nanmedian(den_no), np.nanmin(den_no), np.nanmax(den_no))
    stats['N4S'] = (np.nanmean(den_n4s), np.nanstd(den_n4s), np.nanmedian(den_n4s), np.nanmin(den_n4s), np.nanmax(den_n4s))
    stats['e-'] = (np.nanmean(den_e), np.nanstd(den_e), np.nanmedian(den_e), np.nanmin(den_e), np.nanmax(den_e))
    _, den_o = fill_array(den_o, tstamps)
    _, den_o2 = fill_array(den_o2, tstamps)
    _, den_n2 = fill_array(den_n2, tstamps)
    _, den_no = fill_array(den_no, tstamps)
    _, den_n4s = fill_array(den_n4s, tstamps)
    tstamps, den_e = fill_array(den_e, tstamps)
    tstamps = np.asarray(tstamps, dtype=float)
    tstamps /= 3600  # convert to hours
    fig = plt.figure(figsize=(4.8, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(tstamps, den_o, 
            label='O', color='blue',
             linewidth=0.75, linestyle=linestyle_dict['dotted'])
    ax.plot(tstamps, den_o2, label='O$_2$', color='red',
             linewidth=0.75, linestyle=linestyle_dict['loosely dashed'])
    ax.plot(tstamps, den_n2, label='N$_2$', color='green',
             linewidth=0.75, linestyle=linestyle_dict['dashdot'])
    ax.plot(tstamps, den_no, label='NO', color='purple',
             linewidth=0.75, linestyle=linestyle_dict['densely dashdotted'])
    ax.plot(tstamps, den_n4s, label='N$(^4S)$', color='orange', linestyle=linestyle_dict['dashdotdotted'], linewidth=0.75)
    ax.plot(tstamps, den_e, label='e$^-$', color='black',
             linewidth=0.75)
    ax.set_xlabel('Local Time (Hours)')
    ax.set_ylabel('Density Perturbation')
    ax.set_xlim(0, 9)
    xticks = ax.get_xticks()
    xticklabels = list(map(lambda x: fmt_time(x, sstart), xticks))
    ax.set_xticklabels(xticklabels)
    ax.legend()
    ax.set_ylim(0.25, 3)
    ax.set_title(f'{sstart:%Y-%m-%d} {start:%H:%M} - {end:%H:%M} (UTC-05:00)')
    if FIT_SAVE_FIGS:
        fig.savefig(f'{VERTPROPS_DIR}/fit_den_{date}.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    ofile.write('\n')
    ofile.write(f'{sstart:%Y-%m-%d},\t')
    tfile.write(f'{sstart:%Y-%m-%d} ')
    for key in keys:
        vals = stats[key]
        ofile.write(f'{vals[0]:.3f}+/-{vals[1]:.3f},\t{vals[3]:.3f},\t{vals[4]:.3f},\t')
        tfile.write(f'& ${vals[0]:.2f}^{{{vals[4]:.2f}}}_{{{vals[3]:.2f}}}$ ')
    tfile.write(r'\\' + '\n')
    return tstamps, den_part


# %% Main
if __name__ == '__main__':
    files = glob.glob(f'{MODEL_DIR}/fitres*.xz')
    files.sort(key=get_date)
    keys = ['O', 'O2', 'N2', 'NO', 'N4S', 'e-']
    dates = list(map(get_date, files))
    with open(f'{VERTPROPS_DIR}/fit_den_stats.csv', 'w') as ofile, open(f'{VERTPROPS_DIR}/fit_den_tex.tex', 'w') as tfile:
        ofile.write('Date,\t')
        for key in keys:
            ofile.write(f'{key} Mean,\t{key} Min,\t{key} Max,\t')
        tfile.write(
    r"""
\begin{tabular}{r c c c c c c}
    \hline
    Date & O & O$_2$ & N$_2$ & NO & N($^4S$) & e$^-$ \\
    \hline
    """
        )
        for date, file in zip(dates, files):
            tstamp, den_part = generate_vert(date, file, ofile, keys, tfile)
        tfile.write(r"""
    \hline
\end{tabular}
        """)

# %%
