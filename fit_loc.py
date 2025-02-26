# %% Imports
from __future__ import annotations
from collections.abc import Iterable
import datetime as dt
from functools import partial
import gc
import lzma
import multiprocessing
import pickle
from typing import Dict, List, Sequence, SupportsFloat as Numeric, Tuple
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

from settings import MODEL_DIR, VERTPROPS_DIR, FITPROPS_DIR
# %% Interpolate + Smoothing


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
        f107a.append(ip["f107s"].iloc[1])
        f107.append(ip['f107'].iloc[1])
        f107p.append(ip['f107'].iloc[0])
        ap.append(ip["Ap"].iloc[1])
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
# %% For each day


def generate_vert(*param):
    if len(param) == 1:
        param = param[0]
    date, file = param
    print('Processing', date)
    lat, lon = 42.64981361744372, -71.31681056737486
    if os.path.exists(f'{MODEL_DIR}/vert_{date}.nc'):
        ionos = xr.load_dataset(f'{MODEL_DIR}/vert_{date}.nc')
        return
    ionos = []
    with lzma.open(file, 'rb') as f:
        fitres = pickle.load(f)
    # Get the model data
    tstamps = [x[0] for x in fitres]
    _, ap, f107, f107a, f107p = get_smoothed_geomag(tstamps)
    # pbar = tqdm(range(len(tstamps)))
    pbar = range(len(tstamps))
    ionos = []
    for idx in pbar:
        geomag_params = (f107a[idx], f107[idx], f107p[idx], ap[idx])
        res = fitres[idx][1]
        if res is None:
            print('None')
        else:
            time = pd.to_datetime(
                tstamps[idx]).to_pydatetime().astimezone(pytz.utc)
            density_pert = (res.x[0], res.x[1], res.x[2],
                            res.x[3], res.x[4], 1, res.x[5])
            iono = no_precipitation(
                time, lat, lon, 100, density_pert, geomag_params=geomag_params)
            # geomag_params = iono.attrs['geomag_params']
            # if 'geomag_params' in iono.attrs:
            #     del iono.attrs['geomag_params']
            if 'precip' in iono.attrs:
                del iono.attrs['precip']
            # else:
            #     print('No precip')
            # for key, val in geomag_params.items():
            #     iono.attrs[key] = val
            iono.attrs['density_perturbation'] = density_pert
            ionos.append(iono)
    ionos = xr.concat(ionos, pd.Index(tstamps, name='tstamp'))
    ionos.to_netcdf(f'{MODEL_DIR}/vert_{date}.nc')
    return ionos
# %%


def fill_array(arr: np.ndarray, tstamps: List[dt.datetime], axis: int = 1) -> Tuple[List[dt.datetime], np.ndarray, bool]:
    if arr.ndim != 2:
        raise ValueError('Array must be 2 dim')
    if axis >= arr.ndim or axis < 0:
        raise ValueError('Axis invalid')
    ts = np.asarray(list(map(lambda t: t.timestamp(), tstamps)), dtype=float)
    dts = np.diff(ts)
    t_delta = dts.min()
    gaps = dts[np.where(dts > t_delta)[0]]
    gaps = np.asarray(gaps // t_delta, dtype=int)
    dts = np.diff(dts)
    oidx = np.where(dts < 0)[0]
    if len(oidx) == 0:
        return tstamps, arr, False
    tstamps = []
    tlen = int((ts[-1] - ts[0]) // t_delta) + 1
    for idx in range(tlen):
        tstamps.append(dt.datetime.fromtimestamp(
            ts[0] + t_delta*idx).astimezone(pytz.utc))
    if axis == 0:
        out = np.full((tlen, arr.shape[1]), dtype=arr.dtype, fill_value=np.nan)
    elif axis == 1:
        out = np.full((arr.shape[0], tlen), dtype=arr.dtype, fill_value=np.nan)
    else:
        raise RuntimeError('Should not reach')
    start = 0
    dstart = 0
    for idx, oi in enumerate(oidx):
        if axis == 0:
            out[start:oi+1] = arr[dstart:oi+1]
        else:
            out[:, start:oi+1] = arr[:, dstart:oi+1]
        start = oi + gaps[idx]
        dstart = oi + 1
        if idx == len(oidx) - 1:  # end
            if axis == 0:
                out[start:] = arr[dstart:]
            else:
                out[:, start:] = arr[:, dstart:]
    return (tstamps, out, True)
# %%


def make_color_axis(ax: plt.Axes | Iterable, position: str = 'right', size: str = '1.5%', pad: float = 0.05) -> plt.Axes | list:
    if isinstance(ax, Iterable):
        mmake_color_axis = partial(
            make_color_axis, position=position, size=size, pad=pad)
        cax = list(map(mmake_color_axis, ax))
        return cax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    return cax


# %%
def plot_density(iono: xr.Dataset, keys: Sequence[str], exkeys: Sequence[str], dir_prefix: str, file_prefix: str, *, vmin=None, vmax=None, log=False, cmap='bone'):
    dtime = parse(iono.time).astimezone(
        get_localzone()) - dt.timedelta(hours=8)
    day = dtime.strftime('%Y-%m-%d')
    tstamps = iono.tstamp.values
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    vals = {}
    for key in keys[:-1]:
        arr = iono[key].values
        _, arr, _ = fill_array(arr, tstamps, axis=0)
        vals[key] = arr
    tstamps, vals[keys[-1]], nanfill = fill_array(
        iono[keys[-1]].values, tstamps, axis=0
    )
    fig, axs = plt.subplots(len(keys), 1, figsize=(
        6, 4.8), sharex=True, tight_layout=True)
    fig.suptitle('Vertical Profile\n%s - %s (UTC-5:00)' %
                 (tstamps[0].strftime('%Y-%m-%d %H:%M'), tstamps[-1].strftime('%Y-%m-%d %H:%M')))
    cax = make_color_axis(axs)
    fig.set_dpi(300)
    matplotlib.rcParams.update({'font.size': 10})
    matplotlib.rcParams.update({'axes.titlesize': 10})
    matplotlib.rcParams.update({'axes.labelsize': 10})

    def fmt(x, pos):
        a, b = '{:.1e}'.format(x).split('e')
        b = int(b)
        return r'$10^{{{}}}$'.format(a)

    def fmt2(x, pos):
        x = int(x + 18)
        return r'${}^\circ$'.format(x)
    for ax in axs:
        # ax.yaxis.set_major_formatter(fmt2)
        ax.locator_params(axis='y', nbins=5)
        ax.set_ylabel('Altitude (km)')
    extent = (0, (tstamps[-1] - tstamps[0]).total_seconds() /
              3600, iono.alt_km.values.min(), iono.alt_km.values.max())
    for idx, kax in enumerate(zip(keys, axs)):
        key, ax = kax
        title = exkeys[idx]
        if ' ' not in title:
            ax.set_title(r'$%s$' % title)
        else:
            ax.set_title(title)
        val = vals[key].T.copy()
        val[np.where(np.isnan(val))] = 1e-4
        if log:
            im = ax.imshow(np.log10(val), origin='lower',
                           aspect='auto', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)
            ticks = (np.asarray(cbar.ax.get_yticks()))
            ticks = np.round(ticks, decimals=0)
            ticks = np.linspace(ticks.min(), ticks.max(), 4, endpoint=True)
            ticks = np.round(ticks, decimals=1)
            cbar.ax.set_yticks(ticks)
            cbar.ax.set_yticklabels(
                [r'$10^{%.1f}$' % (tval) for tval in ticks])
            cbar.ax.locator_params('y')
            cbar.ax.tick_params(labelsize=8)
        else:
            im = ax.imshow(val, origin='lower',
                           aspect='auto', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)
            cbar.ax.locator_params('y')
            cbar.ax.tick_params(labelsize=8)
            ticks = np.asarray(cbar.ax.get_yticks())
            if np.log10(ticks.max() - ticks.min()) > np.log10(3e3):
                cbar.formatter.set_powerlimits((0, 0))
                # to get 10^3 instead of 1e3
                cbar.formatter.set_useMathText(True)
        # if 'comment' in iono[key].attrs:
        #     desc = str(iono[key].attrs['comment']).title()
        # else:
        #     desc = str(iono[key].attrs['long_name']).title()
        cbar.ax.set_ylabel(r'%s ($%s$)' % (
            iono[key].attrs['long_name'].title(), iono[key].attrs['units']), fontsize=8)
    xticks = np.asarray(axs[-1].get_xticks())
    xticks = np.round(xticks, decimals=1)
    xticks = list(map(lambda x: fmt_time(x, tstamps[0]), xticks))
    axs[-1].set_xticklabels(xticks)
    axs[-1].set_xlabel('Local Time')
    os.makedirs(dir_prefix, exist_ok=True)
    fig.savefig(
        f'{dir_prefix}/{file_prefix}_{day.replace("-", "")}.pdf', dpi=600)
    plt.show()

# %%


def plot_density2(iono: xr.Dataset, keys: Sequence[str], exkeys: Sequence[str], dir_prefix: str, file_prefix: str, *, vmin: Numeric | Iterable = None, vmax: Numeric | Iterable = None, log: bool = False, cmap: str = 'bone', alt_min: Numeric | Iterable = None, alt_max: Numeric | Iterable = None):
    dtime = parse(iono.time).astimezone(
        get_localzone()) - dt.timedelta(hours=8)
    day = dtime.strftime('%Y-%m-%d')
    tstamps = iono.tstamp.values
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    vals = {}
    key_ver = {}
    if alt_min is None:
        alt_min = 60
    if alt_max is None:
        alt_max = 800
    if not isinstance(vmin, Iterable):
        vmin = [vmin]*len(keys)
    if not isinstance(vmax, Iterable):
        vmax = [vmax]*len(keys)
    if not isinstance(alt_min, Iterable):
        alt_min = [alt_min]*len(keys)
    if not isinstance(alt_max, Iterable):
        alt_max = [alt_max]*len(keys)
    for idx, key in enumerate(keys):
        try:
            _ = int(key)
            arr = iono['ver'].sel(
                {'wavelength': key, 'alt_km': slice(alt_min[idx], alt_max[idx])}).values
            key_ver[key] = True
        except ValueError:
            arr = iono[key].sel({'alt_km': slice(alt_min[idx], alt_max[idx])}).values
            key_ver[key] = False
        # _, arr = fill_array(arr, tstamps, axis=0)
        vals[key] = arr
    for key in keys[:-1]:
        _, arr, _ = fill_array(vals[key].copy(), tstamps, axis=0)
        vals[key] = arr
    tstamps, vals[keys[-1]], nanfill = \
        fill_array(vals[keys[-1]].copy(), tstamps, axis=0)

    start = tstamps[0].astimezone(pytz.timezone('US/Eastern'))
    start = pd.to_datetime(start).round('1h').to_pydatetime()
    # start = dt.datetime(start.year, start.month, start.day,
    #                     start.hour, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    end = tstamps[-1].astimezone(pytz.timezone('US/Eastern'))
    end = pd.to_datetime(end).round('1h').to_pydatetime()

    ttstamps = [(t.timestamp() - start.timestamp()) /
                3600 for t in tstamps]

    fig, axs = plt.subplots(len(keys) // 2, 2, figsize=(
        6, (len(keys) // 2)*1.6), sharex=True, tight_layout=True)
    fig.suptitle('Vertical Profile\n%s - %s (UTC-5:00)' %
                 (tstamps[0].strftime('%Y-%m-%d %H:%M'), tstamps[-1].strftime('%Y-%m-%d %H:%M')))
    cax = make_color_axis(axs.flatten())
    fig.set_dpi(300)
    matplotlib.rcParams.update({'font.size': 10})
    matplotlib.rcParams.update({'axes.titlesize': 10})
    matplotlib.rcParams.update({'axes.labelsize': 10})

    def fmt(x, pos):
        a, b = '{:.1e}'.format(x).split('e')
        b = int(b)
        return r'$10^{{{}}}$'.format(a)

    def fmt2(x, pos):
        x = int(x + 18)
        return r'${}^\circ$'.format(x)
    for ax in axs.flatten():
        # ax.yaxis.set_major_formatter(fmt2)
        ax.locator_params(axis='y', nbins=5)
        ax.set_ylabel('Altitude (km)')

    for idx, kax in enumerate(zip(keys, axs.flatten())):
        key, ax = kax
        ax: plt.Axes = ax
        title = exkeys[idx]
        if ' ' not in title:
            ax.set_title(rf'{title}')
        else:
            ax.set_title(title)
        val = vals[key].T.copy()
        val[np.where(np.isnan(val))] = 1e-4
        alt_km = iono.alt_km.sel({'alt_km': slice(alt_min[idx], alt_max[idx])}).values
        tx, hy = np.meshgrid(ttstamps, alt_km)
        if log:
            im = ax.pcolormesh(tx, hy, np.log10(
                val), cmap=cmap, vmin=vmin[idx], vmax=vmax[idx])
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)
            ticks = (np.asarray(cbar.ax.get_yticks()))
            ticks = np.round(ticks, decimals=0)
            ticks = np.linspace(ticks.min(), ticks.max(), 4, endpoint=True)
            ticks = np.round(ticks, decimals=1)
            cbar.ax.set_yticks(ticks)
            cbar.ax.set_yticklabels(
                [r'$10^{%.1f}$' % (tval) for tval in ticks])
            cbar.ax.locator_params('y')
            cbar.ax.tick_params(labelsize=8)
        else:
            im = ax.pcolormesh(tx, hy, val, cmap=cmap,
                               vmin=vmin[idx], vmax=vmax[idx])
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)
            cbar.ax.locator_params('y')
            cbar.ax.tick_params(labelsize=8)
            ticks = np.asarray(cbar.ax.get_yticks())
            if np.log10(ticks.max() - ticks.min()) > np.log10(3e3):
                cbar.formatter.set_powerlimits((0, 0))
                # to get 10^3 instead of 1e3
                cbar.formatter.set_useMathText(True)
        # if 'comment' in iono[key].attrs:
        #     desc = str(iono[key].attrs['comment']).title()
        # else:
        #     desc = str(iono[key].attrs['long_name']).title()
        try:
            _ = int(key)
            arr = iono['ver'].sel({'wavelength': key})
        except ValueError:
            arr = iono[key]
        if not key_ver[key]:
            cbar.ax.set_ylabel(r'%s ($%s$)' % (
                arr.attrs['long_name'].title(), arr.attrs['units']), fontsize=8)
        else:
            cbar.ax.set_ylabel(r'%s ($%s$)' %
                               ('VER', arr.attrs['units']), fontsize=8)
    xticks = np.asarray(axs[-1, 0].get_xticks())
    xticks = np.round(xticks, decimals=1)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    axs[-1, 0].set_xticklabels(xticks)
    axs[-1, 1].set_xticklabels(xticks)
    axs[-1, 0].set_xlabel('Local Time (UTC$-$5:00)')
    axs[-1, 1].set_xlabel('Local Time (UTC$-$5:00)')
    os.makedirs(dir_prefix, exist_ok=True)
    ofile = f'{dir_prefix}/{file_prefix}_{day.replace("-", "")}.pdf'
    fig.savefig(ofile, dpi=600, bbox_inches='tight')
    print(f'Saved {ofile}')
    plt.show()


# %%
if __name__ == '__main__':
    files = glob.glob(f'{MODEL_DIR}/fitres*.xz')
    files.sort(key=get_date)

    dates = list(map(get_date, files))
    lat, lon = 42.64981361744372, -71.31681056737486

    with multiprocessing.Pool(4) as pool:
        res = pool.map(generate_vert, zip(dates, files))
    # res = list(map(generate_vert, zip(dates, files)))
    ionos = res[0]
    # %%
    for date, file in zip(dates, files):
        iono = xr.load_dataset(f'{MODEL_DIR}/vert_{date}.nc')
        # keys = ['O', 'O+', 'O2+']
        # exkeys = ['O', 'O^+', 'O_2^+']
        # plot_density(iono, keys, exkeys, 'fitprops_vert', 'o_den', vmin=2, log=False, cmap='gist_ncar_r')
        # keys = ['NS', 'N2D', 'NeIn']
        # exkeys = ['N(2S)', 'N(2D)', 'e^-']
        # plot_density(iono, keys, exkeys, 'fitprops_vert', 'n_den', vmin=2, log=False, cmap='gist_ncar_r')
        # keys = ['O', 'O+', 'O2+'] + ['NS', 'N2D', 'NeIn']
        keys = ['O', 'O+', 'O2+'] + ['O2', 'N2', 'NeIn'] #, '5577', '6300']
        # exkeys = ['O', 'O^+', 'O_2^+'] + ['N(2S)', 'N(2D)', 'e^-']
        exkeys = ['O', 'O$^+$', 'O$_2^+$'] + ['O$_2$', 'N$_2$', 'e$^-$'] #, '5577 Å', '6300 Å']
        altmin = [70, 100, 70, 60, 60, 100]
        altmax = [200, 800, 400, 100, 100, 800]
        plot_density2(iono, keys, exkeys, VERTPROPS_DIR, 'all_den', vmin=[
                      2, 2, 2, 2, 2, 2]
                      #, 1e-4, 1e-4]
                      , log=False, cmap='gist_ncar_r',
                      alt_min=altmin, alt_max=altmax)
        keys = ['Tn', 'Ti', 'Te']
        exkeys = ['Neutral Temperature',
                  'Ion Temperature', 'Electron Temperature']
        plot_density(iono, keys, exkeys, VERTPROPS_DIR,
                     'temps', vmin=100, log=False, cmap='hot')

# %%