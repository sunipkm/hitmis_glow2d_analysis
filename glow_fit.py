# %% Imports
from __future__ import annotations
from settings import MODEL_DIR, FIT_SAVE_FIGS, FIT_SHOW_FIGS, COUNTS_DIR
from collections.abc import Iterable
import datetime as dt
from functools import partial
import gc
import lzma
import pickle
from time import perf_counter_ns
from typing import Dict, List, SupportsFloat as Numeric, Tuple
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
import multiprocessing as mp

import geomagdata as gi
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import least_squares, OptimizeResult
from skmpython import GenericFit

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

print(f'Model directory: {MODEL_DIR}')
# %%


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


# %%
sds = xr.load_dataset('keo_scale.nc')
scale_5577 = sds['5577'].values[::-1]
scale_6300 = sds['6300'].values[::-1]
za_min = sds['za_min'].values
za_max = sds['za_max'].values
# %%


class GLOWMin:
    def __init__(self, time: dt.datetime, lat: Numeric, lon: Numeric, heading: Numeric, geomag_params: Dict[str, Numeric], za_min: np.ndarray, za_max: np.ndarray, za_idx: int, br: Numeric, ratio: Numeric, d_br: Numeric, d_rat: Numeric, save_walk: bool):
        self._time = time
        self._lat = lat
        self._lon = lon
        self._heading = heading
        self._geopar = geomag_params
        self._zamin = za_min
        self._zamax = za_max
        self._zaidx = za_idx
        self._br = br
        self._ratio = ratio
        self._dbr = d_br
        self._drat = d_rat
        self._iter = 0
        self._param = None
        self._diff = None
        self._bright: List[np.ndarray, np.ndarray] = None
        self._out = []
        self._save = save_walk
        self._pool = mp.Pool(processes=mp.cpu_count())
        self._start = perf_counter_ns()

    @property
    def fit_params(self):
        return self._param

    @property
    def fit_perf(self):
        return self._diff

    @property
    def walk(self):
        return np.asarray(self._out).T

    @property
    def emission(self):
        return self._bright

    def update(self, *params):
        self._iter += 1
        if len(params) == 1:
            params = params[0]
        self._param = params
        iono = glow2d.polar_model(self._time, self._lat, self._lon, self._heading, n_pts=20,
                                  geomag_params=self._geopar, Q=None, Echar=None,
                                  density_perturbation=(
                                      params[0], params[1], params[2], params[3], params[4], 1, params[5]),
                                  show_progress=False, mpool=self._pool)
        ec5577 = glow2d.glow2d_polar.get_emission(
            # ascending
            iono, feature='5577', za_min=self._zamin, za_max=self._zamax)[::-1]
        ec6300 = glow2d.glow2d_polar.get_emission(
            iono, feature='6300', za_min=self._zamin, za_max=self._zamax)[::-1]
        # 16 points around the midpoint
        idxs = slice(self._zaidx-8, self._zaidx+8)
        # idxs = [self._zaidx] # single point solver
        br_val = np.nanmean(ec6300[idxs])
        ratio_val = np.nanmean(ec5577[idxs] / br_val)
        ret = ((((br_val - self._br) / self._br)**2) * 65
               + 35 * (((ratio_val - self._ratio) / self._ratio)**2)) / 100
        if self._save:
            self._out.append(
                (params[0], params[1], params[2], params[3], params[4], params[5], ret))
        now = perf_counter_ns()
        if (now - self._start) > 120e9:
            print(f'Iteration {self._iter}: ({params}) | Err: {
                  ret:.2e}', end='\r')
            sys.stdout.flush()
        # ret = ((br_val - self._br)/self._dbr)**2 + ((ratio_val - self._ratio)/self._drat)**2
        self._diff = (br_val, self._br, ratio_val, self._ratio, ret)
        self._bright = [ec5577[::-1], ec6300[::-1]]  # descending
        # if not self._iter % 5:
        #     print(f'Iteration {self._iter}: ({params}) Brightness {br_val:.2e} ({self._br:.2e}) | Ratio {ratio_val:.2e} ({self._ratio:.2e}) | Err: {ret:.2e}')
        #     sys.stdout.flush()

        return ret


# %%
# dates = ['20220209']
dates = ['20220126', '20220209', '20220215', '20220218',
         '20220219', '20220226', '20220303', '20220304']
za_idx = 20
for date in dates:
    ds = xr.load_dataset(f'{COUNTS_DIR}/hitmis_cts_{date}.nc')
    tstamps = ds.tstamp.values
    start = pd.to_datetime(tstamps[0]).to_pydatetime()
    end = pd.to_datetime(tstamps[-1]).to_pydatetime()
    start += dt.timedelta(hours=1)
    end -= dt.timedelta(hours=1)
    # start = end - dt.timedelta(hours=2)
    # end = start + dt.timedelta(hours=2)
    ds = ds.loc[dict(tstamp=slice(start, end))]
    tstamps = ds.tstamp.values
    height = sds.height.values
    dheight = np.mean(np.diff(height))
    tstamps = list(map(lambda t: pd.to_datetime(
        t).to_pydatetime().astimezone(pytz.utc), tstamps))
    ttstamps = list(map(lambda i: (
        tstamps[i] - tstamps[0]).total_seconds()/3600, range(len(tstamps))))
    imgs_5577 = (ds['5577'].values.T[::-1, :])[za_idx, :]
    stds_5577 = (ds['5577_std'].values.T[::-1, :])[za_idx, :]
    imgs_6300 = (ds['6300'].values.T[::-1, :])[za_idx, :]
    stds_6300 = (ds['6300_std'].values.T[::-1, :])[za_idx, :]
    imgs_6306 = (ds['6306'].values.T[::-1, :])[za_idx, :]
    stds_6306 = (ds['6306'].values.T[::-1, :])[za_idx, :]
    imgs_5577 = gaussian_filter(np.ma.array(
        imgs_5577, mask=np.isnan(imgs_5577)), sigma=2)*scale_5577[za_idx]
    stds_5577 = gaussian_filter(np.ma.array(
        stds_5577, mask=np.isnan(stds_5577)), sigma=2)*scale_5577[za_idx]
    imgs_6300 = gaussian_filter(np.ma.array(
        imgs_6300, mask=np.isnan(imgs_6300)), sigma=2)*scale_6300[za_idx]
    stds_6300 = gaussian_filter(np.ma.array(
        stds_6300, mask=np.isnan(stds_6300)), sigma=2)*scale_6300[za_idx]

    lat, lon = 42.64981361744372, -71.31681056737486
    _, ap, f107, f107a, f107p = get_smoothed_geomag(tstamps)
    br6300 = np.zeros((len(ds.tstamp), len(ds.height)), dtype=float)
    br5577 = np.zeros((len(ds.tstamp), len(ds.height)), dtype=float)
    fparams = np.zeros((len(ds.tstamp), 6), dtype=float)
    fit_res = []
    failed = 0
    pbar = tqdm(range(len(ds.tstamp.values)))

    if FIT_SHOW_FIGS:
        fig, ax = plt.subplots(2, 1, figsize=(
            6, 4.8), sharex=True, tight_layout=True)
        fig.suptitle('%s - %s (US/East)' %
                     (start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M')))
        # cax = make_color_axis(ax)
        fig.set_dpi(100)
        matplotlib.rcParams.update({'font.size': 10})
        matplotlib.rcParams.update({'axes.titlesize': 10})
        matplotlib.rcParams.update({'axes.labelsize': 10})
        [ax[i].set_title(wl) for i, wl in enumerate(('5577 Å', '6300 Å'))]

        line, = ax[0].plot(
            ttstamps, (imgs_5577), color='g')
        l_5577, = ax[0].plot([0], [np.nan], color='g', ls='-.')
        s_5577 = ax[0].scatter([0], [np.nan], marker='x', color='k')
        line, = ax[1].plot(
            ttstamps, (imgs_6300), color='r')
        l_6300, = ax[1].plot([0], [np.nan], color='r', ls='-.')
        s_6300 = ax[1].scatter([0], [np.nan], marker='x', color='k')
        ax[0].fill_between(ttstamps, imgs_5577 + stds_5577,
                           imgs_5577 - stds_5577, alpha=0.5, color='r')
        ax[1].fill_between(ttstamps, imgs_6300 + stds_6300,
                           imgs_6300 - stds_6300, alpha=0.5, color='r')
        ax[1].set_xlim(np.min(ttstamps), np.max(ttstamps))
        plt.ion()
        plt.show()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    LOW = 0.1
    HIGH = 4.0

    x0 = (1, 1, 1, 1, 1, 1)
    for idx in pbar:
        if idx == (len(ds.tstamp.values) // 2):
            save = True
        else:
            save = False
        # do fit
        try:
            bgt = imgs_6300[idx]
            rat = (imgs_5577[idx] / imgs_6300[idx])
            if np.isnan(bgt) or np.isnan(rat):
                raise ValueError('bgt/rat NaN')
            b63 = uncertainties.ufloat(imgs_6300[idx], stds_6300[idx])
            b57 = uncertainties.ufloat(imgs_5577[idx], stds_5577[idx])
            brat: uncertainties.UFloat = b57 / b63
            rat = brat.nominal_value
            d_rat = brat.std_dev
            geomag_params = (f107a[idx], f107[idx], f107p[idx], ap[idx])
            minf = GLOWMin(tstamps[idx], lat, lon, 40, geomag_params=geomag_params, za_min=za_min,
                           za_max=za_max, za_idx=za_idx, br=bgt, ratio=rat, d_br=b63.std_dev, d_rat=d_rat, save_walk=save)
            res: OptimizeResult = \
                least_squares(minf.update, x0=x0,
                              bounds=((LOW, LOW, LOW, LOW, LOW, LOW),
                                      (HIGH, HIGH, HIGH, HIGH, HIGH, HIGH)),
                              diff_step=0.05, xtol=1e-10, ftol=1e-3, max_nfev=3000)
            if save:
                out = minf.walk

            fit_res.append((ds.tstamp.values[idx], res))
            x0 = (res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5])
            fp = minf.fit_params
            perf = list(minf.fit_perf)
            br_diff = ((perf[0] - perf[1]) / perf[1])*100
            br_diff_str = '%+.2f' % (br_diff)
            pbar.set_description(
                f'[{fp[0]:.2f} {fp[1]:.2f} {fp[2]:.2f} {fp[3]:.2f} {fp[4]:.2f} {fp[5]:.2f}] ({perf[1]:.2e}){br_diff_str}% | {perf[2]:.2f}<->{perf[3]:.2f} ({failed}) ', refresh=True)
            out = minf.emission
            br5577[idx, :] += out[0]
            br6300[idx, :] += out[1]
            fparams[idx, :] += fp
        except Exception:
            fit_res.append((ds.tstamp.values[idx], None))
            br5577[idx, :] += np.nan
            br6300[idx, :] += np.nan
            fparams[idx, :] += np.nan
            failed += 1
            pbar.set_description(f'Failed {idx + 1}', refresh=True)

        if FIT_SHOW_FIGS:
            l_5577.set_data(ttstamps[:idx+1],
                            br5577.T[::-1, :idx+1][za_idx, :])
            s_5577.set_offsets([ttstamps[idx], br5577.T[::-1, :][za_idx, idx]])

            ax[0].set_ylim(min(np.ma.array(br5577[:idx+1, za_idx], mask=np.isnan(br5577[:idx+1, za_idx])).min(), np.ma.array((imgs_5577 - 2*stds_5577), mask=np.isnan((imgs_5577 - 2*stds_5577))).min()),
                           max(np.ma.array(br5577[:idx+1, za_idx], mask=np.isnan(br5577[:idx+1, za_idx])).max(), np.ma.array((imgs_5577 + 2*stds_5577), mask=np.isnan((imgs_5577 + 2*stds_5577))).max()))

            l_6300.set_data(ttstamps[:idx+1],
                            br6300.T[::-1, :idx+1][za_idx, :])
            s_6300.set_offsets([ttstamps[idx], br6300.T[::-1, :][za_idx, idx]])

            ax[1].set_ylim(min(np.ma.array(br6300[:idx+1, za_idx], mask=np.isnan(br6300[:idx+1, za_idx])).min(), np.ma.array((imgs_6300 - 2*stds_6300), mask=np.isnan((imgs_6300 - 2*stds_6300))).min()),
                           max(np.ma.array(br6300[:idx+1, za_idx], mask=np.isnan(br6300[:idx+1, za_idx])).max(), np.ma.array((imgs_6300 + 2*stds_6300), mask=np.isnan((imgs_6300 + 2*stds_6300))).max()))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    kds = xr.Dataset(
        data_vars={'5577': (('tstamp', 'height'), br5577),
                   '6300': (('tstamp', 'height'), br6300),
                   'ap': (('tstamp'), ap),
                   'f107a': (('tstamp'), f107a),
                   'f107': (('tstamp'), f107),
                   'f107p': (('tstamp'), f107p),
                   'density_perturbation': (('tstamp', 'elems'), fparams),
                   'lat': (('tstamp'), [lat]*len(tstamps)),
                   'lon': (('tstamp'), [lon]*len(tstamps)),
                   'to_r': 1/(dheight * 4*np.pi*1e-6)},
        coords={'tstamp': ds.tstamp.values, 'height': sds.height.values,
                'elems': ['O', 'O2', 'N2', 'N4S', 'N2D', 'e']}
    )
    unit_desc = {
        '5577': ('cm^{-2} s^{-1} rad^{-1}', '5577 Brightness'),
        '6300': ('cm^{-2} s^{-1} rad^{-1}', '6300 Brightness'),
        'ap': ('', 'Planetary ap index (3 hour UTC)'),
        'f107a': ('sfu', '81-day rolling average of F10.7 solar flux'),
        'f107': ('sfu', 'F10.7 solar flux on present day'),
        'f107p': ('sfu', 'F10.7 solar flux on previous day'),
        'density_perturbation': ('', 'Relative density perturbation'),
        'lat': ('deg', 'Latitude'),
        'lon': ('deg', 'Longitude'),
        'to_r': ('R rad^{-1}', 'Convert brightness to Rayleigh')
    }
    _ = list(map(lambda x: kds[x].attrs.update(
        {'units': unit_desc[x][0], 'description': unit_desc[x][1]}), unit_desc.keys()))
    kds.to_netcdf(f'{MODEL_DIR}/keofit_{date}.nc')

    with lzma.open(f'{MODEL_DIR}/fitres_{date}.xz', 'wb') as fstr:
        pickle.dump(fit_res, fstr)

    mod_5577 = kds['5577'].values.T[::-1, :]
    mod_6300 = kds['6300'].values.T[::-1, :]

    if FIT_SHOW_FIGS:
        if FIT_SAVE_FIGS:
            fig.savefig(f'{MODEL_DIR}/fit_{date}.png')

        plt.close(fig=fig)
        plt.ioff()

# %%
