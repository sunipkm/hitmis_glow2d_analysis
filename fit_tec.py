# %% Imports
from __future__ import annotations
from settings import MODEL_DIR, FITPROPS_DIR
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

print(f'Loaded settings: {MODEL_DIR}, {FITPROPS_DIR}')
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


def fill_array(arr: np.ndarray, tstamps: List[dt.datetime], axis: int = 1) -> Tuple[List[dt.datetime], np.ndarray]:
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
        return tstamps, arr
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
    return (tstamps, out)
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

# %% GPS TEC


def geocent_to_geodet(lat: Numeric, ell: Tuple[Numeric, Numeric] = (6378137.0, 6356752.3142)) -> Numeric:
    """Converts geocentric latitude to geodetic latitude

    Args:
        lat (Numeric): Geographic latitude (degrees)
        ell (Tuple[Numeric, Numeric], optional): Semi-major and semi-minor axes. Defaults to WGS84(6378137.0, 6356752.3142).

    Returns:
        Numeric: Geodedic latitude (degrees)
    """
    a, b = ell
    assert (a > 0 and b > 0)
    return np.rad2deg(np.arctan2(a*np.tan(np.deg2rad(lat)), b))


def geodet_to_geocent(lat: Numeric, ell: Tuple[Numeric, Numeric] = (6378137.0, 6356752.3142)) -> Numeric:
    """Converts geodetic latitude to geocentric latitude

    Args:
        lat (Numeric): Geodetic latitude (degrees)
        ell (Tuple[Numeric, Numeric], optional): Semi-major and semi-minor axes. Defaults to WGS84(6378137.0, 6356752.3142).

    Returns:
        Numeric: Geocentric latitude (degrees)
    """
    a, b = ell
    assert (a > 0 and b > 0)
    return np.rad2deg(np.arctan(b*np.tan(np.deg2rad(lat))/a))


@staticvars(gpstec=None)
def get_gps_tec(tstart: Numeric, tstop: Numeric, latrange: slice = None, lonrange: slice = None, *, fname: str = 'gpstec_lowell.nc') -> xr.Dataset:
    if get_gps_tec.gpstec is None:
        get_gps_tec.gpstec = xr.open_dataset(fname)
    gpstec: xr.Dataset = get_gps_tec.gpstec
    gpstec = gpstec.sel(timestamps=slice(tstart, tstop))
    if latrange is not None:
        gpstec = gpstec.sel(gdlat=latrange)
    if lonrange is not None:
        gpstec = gpstec.sel(glon=lonrange)
    return gpstec
# %%


def get_tec(iono: xr.Dataset) -> np.ndarray:
    from scipy.integrate import trapz
    ne = iono['NeOut'].values.copy()
    ne = np.nan_to_num(ne, nan=0)
    alt = iono['alt_km'].values
    tsh, _ = ne.shape
    tec = np.zeros(tsh)
    for idx in range(tsh):
        tec[idx] += 2*trapz(ne[idx, :], alt)
    tec *= 1e9  # convert to m^-2
    return tec


def plot_tec(date: str, ax: plt.Axes) -> Tuple[dt.datetime, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    import digisondeindices as di
    iono = xr.load_dataset(f'{MODEL_DIR}/vert_{date}.nc')
    tstamps = iono.tstamp.values
    print(tstamps[0], tstamps[-1])
    lat, lon = 42.64981361744372, -71.31681056737486
    dlat = geocent_to_geodet(lat)
    gpsstart = int(tstamps[0])*1e-9 - 600
    gpsstop = int(tstamps[-1])*1e-9 + 600
    gpstec = get_gps_tec(gpsstart, gpsstop, latrange=slice(
        dlat-0.5, lat+0.5), lonrange=slice(lon-0.5, lon+0.5))
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    start: dt.datetime = tstamps[0].astimezone(pytz.timezone('US/Eastern'))
    start = dt.datetime(start.year, start.month, start.day,
                        start.hour, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    end = tstamps[-1]
    tstamps = np.asarray(np.asarray(
        iono.tstamp.values, dtype=int), dtype=float)*1e-9
    day = start.strftime('%Y%m%d')
    st = start.strftime('%Y-%m-%d %H:%M')
    et = end.strftime('%Y-%m-%d %H:%M')
    tstamps_ = list(
        map(lambda t: dt.datetime.fromtimestamp(t, pytz.utc), tstamps))
    ds = di.get_indices(tstamps_, 'MHJ45')
    tec_tstamp = np.asarray(np.asarray(
        ds.time.values, dtype=int), dtype=float)*1e-9
    tec_val = ds.TEC.values.copy()*1e-16
    tec = get_tec(iono)*1e-16
    gpstec_tstamp = gpstec.timestamps.values.copy()
    tec_tstamp -= tstamps[0]
    gpstec_tstamp -= tstamps[0]
    tstamps -= tstamps[0]
    gpstec_tstamp /= 3600
    tstamps /= 3600
    tec_tstamp /= 3600
    # fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300, tight_layout=True)
    glow_tec, = ax.plot(tstamps[::2], tec[::2], ls='', marker='x',
                        color='k', markersize=4, markeredgewidth=0.5)
    digi_tec, = ax.plot(tec_tstamp, tec_val, ls='', marker='o',
                        color='r', markersize=2, markeredgewidth=0.5)
    gps_tec, _, _ = ax.errorbar(gpstec_tstamp[::2],
                                np.nanmean(gpstec.tec.values,
                                           axis=(1, 2))[::2],
                                yerr=np.nanmean(
                                    gpstec.dtec.values, axis=(1, 2))[::2],
                                color='b', ls='',
                                capsize=2, elinewidth=0.5,
                                markersize=4, markeredgewidth=0.5)
    # ax.set_xlabel('Local Time')
    # ax.set_ylabel(r'TEC ($10^{16} m^{-2}$)')
    # ax.set_xlim(0, tstamps.max())
    # xticks = np.asarray(ax.get_xticks())
    # xticks = np.round(xticks, decimals=1)
    # xticks = list(map(lambda x: fmt_time(x, start), xticks))
    # ax.set_xticklabels(xticks)
    tec_val, tec_tstamp = filter_nan(tec_val, tec_tstamp)
    gtec_val, gtec_tstamp = filter_nan(np.nanmean(
        gpstec.tec.values, axis=(1, 2)), gpstec_tstamp)
    gtec_val = interpolate_nan(gtec_val, gtec_tstamp, tec_tstamp)
    tec_ = interpolate_nan(tec, tstamps, tec_tstamp)
    df = pd.DataFrame({'tec': tec_, 'tec_val': gtec_val})
    gpscorr = df.tec.corr(df.tec_val)
    print('GPS Correlation:', gpscorr)
    tec = interpolate_nan(tec, tstamps, tec_tstamp)
    # print('Correlation:', np.correlate(tec, tec_val)/np.sqrt(np.correlate(tec, tec)*np.correlate(tec_val, tec_val)))
    df = pd.DataFrame({'tec': tec, 'tec_val': tec_val})
    digicorr = df.tec.corr(df.tec_val)
    print('Digisonde Correlation:', digicorr)
    ax.text(0.5, 0.9, f'{start:%Y-%m-%d}', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10)
    # ax.text(0.5, 0.8, f'Digisonde: {digicorr*100:.2f}%, GNSS: {gpscorr*100:.2f}%',
    #         transform=ax.transAxes,
    #         ha='center', va='bottom', fontsize=8)
    # ax.legend([glow_tec, digi_tec, gps_tec], ['GLOW Model TEC', f'Digisonde TEC ({digicorr*100:.2f}%)', f'GNSS TEC ({gpscorr*100:.2f}%)'])
    if tec.shape != tec_val.shape:
        print(tec.shape, tec_val.shape)
        raise RuntimeError('Shapes do not match: %s, %s' %
                           (tec.shape, tec_val.shape))
    return (start, tec, tec_val, tec_, gtec_val, digicorr*100, gpscorr*100)


def filter_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    loc = np.isnan(x)
    return x[~loc], y[~loc]


def interpolate_nan(y0: np.ndarray, x0: np.ndarray, x: np.ndarray) -> np.ndarray:
    loc = np.isnan(y0)
    y0_ = y0[~loc]
    x0_ = x0[~loc]
    y = np.interp(x, x0_, y0_)
    return y


# %%
if __name__ == '__main__':
    files = glob.glob(f'{MODEL_DIR}/fitres*.xz')
    files.sort(key=get_date)

    dates = list(map(get_date, files))
    lat, lon = 42.64981361744372, -71.31681056737486

    with multiprocessing.Pool(4) as pool:
        res = pool.map(generate_vert, zip(dates, files))
    ionos = res[0]

    num_rows = int(np.floor(len(dates) / 2))  # 2 columns
    fig, axes = plt.subplots(num_rows, 2, figsize=(
        4.8, 2*num_rows), sharex=True, sharey=True, dpi=300)
    fig.subplots_adjust(hspace=0, wspace=0.1)
    # fig.suptitle('Keogram Elevation: %.0f$^\circ$' % (np.rad2deg(height[za_idx]) + 18))

    ax_xlim = []
    data_min = []
    data_max = []
    datagaps: dict[int, tuple[Numeric]] = {}

    matplotlib.rcParams.update({'font.size': 10})
    matplotlib.rcParams.update({'axes.titlesize': 10})
    matplotlib.rcParams.update({'axes.labelsize': 10})

    tot_digicorr = 1
    tot_gpscorr = 1

    with open(f'{FITPROPS_DIR}/tec_correlation.csv', 'w') as csvout, open(f'{FITPROPS_DIR}/tec_correlation.tex', 'w') as texout:
        csvout.write('Date,Digisonde Correlation,GPS Correlation')
        texout.write(
            r"""
\begin{tabular}{ccc}
\hline
Date & Digisonde Correlation & GNSS Correlation \\
\hline"""
        )
        for date, ax in zip(dates, axes.flatten()):
            start, _, _, _, _, digicorr, gpscorr = plot_tec(date, ax)
            tot_digicorr *= digicorr
            tot_gpscorr *= gpscorr
            csvout.write(f'\n{start:%Y-%m-%d},{digicorr:.2f},{gpscorr:.2f}')
            texout.write(
                f'\n{start:%Y-%m-%d} & {digicorr:.2f}\% & {gpscorr:.2f}\% \\\\')

        tot_digicorr = tot_digicorr**(1/len(dates))
        tot_gpscorr = tot_gpscorr**(1/len(dates))
        csvout.write(f'\nGeomean,{tot_digicorr:.2f},{tot_gpscorr:.2f}')
        texout.write(
            f'\n\\hline\nGeomean & {tot_digicorr:.2f}\% & {tot_gpscorr:.2f}\% \\\\')
        texout.write(
            r"""
\hline
\end{tabular}
"""
        )
    print(f'Digisonde Correlation Geomean: {tot_digicorr:.2f}%')
    print(f'GPS Correlation Geomean: {tot_gpscorr:.2f}%')

    for axs in axes:
        ax = axs[0]
        ax.set_ylabel(r'VTEC (TECU)')

    for ax in axes.flatten()[-2:]:
        ax.set_xlim(0, 9)
        xticks = np.asarray(ax.get_xticks())
        xticks = np.round(xticks, decimals=1)
        xticks = list(map(lambda x: fmt_time(x, start), xticks))
        ax.set_xticklabels(xticks, rotation=45)
        ax.set_xlabel("Local Time (UTC$-$05:00)")
    plt.savefig(f'{FITPROPS_DIR}/tec_profile.pdf',
                dpi=600, bbox_inches='tight')
    plt.show()
# %%
