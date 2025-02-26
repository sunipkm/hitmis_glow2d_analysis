# %% Imports
from __future__ import annotations
from settings import COUNTS_DIR, MODEL_DIR, KEOGRAMS_DIR
from collections.abc import Iterable
import datetime as dt
from functools import partial
import gc
import lzma
import pickle
from typing import List, Tuple, SupportsFloat as Numeric
from scipy import ndimage
from skmpython import staticvars
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

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import geomagdata as gi
import digisondeindices as di

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
sds = xr.load_dataset('keo_scale.nc')
scale_5577 = sds['5577'].values[::-1]
scale_6300 = sds['6300'].values[::-1]
za_min = sds['za_min'].values
za_max = sds['za_max'].values
# %%
# dates = ['20220218']
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


def fmt_time(x: Numeric, ofst: dt.datetime) -> str:
    x = dt.timedelta(hours=x)
    res = ofst + x
    return res.strftime('%H:%M')


# %% Dates
filter = True
dates = glob.glob(f'{COUNTS_DIR}hitmis_cts_*.nc')
dates = list(map(lambda x: x.split('_')[-1].split('.')[0], dates))
if filter:
    dates_ = glob.glob(f'{MODEL_DIR}/fwdmodel_*.nc')
    dates_ = list(map(lambda x: x.split('_')[-1].split('.')[0], dates_))
    dates = list(set(dates).intersection(dates_))
dates.sort()
# %% Keogram
za_idx = 20
for fidx, date in enumerate(dates):
    ds = xr.load_dataset(f'{COUNTS_DIR}hitmis_cts_{date}.nc')
    if filter:
        mds = xr.load_dataset(f'{MODEL_DIR}/fwdmodel_{date}.nc')
        ds = ds.loc[dict(tstamp=mds.tstamp.values)]
    height = sds.height.values
    dheight = np.diff(height).mean()
    tstamps = ds.tstamp.values
    if (len(tstamps) == 0):
        continue
    imgs_5577 = ds['5577'].values.T[::-1, :]*scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    stds_5577 = ds['5577_std'].values.T[::-1, :]*scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    imgs_6300 = ds['6300'].values.T[::-1, :]*scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    stds_6300 = ds['6300_std'].values.T[::-1, :]*scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    imgs_6306 = ds['6306'].values.T[::-1, :]*scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    stds_6306 = ds['6306'].values.T[::-1, :]*scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    _, imgs_5577 = fill_array(imgs_5577, tstamps)
    _, stds_5577 = fill_array(stds_5577, tstamps)
    _, imgs_6300 = fill_array(imgs_6300, tstamps)
    _, stds_6300 = fill_array(stds_6300, tstamps)
    _, imgs_6306 = fill_array(imgs_6306, tstamps)
    tstamps, stds_6306 = fill_array(stds_6306, tstamps)
    start = tstamps[0].astimezone(pytz.timezone('US/Eastern'))
    start = pd.to_datetime(start).round('1h').to_pydatetime()
    # start = dt.datetime(start.year, start.month, start.day,
    #                     start.hour, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    end = tstamps[-1].astimezone(pytz.timezone('US/Eastern'))
    end = pd.to_datetime(end).round('1h').to_pydatetime()
    ttstamps = [(t.timestamp() - start.timestamp()) / 3600 for t in tstamps]
    height_ang = np.rad2deg(height[::-1])
    height_ang -= height_ang[za_idx] - 35

    print(f'Min height: {min(height_ang)}, Max height: {max(height_ang)}')

    tx, hy = np.meshgrid(ttstamps, height_ang)

    ts = list(map(lambda t: t.timestamp(), tstamps))
    dts = np.diff(ts)
    t_delta = dts.min()
    dts = np.diff(dts)
    oidx = np.where(dts < 0)[0]
    print(t_delta, oidx)
    nanloc = np.where(np.isnan(imgs_6300[0, :]))[0]
    nanfill = False
    if len(nanloc) > 0 and nanloc[-1] - nanloc[0] > 2:
        print('Too many nans')
        nanfill = True
    fig, ax = plt.subplots(3, 1, figsize = (6, 4.8), sharex = True, tight_layout = True)
    fig.suptitle('%s - %s (UTC-5:00)'%(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M')))
    cax = make_color_axis(ax)
    fig.set_dpi(300)
    matplotlib.rcParams.update({'font.size': 10})
    matplotlib.rcParams.update({'axes.titlesize': 10})
    matplotlib.rcParams.update({'axes.labelsize': 10})
    def fmt(x, pos):
        a, b = '{:.1e}'.format(x).split('e')
        b = int(b)
        return r'$10^{{{}}}$'.format(a)
    def fmt2(x, pos):
        # if np.allclose([x], [int(x)], atol=1e-3):
        #     x = int(x)
        #     return r'${}^\circ$'.format(x)
        # else:
            return r'${:.1f}^\circ$'.format(x)
    ax[0].yaxis.set_major_formatter(fmt2)
    ax[0].locator_params(axis='y', nbins=7)
    ax[1].yaxis.set_major_formatter(fmt2)
    ax[1].locator_params(axis='y', nbins=7)
    ax[2].yaxis.set_major_formatter(fmt2)
    ax[2].locator_params(axis='y', nbins=7)
    for axs in ax:
        axs.set_ylabel('Elevation')
    [ax[i].set_title(wl) for i, wl in enumerate(('5577 Å (Green)', '6300 Å (Red)', '6306 Å (Cloud Indicator)'))]
    im = ax[0].pcolormesh(tx, hy, np.log10(imgs_5577), cmap='bone')#, vmin=1.5, vmax=4)
    # im = ax[0].imshow(np.log10(imgs_5577), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone') #, vmin=1.5, vmax=4)
    cbar = fig.colorbar(im, cax=cax[0], shrink=0.5, format=fmt)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Intensity (R)', fontsize=8)
    im = ax[1].pcolormesh(tx, hy, np.log10(imgs_6300), cmap='bone')#, vmin=1.5, vmax=4)
    # im = ax[1].imshow(np.log10(imgs_6300), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone') #, vmin=1.5, vmax=4)
    cbar = fig.colorbar(im, cax=cax[1], shrink=0.5, format=fmt)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Intensity (R)', fontsize=8)
    im = ax[2].pcolormesh(tx, hy, np.log10(imgs_6306), cmap='bone', vmin=np.nanpercentile(np.log10(imgs_6306), 1), vmax=np.nanpercentile(np.log10(imgs_6306), 99))
    # im = ax[2].imshow(np.log10(imgs_6306), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone', vmin=np.nanpercentile(np.log10(imgs_6306), 1), vmax=np.nanpercentile(np.log10(imgs_6306), 99))
    cbar=fig.colorbar(im, cax=cax[2], shrink=0.5, format=fmt)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Intensity (R)', fontsize=8)
    ax[-1].set_xlim(0, 9)
    xticks = np.arange(10).astype(float)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    ax[-1].set_xticklabels(xticks)
    ax[-1].set_xlabel("Local Time")
    if nanfill:
        # 1. create axis
        trange = np.linspace(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, len(imgs_6300[0, :]), endpoint=True)
        tmin = nanloc[0] - 1
        tmax = nanloc[-1] + 1
        trange = trange[tmin:tmax + 1]
        # 2. Find nan locs
        ax[0].text((trange[-1] + trange[0])*0.5, np.mean(height_ang), 'Unavailable', ha='center', va='center', fontsize=8, rotation='vertical', color='r')
        ax[1].text((trange[-1] + trange[0])*0.5, np.mean(height_ang), 'Unavailable', ha='center', va='center', fontsize=8, rotation='vertical', color='r')
        ax[2].text((trange[-1] + trange[0])*0.5, np.mean(height_ang), 'Unavailable', ha='center', va='center', fontsize=8, rotation='vertical', color='r')
    plt.savefig(f'{KEOGRAMS_DIR}/hitmis_keo_{date}.pdf')
    plt.show()
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
def get_gps_tec(tstamps: Iterable[Numeric], lat: Iterable[Numeric], lon: Iterable[Numeric], angle: Iterable[Numeric], *, fname: str = 'gpstec_lowell.nc') -> xr.Dataset:
    if get_gps_tec.gpstec is None:
        get_gps_tec.gpstec = xr.open_dataset(fname)
    gpstec: xr.Dataset = get_gps_tec.gpstec
    gdlat = geocent_to_geodet(lat)
    assert (len(gdlat) == len(lon) == len(angle))
    gpstec = gpstec.sel(timestamps=tstamps, method='nearest')
    tecvals = np.zeros((len(tstamps), len(angle)))
    dtecvals = np.zeros((len(tstamps), len(angle)))
    for idx, (gl, lo) in enumerate(zip(gdlat, lon)):
        val = gpstec.sel(gdlat=gl, method='nearest')
        val = val.sel(glon=lo, method='nearest')
        tecvals[:, idx] = val.tec.values
        dtecvals[:, idx] = val.dtec.values
    gpstec = xr.Dataset({'tec': (('timestamps', 'angle'), tecvals),
                         'dtec': (('timestamps', 'angle'), dtecvals)},
                        coords={'timestamps': tstamps, 'angle': angle, 'lat': ('angle', gdlat), 'lon': ('angle', lon)})
    return gpstec


# %%
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

for fidx, (date, ax) in enumerate(zip(dates, axes.flatten())):
    ax: plt.Axes = ax
    ds = xr.load_dataset(f'{COUNTS_DIR}hitmis_cts_{date}.nc')
    mds = xr.load_dataset(f'{MODEL_DIR}/fwdmodel_{date}.nc')
    ds = ds.loc[dict(tstamp=mds.tstamp.values)]
    tstamps = ds.tstamp.values
    lat, lon = 42.64981361744372, -71.31681056737486
    if (len(tstamps) == 0):
        continue
    tecsrc = get_gps_tec(tstamps.astype(int)*1e-9, [lat], [lon], [0])
    height = sds.height.values
    dheight = np.diff(height).mean()
    imgs_5577 = ds['5577'].values.T[::-1, :] * \
        scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    stds_5577 = ds['5577_std'].values.T[::-1, :] * \
        scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    imgs_6300 = ds['6300'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    stds_6300 = ds['6300_std'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    imgs_6306 = ds['6306'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    stds_6306 = ds['6306'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    mds_5577 = mds['5577'].values.T[::-1, :] / dheight * 4*np.pi*1e-6
    mds_6300 = mds['6300'].values.T[::-1, :] / dheight * 4*np.pi*1e-6
    try:
        mds_ap = mds['ap'].values
    except Exception:
        continue
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    _, imgs_5577 = fill_array(imgs_5577, tstamps)
    _, stds_5577 = fill_array(stds_5577, tstamps)
    _, imgs_6300 = fill_array(imgs_6300, tstamps)
    _, stds_6300 = fill_array(stds_6300, tstamps)
    _, imgs_6306 = fill_array(imgs_6306, tstamps)
    _, stds_6306 = fill_array(stds_6306, tstamps)
    _, mds_5577 = fill_array(mds_5577, tstamps)
    _, mds_ap = fill_array(mds_ap[:, None], tstamps, axis=0)
    tstamps, mds_6300 = fill_array(mds_6300, tstamps)
    # _, mds_ap, _, _, _ = get_smoothed_geomag(tstamps)

    start = tstamps[0].astimezone(pytz.timezone('US/Eastern'))
    start = pd.to_datetime(start).round('1h').to_pydatetime()
    # start = dt.datetime(start.year, start.month, start.day,
    #                     start.hour, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    end = tstamps[-1].astimezone(pytz.timezone('US/Eastern'))
    end = pd.to_datetime(end).round('1h').to_pydatetime()
    ttstamps = [(t.timestamp() - start.timestamp()) / 3600 for t in tstamps]
    gps_tstamp = tecsrc.timestamps.values.copy()
    gps_tstamp = [pd.to_datetime(t*1e9).to_pydatetime() for t in gps_tstamp]
    gps_tstamp = [(t.timestamp() - start.timestamp()) /
                  3600 for t in gps_tstamp]
    height_ang = np.rad2deg(height[::-1])
    height_ang -= height_ang[za_idx] - 35
    # fig.suptitle('%s - %s (UTC-5:00) [Elevation: %.0f$^\circ$]' % (start.strftime(
    # '%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M'), height_ang))
    # cax = make_color_axis(ax)
    # fig.set_dpi(300)
    mds_ap[np.where(np.isnan(mds_6300[za_idx, :]))] = np.nan
    nanloc = np.where(np.isnan(imgs_6300[za_idx, :]))[0]
    nanfill = False
    if len(nanloc) > 0 and nanloc[-1] - nanloc[0] > 2:
        nanfill = True
    # [ax[i].set_title(wl) for i, wl in enumerate(('5577 Å', '6300 Å', '6306 Å'))]
    # im = ax[0].imshow((imgs_5577), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone')
    # fig.colorbar(im, cax=cax[0], shrink=0.5).ax.locator_params(nbins=5)
    # im = ax[1].imshow((imgs_6300 - mds_6300) / imgs_6300 * 100, aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone', vmin=-10, vmax=10)
    # fig.colorbar(im, cax=cax[1], shrink=0.5).ax.locator_params(nbins=5)
    # im = ax[2].imshow((imgs_6306), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone')
    # fig.colorbar(im, cax=cax[2], shrink=0.5).ax.locator_params(nbins=5)
    # tax = ax.twinx()
    # tax.set_ylabel('a$_p$ Index')
    # tax.set_ylim(0, 50)
    # l_ap, = tax.plot(ttstamps, mds_ap, ls='-.', color='k', lw=0.65)
    # tax = ax.twinx()
    # tax.set_ylim(-2, 32)
    # if fidx % 2:
    #     tax.set_ylabel('GNSS VTEC (TECU)', fontsize=8)
    # else:
    #      plt.setp(tax.get_yticklabels(), visible=False)

    # gps_tec, _, _ = tax.errorbar(gps_tstamp[::2],
    #                         tecsrc.tec.values.flatten()[::2],
    #                         yerr=tecsrc.dtec.values.flatten()[::2],
    #                         color='k', ls='',
    #                         capsize=2, elinewidth=0.5,
    #                         markersize=4, markeredgewidth=0.5,
    #                         zorder=0)
    ax.set_yscale('log')

    l_55, = ax.plot(
        ttstamps, imgs_5577[za_idx, :], ls=':', lw=0.65, color='forestgreen', zorder=1)
    m_55, = ax.plot(ttstamps, mds_5577[za_idx, :],
                    ls='-', lw=0.65, color='forestgreen', zorder=1)
    l_63, = ax.plot(
        ttstamps, imgs_6300[za_idx, :], ls=':', lw=0.65, color='r', zorder=1)
    m_63, = ax.plot(ttstamps, mds_6300[za_idx, :],
                    ls='-', lw=0.65, color='r', zorder=1)
    # ax.plot(ttstamps, imgs_6306[za_idx, :], ls='-', lw=0.65, color='k')
    f_55 = ax.fill_between(ttstamps, imgs_5577[za_idx, :] + 1*stds_5577[za_idx, :],
                           imgs_5577[za_idx, :] - 1*stds_5577[za_idx, :], alpha=0.4, color='forestgreen', edgecolor=None, zorder=1)
    f_63 = ax.fill_between(ttstamps, imgs_6300[za_idx, :] + 1*stds_6300[za_idx, :],
                           imgs_6300[za_idx, :] - 1*stds_6300[za_idx, :], alpha=0.25, color='r', edgecolor=None, zorder=1)

    # f_55 = ax.fill_between(ttstamps, imgs_5577[za_idx, :] + 2*stds_5577[za_idx, :],
    #                        imgs_5577[za_idx, :] - 2*stds_5577[za_idx, :], alpha=0.25, color='b', edgecolor=None)
    # f_63 = ax.fill_between(ttstamps, imgs_6300[za_idx, :] + 2*stds_6300[za_idx, :],
    #                        imgs_6300[za_idx, :] - 2*stds_6300[za_idx, :], alpha=0.1, color='r', edgecolor=None)
    ax_xlim.append((end - start).total_seconds() / 3600)
    ylim = ax.get_ylim()
    if fidx % 2 == 0:
        ax.set_ylabel('Intensity (R)')
    else:
        ax.yaxis.set_ticks_position('none')
    lobjs = [(l_55, f_55), m_55, (l_63, f_63), m_63]  # , l_ap]
    ltext = ['5577Å Measurement', '5577Å Model',
             '6300Å Measurement', '6300Å Model']  # , 'a$_p$ Index']
    if nanfill:
        tmin = nanloc[0] - 1
        tmax = nanloc[-1] + 1
        trange = np.asarray(ttstamps)[tmin:tmax + 1]
        nfb = ax.fill_between(trange, 1e-4, 1e8, color='k',
                              alpha=0.2, edgecolor=None, hatch='//')
        datagaps[fidx] = (trange.mean(),)
        lobjs.append(nfb)
        ltext.append('Data Unavailable')
    ax.set_ylim(ylim)
    ax.text(0.5, 0.99, start.strftime('%Y-%m-%d'),
            ha='center', va='top', transform=ax.transAxes)

    data_max.append(
        max(
            np.nanmax(imgs_5577[za_idx, :] + stds_5577[za_idx, :]),
            np.nanmax(imgs_6300[za_idx, :] + stds_6300[za_idx, :]),
            np.nanmax(mds_5577[za_idx, :]),
            np.nanmax(mds_6300[za_idx, :])
        )
    )
    data_min.append(
        min(
            np.nanmin(imgs_5577[za_idx, :] - stds_5577[za_idx, :]),
            np.nanmin(imgs_6300[za_idx, :] - stds_6300[za_idx, :]),
            np.nanmin(mds_5577[za_idx, :]),
            np.nanmin(mds_6300[za_idx, :])
        )
    )
    # ax.legend(
    #     lobjs, ltext
    # )
    # plt.savefig(f'{plotdir}/keo_fit_{date}.png', dpi=600)

dmin = min(data_min)
dmax = max(data_max)

formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
for idx, ax in enumerate(axes.flatten()):
    ax.set_xlim(0, max(ax_xlim))
    ax.set_ylim(dmin, dmax)
    if idx % 2 == 0:
        ax.yaxis.set_major_formatter(formatter)

for k, v in datagaps.items():
    ax = axes.flatten()[k]
    ylim = ax.get_ylim()
    ax.text(v[0], np.mean(ylim), 'Data Unavailable', ha='center',
            va='top', fontsize=8, color='r', rotation='vertical')

for ax in axes.flatten()[-2:]:
    xticks = np.asarray(ax.get_xticks())
    xticks = np.round(xticks, decimals=1)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlabel("Local Time (UTC$-$05:00)")
fig.savefig(f'{KEOGRAMS_DIR}/fwdmodel_lowell.pdf',
            dpi=600, bbox_inches='tight')
plt.show()
# %%


def filter_nan_gaussian_conserving(arr, sigma):
    from scipy import ndimage
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
        loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
        gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss


za_idx = 20

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

for fidx, (date, ax) in enumerate(zip(dates, axes.flatten())):
    ax: plt.Axes = ax
    ds = xr.load_dataset(f'{COUNTS_DIR}hitmis_cts_{date}.nc')
    mds = xr.load_dataset(f'{MODEL_DIR}/fwdmodel_{date}.nc')
    ds = ds.loc[dict(tstamp=mds.tstamp.values)]
    tstamps = ds.tstamp.values
    lat, lon = 42.64981361744372, -71.31681056737486
    if (len(tstamps) == 0):
        continue
    tecsrc = get_gps_tec(tstamps.astype(int)*1e-9, [lat], [lon], [0])
    height = sds.height.values
    dheight = np.diff(height).mean()
    imgs_5577 = ds['5577'].values.T[::-1, :] * \
        scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    stds_5577 = ds['5577_std'].values.T[::-1, :] * \
        scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    imgs_6300 = ds['6300'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    stds_6300 = ds['6300_std'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    imgs_6306 = ds['6306'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    stds_6306 = ds['6306'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    mds_5577 = mds['5577'].values.T[::-1, :] / dheight * 4*np.pi*1e-6
    mds_6300 = mds['6300'].values.T[::-1, :] / dheight * 4*np.pi*1e-6
    try:
        mds_ap = mds['ap'].values
    except Exception:
        continue
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    dds = di.get_indices(tstamps, 'MHJ45')
    _, imgs_5577 = fill_array(imgs_5577, tstamps)
    _, stds_5577 = fill_array(stds_5577, tstamps)
    _, imgs_6300 = fill_array(imgs_6300, tstamps)
    _, stds_6300 = fill_array(stds_6300, tstamps)
    _, imgs_6306 = fill_array(imgs_6306, tstamps)
    _, stds_6306 = fill_array(stds_6306, tstamps)
    _, mds_5577 = fill_array(mds_5577, tstamps)
    _, mds_ap = fill_array(mds_ap[:, None], tstamps, axis=0)
    _, hmf = fill_array(dds['hmF'].values[:, None], tstamps, axis=0)
    tstamps, mds_6300 = fill_array(mds_6300, tstamps)
    # _, mds_ap, _, _, _ = get_smoothed_geomag(tstamps)

    start = tstamps[0].astimezone(pytz.timezone('US/Eastern'))
    start = dt.datetime(start.year, start.month, start.day,
                        start.hour, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    end = tstamps[-1].astimezone(pytz.timezone('US/Eastern'))
    end = dt.datetime(end.year, end.month, end.day, end.hour,
                      0, 0, tzinfo=pytz.timezone('US/Eastern'))
    ttstamps = [(t.timestamp() - start.timestamp()) / 3600 for t in tstamps]
    hmf = hmf.flatten()
    height_ang = np.rad2deg(height[::-1])
    height_ang -= height_ang[za_idx] - 35
    print(f'Min height: {height_ang.min():.2f}, Max height: {height_ang.max():.2f}')
    # fig.suptitle('%s - %s (UTC-5:00) [Elevation: %.0f$^\circ$]' % (start.strftime(
    # '%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M'), height_ang))
    # cax = make_color_axis(ax)
    # fig.set_dpi(300)
    mds_ap[np.where(np.isnan(mds_6300[za_idx, :]))] = np.nan
    nanloc = np.where(np.isnan(imgs_6300[za_idx, :]))[0]
    nanfill = False
    if len(nanloc) > 0 and nanloc[-1] - nanloc[0] > 2:
        nanfill = True
    # [ax[i].set_title(wl) for i, wl in enumerate(('5577 Å', '6300 Å', '6306 Å'))]
    # im = ax[0].imshow((imgs_5577), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone')
    # fig.colorbar(im, cax=cax[0], shrink=0.5).ax.locator_params(nbins=5)
    # im = ax[1].imshow((imgs_6300 - mds_6300) / imgs_6300 * 100, aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone', vmin=-10, vmax=10)
    # fig.colorbar(im, cax=cax[1], shrink=0.5).ax.locator_params(nbins=5)
    # im = ax[2].imshow((imgs_6306), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone')
    # fig.colorbar(im, cax=cax[2], shrink=0.5).ax.locator_params(nbins=5)
    # tax = ax.twinx()
    # tax.set_ylabel('a$_p$ Index')
    # tax.set_ylim(0, 50)
    # l_ap, = tax.plot(ttstamps, mds_ap, ls='-.', color='k', lw=0.65)

    tax: plt.Axes = ax.twinx()
    tax.set_ylim(180, 380)
    if fidx % 2:
        tax.set_ylabel('hmF', fontsize=8)
    else:
        plt.setp(tax.get_yticklabels(), visible=False)
    tax.plot(ttstamps, hmf, markersize=0.4,
             marker='o', ls='', lw=0.65, color='b')
    # ax.set_yscale('log')

    gbr = np.einsum('ij,i->j', imgs_5577, np.arange(imgs_5577.shape[0]))
    rbr = np.einsum('ij,i->j', imgs_6300, np.arange(imgs_6300.shape[0]))
    BIGSIG = 3  # 1.5*6
    SMALLSIG = 2
    gbr_mean = np.nanmean(gbr)
    gbr = gbr - filter_nan_gaussian_conserving(gbr, BIGSIG)
    gbr = filter_nan_gaussian_conserving(gbr, SMALLSIG)
    gbr -= np.nanmin(gbr)
    gbr_std = np.nanstd(gbr)/gbr_mean
    gbr /= np.nanmax(gbr)
    rbr_mean = np.nanmean(rbr)
    rbr = rbr - filter_nan_gaussian_conserving(rbr, BIGSIG)
    rbr = filter_nan_gaussian_conserving(rbr, SMALLSIG)
    rbr -= np.nanmin(rbr)
    rbr_std = np.nanstd(rbr)/rbr_mean
    rbr /= np.nanmax(rbr)
    hmf_mean = np.nanmean(hmf)
    hmf = hmf - filter_nan_gaussian_conserving(hmf, BIGSIG)
    hmf_std = np.nanstd(hmf)
    hmf = filter_nan_gaussian_conserving(hmf, SMALLSIG)
    hmf -= np.nanmin(hmf)
    hmf /= np.nanmax(hmf)
    l_55, = ax.plot(ttstamps, gbr, ls='-', lw=0.65,
                    color='forestgreen', zorder=1)
    l_63, = ax.plot(ttstamps, rbr, ls=':', lw=0.65, color='r', zorder=1)
    l_hmf, = ax.plot(ttstamps, hmf, ls='-.', color='k', lw=0.65)
    # m_55, = ax.plot(ttstamps, mds_5577[za_idx, :], ls='-', lw=0.65, color='forestgreen', zorder=1)
    # l_63, = ax.plot(ttstamps,np.einsum('ij,i->j', imgs_6300, np.arange(imgs_6300.shape[0])), ls=':', lw=0.65, color='r', zorder=1)
    # m_63, = ax.plot(ttstamps, mds_6300[za_idx, :], ls='-', lw=0.65, color='r', zorder=1)
    # ax.plot(ttstamps, imgs_6306[za_idx, :], ls='-', lw=0.65, color='k')

    # f_55 = ax.fill_between(ttstamps, imgs_5577[za_idx, :] + 2*stds_5577[za_idx, :],
    #                        imgs_5577[za_idx, :] - 2*stds_5577[za_idx, :], alpha=0.25, color='b', edgecolor=None)
    # f_63 = ax.fill_between(ttstamps, imgs_6300[za_idx, :] + 2*stds_6300[za_idx, :],
    #                        imgs_6300[za_idx, :] - 2*stds_6300[za_idx, :], alpha=0.1, color='r', edgecolor=None)
    ax_xlim.append((end - start).total_seconds() / 3600)
    ylim = ax.get_ylim()
    if fidx % 2 == 0:
        ax.set_ylabel('Normalized Variation')
    else:
        ax.yaxis.set_ticks_position('none')
    lobjs = [(l_55, f_55), m_55, (l_63, f_63), m_63]  # , l_ap]
    ltext = ['5577Å Measurement', '5577Å Model',
             '6300Å Measurement', '6300Å Model']  # , 'a$_p$ Index']
    if nanfill:
        tmin = nanloc[0] - 1
        tmax = nanloc[-1] + 1
        trange = np.asarray(ttstamps)[tmin:tmax + 1]
        nfb = ax.fill_between(trange, -0.05, 1.05, color='k',
                              alpha=0.2, edgecolor=None, hatch='//')
        datagaps[fidx] = (trange.mean(),)
        lobjs.append(nfb)
        ltext.append('Data Unavailable')
    ax.set_ylim(ylim)
    ax.text(0.5, 0.99, f'{start:%Y-%m-%d}\nhmF variation: {2*hmf_std:.2f} km',
            ha='center', va='top', transform=ax.transAxes, fontsize=8)
    # ax.legend(
    #     lobjs, ltext
    # )
    # plt.savefig(f'{plotdir}/keo_fit_{date}.png', dpi=600)

formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
for idx, ax in enumerate(axes.flatten()):
    ax.set_xlim(0, max(ax_xlim))
    ax.set_ylim(-0.05, 1.05)
    if idx % 2 == 0:
        ax.yaxis.set_major_formatter(formatter)

for k, v in datagaps.items():
    ax = axes.flatten()[k]
    ylim = ax.get_ylim()
    ax.text(v[0], 0.5, 'Data Unavailable', ha='center',
            va='center', fontsize=8, color='r', rotation='vertical')

for ax in axes.flatten()[-2:]:
    xticks = np.asarray(ax.get_xticks())
    xticks = np.round(xticks, decimals=1)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlabel("Local Time (UTC$-$05:00)")
fig.savefig(f'{KEOGRAMS_DIR}/hmf_variation.pdf', dpi=600, bbox_inches='tight')
plt.show()
# %%
