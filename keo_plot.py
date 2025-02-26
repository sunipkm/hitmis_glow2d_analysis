# %% Imports
from __future__ import annotations
from collections.abc import Iterable
import datetime as dt
from functools import partial
import gc
import lzma
import pickle
from typing import List, Tuple, SupportsFloat as Numeric
from matplotlib.gridspec import GridSpec
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from BaselineRemoval import BaselineRemoval
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

from settings import COUNTS_DIR, MODEL_DIR, KEOGRAMS_DIR
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


# %% Keogram
za_idx = 20

dates = ['20220126', '20220209', '20220215', '20220218',
         '20220219', '20220226', '20220303', '20220304']
# %%
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
    ds = xr.load_dataset(f'{COUNTS_DIR}/hitmis_cts_{date}.nc')
    mds = xr.load_dataset(f'{MODEL_DIR}/keofit_{date}.nc')
    ds = ds.loc[dict(tstamp=mds.tstamp.values)]
    tstamps = ds.tstamp.values
    if (len(tstamps) == 0):
        continue
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
    print(start, end)
    # end = dt.datetime(end.year, end.month, end.day, end.hour,
    #                   0, 0, tzinfo=pytz.timezone('US/Eastern'))
    ttstamps = [(t.timestamp() - start.timestamp()) / 3600 for t in tstamps]
    height_ang = np.rad2deg(height[::-1][za_idx])
    height_ang -= height_ang - 35
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
    ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter('%.0f'))
    l_55, = ax.plot(ttstamps, imgs_5577[za_idx, :], ls=':', lw=0.65, color='forestgreen')
    m_55, = ax.plot(ttstamps, mds_5577[za_idx, :], ls='-', lw=0.65, color='forestgreen')
    l_63, = ax.plot(ttstamps, imgs_6300[za_idx, :], ls=':', lw=0.65, color='r')
    m_63, = ax.plot(ttstamps, mds_6300[za_idx, :], ls='-', lw=0.65, color='r')
    # ax.plot(ttstamps, imgs_6306[za_idx, :], ls='-', lw=0.65, color='k')
    f_55 = ax.fill_between(ttstamps, imgs_5577[za_idx, :] + 1*stds_5577[za_idx, :],
                           imgs_5577[za_idx, :] - 1*stds_5577[za_idx, :], alpha=0.4, color='forestgreen', edgecolor=None)
    f_63 = ax.fill_between(ttstamps, imgs_6300[za_idx, :] + 1*stds_6300[za_idx, :],
                           imgs_6300[za_idx, :] - 1*stds_6300[za_idx, :], alpha=0.25, color='r', edgecolor=None)
    # f_55 = ax.fill_between(ttstamps, imgs_5577[za_idx, :] + 2*stds_5577[za_idx, :],
    #                        imgs_5577[za_idx, :] - 2*stds_5577[za_idx, :], alpha=0.25, color='forestgreen', edgecolor=None)
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
fig.savefig(f'{KEOGRAMS_DIR}/keo_fit_lowell.pdf', dpi=600, bbox_inches='tight')
plt.show()
# %% All images in one
ax_xlim = []
num_rows = int(np.floor(len(dates) / 2)) * 2  # 2 columns, 2 colors
fig = plt.figure(figsize=(6, 0.8*num_rows), dpi=300)
outer = fig.add_gridspec(1, 2, wspace=0.1, hspace=0, width_ratios=[1, 0.02])
inner = outer[0].subgridspec(num_rows, 2, wspace=0.1, hspace=0.1)
# gspec = GridSpec(num_rows, 3, figure=fig, width_ratios=[
#                  1, 1, 0.05], hspace=0, wspace=0)
axs = []
for i in range(num_rows):
    axs.append([])
    axs[i].append(fig.add_subplot(
        inner[i, 0], sharex=axs[0][0] if i > 0 else None))
    axs[i].append(fig.add_subplot(inner[i, 1], sharex=axs[0]
                  [1] if i > 0 else None, sharey=axs[i][0]))
    fig.add_subplot(axs[i][0])
    fig.add_subplot(axs[i][1])
# axs = np.asarray(axs).flatten().reshape((2, num_rows)).T
cax = fig.add_subplot(outer[1])

axs = np.asarray(axs)
print(axs.shape)

datagaps: dict[int, tuple[Numeric]] = {}

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'axes.titlesize': 10})
matplotlib.rcParams.update({'axes.labelsize': 10})

for fidx, (date, ax) in enumerate(zip(dates, axs)):
    ds = xr.load_dataset(f'{COUNTS_DIR}/hitmis_cts_{date}.nc')
    mds = xr.load_dataset(f'{MODEL_DIR}/keofit_{date}.nc')
    ds = ds.loc[dict(tstamp=mds.tstamp.values)]
    height = sds.height.values
    dheight = np.diff(height).mean()
    print(f'd Height: {np.rad2deg(dheight):.2f}')
    tstamps = ds.tstamp.values
    if (len(tstamps) == 0):
        continue
    imgs_5577 = ds['5577'].values.T[::-1, :] * \
        scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    stds_5577 = ds['5577_std'].values.T[::-1, :] * \
        scale_5577[::-1, None] / dheight * 4*np.pi*1e-6
    imgs_6300 = ds['6300'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    stds_6300 = ds['6300_std'].values.T[::-1, :] * \
        scale_6300[::-1, None] / dheight * 4*np.pi*1e-6
    mds_5577 = mds['5577'].values.T[::-1, :] / dheight * 4*np.pi*1e-6
    mds_6300 = mds['6300'].values.T[::-1, :] / dheight * 4*np.pi*1e-6
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    dds = di.get_indices(tstamps, 'MHJ45')
    _, imgs_5577 = fill_array(imgs_5577, tstamps)
    _, stds_5577 = fill_array(stds_5577, tstamps)
    _, imgs_6300 = fill_array(imgs_6300, tstamps)
    _, stds_6300 = fill_array(stds_6300, tstamps)
    _, mds_5577 = fill_array(mds_5577, tstamps)
    tstamps, mds_6300 = fill_array(mds_6300, tstamps)

    start = tstamps[0].astimezone(pytz.timezone('US/Eastern'))
    start = pd.to_datetime(start).round('1h').to_pydatetime()
    # start = dt.datetime(start.year, start.month, start.day,
    #                     start.hour, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    end = tstamps[-1].astimezone(pytz.timezone('US/Eastern'))
    end = pd.to_datetime(end).round('1h').to_pydatetime()
    print(start, end)
    ttstamps = [(t.timestamp() - start.timestamp()) / 3600 for t in tstamps]
    height_ang = np.rad2deg(height[::-1])
    height_ang -= height_ang[za_idx] - 35

    print(f'Min height: {min(height_ang)}, Max height: {max(height_ang)}')

    tx, hy = np.meshgrid(ttstamps, height_ang)

    nanloc = np.where(np.isnan(imgs_6300[0, :]))[0]
    nanfill = False
    if len(nanloc) > 0 and nanloc[-1] - nanloc[0] > 2:
        print('Too many nans')
        nanfill = True

    def fmt2(x, pos=None):
        return r'${}^\circ$'.format(x)
    
    def fmt3(x, pos=None):
        return ''
    
    ax[0].yaxis.set_major_formatter(fmt2)
    ax[0].locator_params(axis='y', nbins=3)
    # ax[0].set_ylabel('Elevation')
    plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[1].yaxis.set_ticks_position('none')
    if fidx == 0:
        [ax[i].set_title(wl) for i, wl in enumerate(
            ('5577 Å (Green)', '6300 Å (Red)'))]
    if fidx != num_rows - 1:
        ax[0].xaxis.set_ticks_position('none')
        ax[1].xaxis.set_ticks_position('none')
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels(), visible=False)

    im = ax[0].contourf(tx, hy, (imgs_5577 - mds_5577) / np.nanmax(stds_5577[za_idx, :]),
                        aspect='auto',
                        cmap='PiYG_r',
                        levels=np.linspace(-4, 4, 17, endpoint=True),
                        extend='both')

    im = ax[1].contourf(tx, hy, (imgs_6300 - mds_6300) / np.nanpercentile(stds_6300[za_idx, :], 99.9),
                        aspect='auto',
                        cmap='PiYG_r',
                        levels=np.linspace(-4, 4, 17, endpoint=True),
                        extend='both')

    if nanfill:
        # 1. create axis
        tmin = nanloc[0] - 1
        tmax = nanloc[-1] + 1
        trange = np.asarray(ttstamps)[tmin:tmax + 1]
        for axi in ax:
            axi.fill_between(trange, height_ang[0] , height_ang[-1], color='k',
                             alpha=0.2, edgecolor=None, hatch='//')
            axi.text(trange.mean(), np.mean(height_ang),
                     'Unavailable', ha='center', va='center', fontsize=6, rotation='vertical', color='r')
    za_idx = 20
    ax[0].axhline(35, color='k', ls='--', lw=0.5)
    ax[1].axhline(35, color='k', ls='--', lw=0.5)
    ax_xlim.append((end - start).total_seconds() / 3600)
    ax[1].text(1.075, 0.5, start.strftime('%Y-%m-%d'),
               ha='right', va='center', transform=ax[1].transAxes, 
               rotation=90, fontsize=8)

    # yticks = np.asarray(ax[0].get_yticks())
    # yticklabels = list(map(fmt2, yticks))
    # ax[0].set_yticks(yticks, labels=yticklabels, rotation=45)
    # plt.savefig(f'{plotdir}/hitmis_keo_diff_{date}.png', dpi=600)
    # plt.show()

for ax in np.asarray(axs).flatten():
    ax: plt.Axes = ax
    ax.set_xlim(0, max(ax_xlim))
    print(ax.get_ylim())

for idx, ax in enumerate(axs[-1, :]):
    xticks = np.asarray(ax.get_xticks())
    xlen = (len(xticks) // 2)*2
    xticks = xticks[:xlen]
    # if idx == 1:
    #     xticks = xticks[1:]
    xticks = np.round(xticks, decimals=1)
    xticklabels = list(map(lambda x: fmt_time(x, start), xticks))
    ax.set_xticks(xticks, labels=xticklabels, rotation=45, ha='right', va='top')
    # ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlabel("Local Time (UTC$-$05:00)")

def fmt(x, pos=None):
    if 1e-1 < abs(x) < 1e3 or x == 0:
        return f'{x:.1f}'
    sgx = np.sign(x)
    x = np.log10(np.abs(x))
    a, b = np.modf(x)
    b = int(b)
    pexp = fr'10^{{{b:.0f}}}'
    a = int(10**a)
    if a > 1:
        pexp = fr'{a}\times 10^{{{b:.0f}}}'
    if sgx < 0:
        pexp = r'-' + pexp
    elif sgx > 0:
        pexp = r'+' + pexp
    else:
        raise RuntimeError('Should not reach')
    return fr'${pexp}$'

def fmt2(x, pos=None):
    x = int(x + 18)
    return r'${}^\circ$'.format(x)

fig.colorbar(im, cax=cax, shrink=0.5, format=fmt, extend='both')
cax.set_ylabel(r'$\Delta / \sigma$')
fig.text(0.03, 0.5, 'Elevation', va='center', rotation='vertical')

# for ax in axs[:-1, 0]:
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.spines['left'].set_visible(True)
#     ax.spines['bottom'].set_visible(False)

# for ax in axs[:-1, 1]:
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.spines['right'].set_visible(True)
#     ax.spines['bottom'].set_visible(False)
fig.savefig(f'{KEOGRAMS_DIR}/keo_diff_lowell.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
