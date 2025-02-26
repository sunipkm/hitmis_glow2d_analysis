# %% Imports
from __future__ import annotations
import glow2d
from scipy.signal import savgol_filter
import multiprocessing as mp
from ast import Tuple
import sys
from typing import Collection, Dict, Iterable, List, Optional, SupportsFloat as Numeric
import pylab as pl
from tqdm import tqdm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import EARTH_RADIUS
from datetime import datetime
import pytz
from time import perf_counter_ns
from glow2d import glow2d_geo, glow2d_polar, polar_model
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc, rcParams
import matplotlib
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FormatStrFormatter
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pylab as pl
from dateutil.parser import parse
from tzlocal import get_localzone
from pysolar.solar import get_hour_angle
from functools import lru_cache
from matplotlib import dates as mdates

rc('font', **{'family': 'serif', 'serif': ['Times']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

assert(glow2d.__version__ < '4.1')
assert(glow2d.__version__ > '4.0')

EXTENSION = 'pdf'
BBOX='tight'
# %%


def get_minmax(iono: xr.Dataset, feature: str = 'Tn', subfeature: dict = None, minPositive: bool = True) -> Tuple[Numeric, Numeric]:
    if subfeature is None:
        val = iono[feature].values
    else:
        val = iono[feature].loc[subfeature].values
    if minPositive:
        minval = val[np.where(val > 0)].min()
    else:
        minval = val.min()
    return (minval, val.max())


def get_all_minmax(ionos: dict[str, xr.Dataset], feature: str = 'Tn', subfeature: dict = None, minPositive: bool = True) -> Tuple[Numeric, Numeric]:
    minmax = []
    for _, iono in ionos.items():
        minmax.append(get_minmax(iono, feature, subfeature, minPositive))
    minmax = np.asarray(minmax).T
    return minmax[0].min(), minmax[1].max()


def get_all_minmax_list(inp: Iterable, minPositive: bool = True) -> Tuple[Numeric, Numeric]:
    def get_minmax_list(item: Iterable, minPositive: bool = True):
        item = np.asarray(item)
        if minPositive:
            minval = item[np.where(item > 0)].min()
        else:
            minval = item.min()
        return (minval, item.max())
    minmax = []
    for item in inp:
        minmax.append(get_minmax_list(item, minPositive))
    minmax = np.asarray(minmax).T
    return minmax[0].min(), minmax[1].max()
# %% 5577


def plot_geo(bds: xr.Dataset, wl: str, file_suffix: str, *, vmin: float = None, vmax: float = None, decimals: int = 0, num_levels: int = 1000, show: bool = False) -> None:
    ofst = 1000
    scale = 1000
    fig = plt.figure(figsize=(4.8, 3.8), dpi=300, constrained_layout=True)
    gspec = GridSpec(2, 1, hspace=0.02, height_ratios=[1, 25], figure=fig)
    ax = fig.add_subplot(gspec[1, 0], projection='polar')
    cax = fig.add_subplot(gspec[0, 0])
    dtime = parse(bds.time).astimezone(get_localzone())
    _, lon = bds.glatlon
    sza = get_hour_angle(dtime, lon) + 90
    day = dtime.strftime('%Y-%m-%d')
    time_of_day = dtime.strftime('%H:%M hrs')
    # fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300, subplot_kw=dict(projection='polar'), constrained_layout=True, squeeze=True)
    # fig.subplots_adjust(right=0.8)
    # cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('top', size='5%', pad=0.05)
    tn = (bds.ver.loc[dict(wavelength=wl)].values)
    alt = bds.alt_km.values
    ang = bds.angle.values
    r, t = (alt + ofst) / scale, ang  # np.meshgrid((alt + ofst), ang)
    # print(r.shape, t.shape)
    tmin, rmin = glow2d_polar.get_global_coords(np.deg2rad(28), EARTH_RADIUS + 1000)
    tmax, rmax = glow2d_polar.get_global_coords(np.deg2rad(43), EARTH_RADIUS + 1000)
    thor, _ = glow2d_polar.get_global_coords(np.deg2rad(90), EARTH_RADIUS + 1000)
    thor = float(thor.flatten()[0])
    tmin = float(tmin.flatten()[0])
    tmax = float(tmax.flatten()[0])
    # print(np.rad2deg(tmin), np.rad2deg(tmax))
    # , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
    ticks = None
    levels = num_levels
    if vmin is not None and vmax is not None:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), num_levels, endpoint=True).tolist()
        ticks = np.linspace(np.log10(vmin), np.log10(vmax), 10, endpoint=True)
        ticks = np.unique(np.round(ticks, decimals=decimals))
    im = ax.contourf(t, r, np.log10(tn.T), cmap='gist_ncar_r', levels=levels)
    cbar = fig.colorbar(im, cax=cax, shrink=0.6, orientation='horizontal', ticks=ticks)
    if ticks is not None:
        cbar.ax.set_xticklabels([r'$10^{%d}$' % (tval) for tval in ticks])
    cbar.ax.tick_params(labelsize=10)
    # cbar.formatter.set_useMathText(True)
    cbar.set_label(r'%s \AA{} VER ($%s$)' % (wl, bds.ver.attrs['units']), fontsize=12)
    earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    ax.add_artist(earth)
    ax.set_thetamax(ang.max()*180/np.pi)

    ax.scatter(0, 1, s=40, marker=r'$\odot$', facecolors='none', edgecolors='blue', clip_on=False)
    # ax.scatter(np.deg2rad(90), 0.272, s=40, marker=r'$\odot$', facecolors='none', edgecolors='blue', clip_on=False)
    ax.text(np.deg2rad(2), 0.98, 'Observer', fontdict={'size': 10}, horizontalalignment='right')

    # the view cone
    ax.plot([0, tmin], [1, 1 + 1000/scale], ls='--', lw=0.5, color='k', clip_on=True)
    ax.plot([0, tmax], [1, 1 + 1000/scale], ls='--', lw=0.5, color='k', clip_on=True)
    ax.text(np.deg2rad(9), 1 + 300/scale, r'HiT\&MIS FoV', fontsize=10, color='k', rotation=35, ha='center', va='center')

    # the horizon
    ax.plot([0, thor], [1, 1 + 1000/scale], ls='-.', lw=0.5, color='k', clip_on=True)
    ax.text(np.deg2rad(14), 1.05, r'Horizon', fontsize=10, color='k', rotation=78, ha='center', va='center')
    # print(np.rad2deg(thor))
    ax.set_ylim([0, (600 / scale) + 1])

    arrline = np.deg2rad(np.linspace(14, 27, 100, endpoint=True))
    ax.plot(arrline, [(750 / scale) + 1]*len(arrline), lw=0.5, color='k', ls='--', clip_on=False)
    ax.arrow(arrline[-1], (750 / scale) + 1, arrline[-1] - arrline[-2], 0, clip_on=False, head_width=0.008)
    ax.text(np.deg2rad(20), (800 / scale) + 1, 'NE', fontdict={'size': 12})

    # ax.arrow(np.deg2rad(14), (750 / scale) + 1, np.deg2rad(9), 0, clip_on=False, head_width=1)

    locs = ax.get_yticks()

    def get_loc_labels(locs, ofst, scale):
        locs = np.asarray(locs)
        locs = locs[np.where(locs > 1.0)]
        labels = ['O', r'R$_{\mbox{\scriptsize E}}$']
        for loc in locs:
            labels.append('%.0f' % (loc*scale - ofst))
        locs = np.concatenate((np.asarray([0, 1]), locs.copy()))
        labels = labels
        return locs, labels

    locs, labels = get_loc_labels(locs, ofst, scale)
    ax.set_yticks(locs)
    ax.set_yticklabels(labels)

    # label_position=ax.get_rlabel_position()
    ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from Earth center (km)',
            rotation=0, ha='center', va='center', fontdict={'size': 12})
    ax.set_position([0.1, -0.45, 0.8, 2])
    # fig.suptitle('GLOW 2D (Geocentric, %s) %s %s'%(file_suffix.capitalize(), day, time_of_day))
    fig.suptitle('GLOW 2D (Geocentric, %s) %s %s (SEA: %.0f deg)' % (file_suffix.capitalize(), day, time_of_day, sza), fontsize=10)
    # ax.set_rscale('ofst_r_scale')
    # ax.set_rscale('symlog')
    # ax.set_rorigin(-1)
    plt.savefig(f'test_geo_{wl}_{file_suffix}.{EXTENSION}', dpi=600)
    if show:
        plt.show()
    else:
        plt.close(fig)
# %%


def plot_geo_local(bds: xr.Dataset, wl: str, file_suffix: str, *, vmin: float = None, vmax: float = None, decimals: int = 0, num_levels: int = 1000, show: bool = False) -> None:
    dtime = parse(bds.time).astimezone(get_localzone())
    _, lon = bds.glatlon
    # sza = get_altitude(lat, lon, dtime)
    sza = get_hour_angle(dtime, lon) + 90
    day = dtime.strftime('%Y-%m-%d')
    time_of_day = dtime.strftime('%H:%M hrs')
    fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(3, 3.2))
    tn = (bds.ver.loc[dict(wavelength=wl)].values).copy()
    r, t = np.meshgrid((bds.alt_km.values + EARTH_RADIUS), bds.angle.values)
    tt, rr = glow2d_polar.get_local_coords(t, r)
    tt = np.pi / 2 - tt
    vidx = np.where(t < 0)
    tn[vidx] = 0
    ticks = None
    levels = num_levels
    if vmin is not None and vmax is not None:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), num_levels, endpoint=True).tolist()
        ticks = np.linspace(np.log10(vmin), np.log10(vmax), 10, endpoint=True)
        ticks = np.unique(np.round(ticks, decimals=decimals))
    im = ax.contourf(tt, rr, np.log10(tn), 100, cmap='gist_ncar_r', levels=levels)
    cbar = fig.colorbar(im, shrink=0.6, ticks=ticks)
    if ticks is not None:
        cbar.ax.set_yticklabels([r'$10^{%d}$' % (tval) for tval in ticks])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'\begin{center}%s \AA{} VER ($%s$)\end{center}' % (wl, bds.ver.attrs['units']), fontsize=10)
    ax.set_thetamax(90)
    ax.text(np.radians(-20), ax.get_rmax()/2, 'Distance from observation location (km)\n',
            rotation=0, ha='center', va='center')
    ax.text(np.radians(90), ax.get_rmax()*1.02, '(Zenith)',
            rotation=0, ha='center', va='center', fontdict={'size': 8})
    fig.suptitle('GLOW 2D (Local Polar, %s)\n%s %s (SEA: %.0f deg)' %
                 (file_suffix.capitalize(), day, time_of_day, sza), fontsize=10)
    ax.fill_between(np.deg2rad([28, 43]), 0, 10000, alpha=0.3, color='b')
    ax.plot(np.deg2rad([28, 28]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.plot(np.deg2rad([43, 43]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.text(np.deg2rad(50), 1600, r'HiT\&MIS FoV', fontsize=10, color='w', rotation=360-45)
    ax.tick_params(labelsize=10)
    # earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    # ax.add_artist(earth)
    # ax.set_thetamax(ang.max()*180/np.pi)
    ax.set_ylim(rr.min(), rr.max())
    # ax.plot([]) # has to be two-point arrow
    # ax.set_rscale('symlog')
    ax.set_rorigin(-rr.min())
    plt.savefig(f'test_loc_{wl}_{file_suffix}.{EXTENSION}', dpi=600)
    if show:
        plt.show()
    else:
        plt.close(fig)

# %%


def plot_local(iono: xr.Dataset, wl: str, file_suffix: str, *, vmin: float = None, vmax: float = None, decimals: int = 0, num_levels: int = 1000, show: bool = False) -> None:
    dtime = parse(iono.time).astimezone(get_localzone())
    _, lon = iono.glatlon
    # sza = get_altitude(lat, lon, dtime)
    sza = get_hour_angle(dtime, lon) + 90
    day = dtime.strftime('%Y-%m-%d')
    time_of_day = dtime.strftime('%H:%M hrs')
    fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(3, 3.2))
    tn = (iono.ver.loc[dict(wavelength=wl)].values).copy()
    rr, tt = np.meshgrid((iono.r.values), iono.za.values)
    tt = np.pi / 2 - tt
    vidx = np.where(tt < 0)
    tn[vidx] = 0
    ticks = None
    levels = num_levels
    if vmin is not None and vmax is not None:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), num_levels, endpoint=True).tolist()
        ticks = np.linspace(np.log10(vmin), np.log10(vmax), 10, endpoint=True)
        ticks = np.unique(np.round(ticks, decimals=decimals))
    im = ax.contourf(tt, rr, np.log10(tn), cmap='gist_ncar_r', levels=levels)
    cbar = fig.colorbar(im, shrink=0.6, ticks=ticks)
    if ticks is not None:
        cbar.ax.set_yticklabels([r'$10^{%d}$' % (tval) for tval in ticks])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'\begin{center}%s \AA{} VER ($%s$)\end{center}' % (wl, iono.ver.attrs['units']), fontsize=10)
    ax.set_thetamax(90)
    ax.text(np.radians(-20), ax.get_rmax()/2, 'Distance from observation location (km)\nTowards NE',
            rotation=0, ha='center', va='center')
    ax.text(np.radians(90), ax.get_rmax()*1.02, '(Zenith)',
            rotation=0, ha='center', va='center', fontdict={'size': 8})
    fig.suptitle('GLOW 2D (Local Polar, %s)\n%s %s (SEA: %.0f deg)' %
                 (file_suffix.capitalize(), day, time_of_day, sza), fontsize=10)
    ax.fill_between(np.deg2rad([28, 43]), 0, 10000, alpha=0.3, color='b')
    ax.plot(np.deg2rad([28, 28]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.plot(np.deg2rad([43, 43]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.text(np.deg2rad(35), 1600, r'HiT\&MIS FoV', fontsize=10, color='w', rotation=40)
    ax.tick_params(labelsize=10)
    # earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    # ax.add_artist(earth)
    # ax.set_thetamax(ang.max()*180/np.pi)
    ax.set_ylim(rr.min(), rr.max())
    # ax.set_rscale('symlog')
    ax.set_rorigin(-rr.min())
    plt.savefig(f'test_loc_{wl}_uniform_{file_suffix}.{EXTENSION}', dpi=600)
    if show:
        plt.show()
    else:
        plt.close(fig)

# %%


def plot_local_ratio(iono: xr.Dataset, wl: Iterable[str], file_suffix: str, *, colors:Iterable = None, show: bool = False):
    if not isinstance(wl, Iterable):
        raise ValueError('wl should be an iterable')
    elif isinstance(wl, str):
        raise ValueError('wl should be an iterable of strings')
    elif len(wl) < 2:
        raise ValueError('wl should have at least two elements')
    if colors is not None and len(colors) != len(wl):
        raise ValueError('colors should have the same length as wl')
    dtime = parse(iono.time).astimezone(get_localzone())
    _, lon = iono.glatlon
    # sza = get_altitude(lat, lon, dtime)
    sza = get_hour_angle(dtime, lon) + 90
    day = dtime.strftime('%Y-%m-%d')
    time_of_day = dtime.strftime('%H:%M hrs')
    fig, ax = plt.subplots(dpi=300, figsize=(2.2, 3.4))
    fig.suptitle('GLOW 2D (Local Polar, %s)\n%s %s (SEA: %.0f deg)' %
                 (file_suffix.capitalize(), day, time_of_day, sza), fontsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    za_min = 90 - np.arange(22, 71, 2, dtype=float)
    za_max = za_min + 2
    za = (za_min + za_max) / 2
    za_min = np.deg2rad(za_min)
    za_max = np.deg2rad(za_max)
    
    for idx, w in enumerate(wl):
        em = glow2d_polar.get_emission(iono, feature=w, za_min=za_min, za_max=za_max)
        em *= 4*np.pi*1e-6  # convert to Rayleigh
        if colors is not None:
            ax.plot(em, 90 - za, lw=0.75, label=fr'{w} \AA{{}}', color=colors[idx])
        else:
            ax.plot(em, 90 - za, lw=0.75, label=fr'{w} \AA{{}}')

    ax.set_xscale('log')
    ax.set_ylim(za.min(), za.max())
    ax.set_ylabel('Elevation Angle (deg)')
    ax.set_xlabel(fr'Intensity (R)')
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    fig.savefig(f'test_loc_intensity_{file_suffix}.{EXTENSION}', dpi=600, bbox_inches=BBOX)
    if show:
        plt.show()
    else:
        plt.close(fig)
# %% Brightness comparison Function


def plot_brightness(tdict, num_pts: int, show: bool = False, mpool=None) -> None:
    wls = ('5577', '6300')
    za_min = np.arange(0, 90, 2.5, dtype=float)
    za_max = za_min + 2.5
    za = za_min + 1.25
    za_min = np.deg2rad(za_min)
    za_max = np.deg2rad(za_max)
    day: str = ''
    time_of_day: Dict[str, str] = {}
    ionos: Dict[str, xr.Dataset] = {}
    photonrate: Dict[str, Dict[str, np.ndarray]] = {}
    photonrate_a: list[np.ndarray] = []
    factor = np.deg2rad(0.1)  # 0.1 deg equivalent for azimuth
    for key, time in tdict.items():
        iono = polar_model(time, 42.64981361744372, -71.31681056737486, 40, 0, n_pts=num_pts, mpool=mpool)
        dtime = parse(iono.time).astimezone(get_localzone())
        day = dtime.strftime('%Y-%m-%d')
        time_of_day[key] = dtime.strftime('%H:%M hrs')
        ionos[key] = iono
    for key in tdict.keys():
        photonrate[key] = {}
        for wl in wls:
            em = glow2d_polar.get_emission(ionos[key], feature=wl, za_min=za_min, za_max=za_max)
            photonrate[key][wl] = em*factor
            photonrate_a.append(em.copy()*factor)
    vmin, vmax = get_all_minmax_list(photonrate_a)  # xmin, xmax
    # vmin = 10**(np.round(np.log10(vmin)) - 1)
    vmax = 10**(np.round(np.log10(vmax)) + 1)

    fig, ax = plt.subplots(dpi=300, figsize=(6.4, 3.6))
    colors = {'dawn': 'k', 'noon': 'r', 'dusk': 'b', 'midnight': 'g'}
    lstyles = {'5577': '-.', '6300': '-'}
    ax.set_title(r'GLOW Intensity on %s' % (day))
    ax.set_xscale('log')
    for key in tdict.keys():  # wish python 3.9 had switch-case
        for wl in wls:
            ax.plot(photonrate[key][wl], za[::-1], ls=lstyles[wl], color=colors[key], lw=0.75)
    # ax.plot(1e4/np.cos(np.deg2rad(90 - za)), za, ls = '--', color='orange', lw=0.75)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(za.min(), za.max())
    ax.set_ylabel('Elevation Angle (deg)')
    ax.set_xlabel(r'Intensity ($cm^{-2} s^{-1}$)')
    ax.text(1e9*factor, 80, 'Dawn', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.text(1e9*factor, 75, 'Noon', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.text(1e9*factor, 70, 'Dusk', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.text(1e9*factor, 65, 'Midnight', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.plot([1.1e9*factor, 2966479394.84*factor], [80, 80], ls='-', color=colors['dawn'], lw=0.75)
    ax.plot([2966479394.84*factor, 8e9*factor], [80, 80], ls='-.', color=colors['dawn'], lw=0.75)
    ax.plot([1.1e9*factor, 2966479394.84*factor], [75, 75], ls='-', color=colors['noon'], lw=0.75)
    ax.plot([2966479394.84*factor, 8e9*factor], [75, 75], ls='-.', color=colors['noon'], lw=0.75)
    ax.plot([1.1e9*factor, 2966479394.84*factor], [70, 70], ls='-', color=colors['dusk'], lw=0.75)
    ax.plot([2966479394.84*factor, 8e9*factor], [70, 70], ls='-.', color=colors['dusk'], lw=0.75)
    ax.plot([1.1e9*factor, 2966479394.84*factor], [65, 65], ls='-', color=colors['midnight'], lw=0.75)
    ax.plot([2966479394.84*factor, 8e9*factor], [65, 65], ls='-.', color=colors['midnight'], lw=0.75)

    # ax.text(1e9, 40, r'$\csc{\theta}$ \\ (midnight)', horizontalalignment='right', verticalalignment='center')
    # ax.plot([1.1e9, 2966479394.84], [40]*2, ls='--', color='orange', lw=0.75)

    ax.text(1e9*factor, 50, r'5577 \AA{}', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.plot([1.1e9*factor, 2966479394.84*factor], [50]*2, ls='-.', color=colors['dawn'], lw=0.75)
    ax.text(1e9*factor, 55, r'6300 \AA{}', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.plot([1.1e9*factor, 2966479394.84*factor], [55]*2, ls='-', color=colors['dawn'], lw=0.75)
    fig.savefig(f'test_prate_{num_pts}.{EXTENSION}', dpi=600, bbox_inches='tight')
    if show:
        fig.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(dpi=300, figsize=(6, 3.4))
    colors = {'dawn': 'k', 'noon': 'r', 'dusk': 'b', 'midnight': 'g'}
    lstyles = {'5577': '-.', '6300': '-'}
    ax.set_title(r'GLOW Intensity Ratio for 5577 \AA{} and 6300 \AA{} on %s' % (day))
    ax.set_xscale('log')
    for key in tdict.keys():  # wish python 3.9 had switch-case
        mval = np.median(photonrate[key]['5577'] / photonrate[key]['6300'])
        if key == 'noon':
            ax.text(10**(np.log10(mval) - 0.015), 15, f'{mval:.2f}', fontdict={'size': 10},
                    horizontalalignment='right', verticalalignment='center', rotation=0, color=colors[key])
        else:
            ax.text(mval*10**0.09, 15, f'{mval:.2f}', fontdict={'size': 10},
                    horizontalalignment='right', verticalalignment='center', rotation=0, color=colors[key])
        ax.axvline(mval, ls=':', color=colors[key], lw=0.65, label='Median Values' if key == 'dawn' else None)
        ax.plot(photonrate[key]['5577'] / photonrate[key]['6300'], za[::-1],
                ls=lstyles['6300'], color=colors[key], lw=0.75, label=f'{key.capitalize()}')
    ax.set_ylim(za.min(), za.max())
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    b_anchor = (0.5, (20 - ymin)/(ymax - ymin))
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=b_anchor)
    # ax.plot(1e4/np.cos(np.deg2rad(90 - za)), za, ls = '--', color='orange', lw=0.75)
    # ax.set_xlim(None, 20)
    ax.set_ylabel('Elevation Angle (deg)')
    ax.set_xlabel(r'Intensity Ratio (dimensionless)')
    # ax.text(1e9, 80, 'Dawn', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    # ax.text(1e9, 75, 'Noon', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    # ax.text(1e9, 70, 'Dusk', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    # ax.text(1e9, 65, 'Midnight', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    # ax.plot([1.1e9, 2966479394.84], [80, 80], ls = '-', color = colors['dawn'], lw = 0.75)
    # ax.plot([2966479394.84, 8e9], [80, 80], ls = '-.', color = colors['dawn'], lw = 0.75)
    # ax.plot([1.1e9, 2966479394.84], [75, 75], ls = '-', color = colors['noon'], lw = 0.75)
    # ax.plot([2966479394.84, 8e9], [75, 75], ls = '-.', color = colors['noon'], lw = 0.75)
    # ax.plot([1.1e9, 2966479394.84], [70, 70], ls = '-', color = colors['dusk'], lw = 0.75)
    # ax.plot([2966479394.84, 8e9], [70, 70], ls = '-.', color = colors['dusk'], lw = 0.75)
    # ax.plot([1.1e9, 2966479394.84], [65, 65], ls = '-', color = colors['midnight'], lw = 0.75)
    # ax.plot([2966479394.84, 8e9], [65, 65], ls = '-.', color = colors['midnight'], lw = 0.75)

    # ax.text(1e9, 40, r'$\csc{\theta}$ \\ (midnight)', horizontalalignment='right', verticalalignment='center')
    # ax.plot([1.1e9, 2966479394.84], [40]*2, ls='--', color='orange', lw=0.75)

    # ax.text(1e9, 50, r'5577 \AA{}', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    # ax.plot([1.1e9, 2966479394.84], [50]*2, ls = '-.', color = colors['dawn'], lw = 0.75)
    # ax.text(1e9, 55, r'6300 \AA{}', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    # ax.plot([1.1e9, 2966479394.84], [55]*2, ls = '-', color = colors['dawn'], lw = 0.75)
    fig.savefig(f'test_pratio_{num_pts}.{EXTENSION}', dpi=600, bbox_inches='tight')
    if show:
        fig.show()
    else:
        plt.close(fig)
# %%


@lru_cache(maxsize=None)
def eval_model(t0: datetime, t1: datetime, num: int, npts: int = 100, mpool=None) -> List[xr.Dataset]:
    assert num > 1
    times = np.linspace(t0.timestamp(), t1.timestamp(), num, endpoint=True)
    times = [datetime.fromtimestamp(t) for t in times]
    ionos = []
    for time in tqdm(times):
        iono = polar_model(time, 42.64981361744372, -71.31681056737486, 40, 0, n_pts=npts, show_progress=False, mpool=mpool)
        ionos.append(iono)
    return times, ionos
# %%


def plot_ratio(num_pts: int, show: bool = False, mpool=None, res=None) -> List[xr.Dataset]:
    za_min = np.arange(10, 80, 0.25, dtype=float)
    za_max = za_min + 0.25
    za = (za_min + za_max) / 2
    za_min = np.deg2rad(za_min)
    za_max = np.deg2rad(za_max)
    t0 = datetime(2022, 2, 15, 19, 0)
    t1 = datetime(2022, 2, 16, 3, 0)
    day: str = f'{t0:%Y-%m-%d}'
    photonrat: List[np.ndarray] = []
    factor = np.deg2rad(0.1)  # 0.1 deg equivalent for azimuth
    if res is None:
        times, ionos = eval_model(t0, t1, 133, npts=num_pts, mpool=mpool)
        for idx, _ in tqdm(enumerate(times)):
            iono = ionos[idx]
            em5577 = glow2d_polar.get_emission(iono, feature='5577', za_min=za_min, za_max=za_max)
            em6300 = glow2d_polar.get_emission(iono, feature='6300', za_min=za_min, za_max=za_max)
            ratio = em5577 / em6300
            photonrat.append(ratio)
        vmin, vmax = get_all_minmax_list(photonrat)  # xmin, xmax
        ds = xr.Dataset({'ratio': (('time', 'za'), photonrat)}, coords={'time': times, 'za': za})
        vmax = 10**(np.round(np.log10(vmax)) + 1)
    else:
        times, ionos, ds = res
        t0 = times[0]
        t1 = times[-1]
    # vmin = 10**(np.round(np.log10(vmin)) - 1)
    fig, ax = plt.subplots(dpi=300, figsize=(6.4, 3.8))
    ax.set_title(r'GLOW Intensity Ratio on %s' % (day))
    # ax.plot(1e4/np.cos(np.deg2rad(90 - za)), za, ls = '--', color='orange', lw=0.75)
    ds['ratio'].T.plot(ax=ax, cmap='bone', cbar_kwargs={'aspect': 100})
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax.text(1e9, 40, r'$\csc{\theta}$ \\ (midnight)', horizontalalignment='right', verticalalignment='center')
    # ax.plot([1.1e9, 2966479394.84], [40]*2, ls='--', color='orange', lw=0.75)
    fig.show()
    return (times, ionos, ds)

# %% Plot point distribution
# Generate the coordinate transform point distribution


def generate_pt_distrib(show: bool = True):
    ttext = ('', '', '', '', '')
    rtext = ('1', '2', '3', '4',)  # '5')
    markers = ('o', 's', 'D', 'p',)  # 'H')

    ofst = 1000
    scale = 1000
    fig = plt.figure(figsize=(3.2, 2.4), dpi=300, constrained_layout=True)
    gspec = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gspec[0, 0], projection='polar')
    alt = np.linspace(60, 550, len(rtext))
    ang = np.linspace(np.deg2rad(2.5), np.deg2rad(27.5), len(markers))  # np.arccos(EARTH_RADIUS/(EARTH_RADIUS + 1000)), 5)
    r = (alt + ofst) / scale
    # , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
    earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    ax.add_artist(earth)
    ax.set_thetamax(np.rad2deg(np.arccos(EARTH_RADIUS/(EARTH_RADIUS + 1000))))

    thor, _ = glow2d_polar.get_global_coords(np.deg2rad(58), EARTH_RADIUS + 1000)
    thor = float(thor.flatten()[0])

    cmap = matplotlib.cm.get_cmap('rainbow')

    for tidx, t in enumerate(ang):
        for ridx, dist in enumerate(r):
            # col = cmap(1 - ((alt[ridx] - alt.min()) / alt.max()))
            col = 'k'
            p, _ = glow2d_polar.get_local_coords(t, alt[ridx] + EARTH_RADIUS)
            p = np.pi/2 - p
            ax.scatter(t, dist, s=80 if tidx < 3 else 120,
                       marker=markers[tidx],
                       facecolors='w' if p > 0 else col,
                       edgecolors=col, clip_on=False)
            ax.annotate(ttext[tidx] + rtext[ridx], xy=(t - np.deg2rad(0.25), dist),
                        color='k' if p > 0 else 'w',
                        weight='heavy', horizontalalignment='center', verticalalignment='center', fontsize=8)

    ax.set_ylim([0, (600 / scale) + 1])
    locs = ax.get_yticks()

    def get_loc_labels(locs, ofst, scale):
        locs = np.asarray(locs)
        locs = locs[np.where(locs > 1.0)]
        labels = ['O', r'R$_{\mbox{\scriptsize E}}$']
        for loc in locs:
            labels.append('%.0f' % (loc*scale - ofst))
        locs = np.concatenate((np.asarray([0, 1]), locs.copy()))
        labels = labels
        return locs, labels

    locs, labels = get_loc_labels(locs, ofst, scale)
    ax.set_yticks(locs)
    ax.set_yticklabels(labels)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)

    ax.plot([0, thor], [1, 1 + 1000/scale], ls='-.', lw=0.5, color='k', clip_on=True, alpha=0.75)

    # label_position=ax.get_rlabel_position()
    ax.text(np.radians(-20), ax.get_rmax()/2, 'Distance from Earth center (km)',
            rotation=0, ha='center', va='center')
    ax.set_position([0.1, -0.45, 0.8, 2])
    fig.suptitle('Distribution of points in geocentric coordinates')
    # fig.suptitle('GLOW 2D Output (2D, geocentric) %s %s'%(day, time_of_day))
    # ax.set_rscale('ofst_r_scale')
    # ax.set_rscale('symlog')
    # ax.set_rorigin(-1)
    plt.savefig(f'pt_distrib_geo.{EXTENSION}', dpi=600, bbox_inches=BBOX)
    if show:
        plt.show()
    else:
        plt.close(fig)
    # Generate the coordinate transform point distribution
    from matplotlib import ticker
    fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(3.2, 3.6))

    r, t = np.meshgrid(alt + EARTH_RADIUS, ang)
    t, r = glow2d_polar.get_local_coords(t, r)
    ax.set_ylim(60, r.max())
    ax.text(np.radians(90), r.max()*1.02, '(Zenith)',
            rotation=0, ha='center', va='center', fontdict={'size': 8})
    ax.text(np.radians(-16), ax.get_rmax()/2, 'Distance from observation location (km)',
            rotation=0, ha='center', va='center')
    ax.fill_between(np.deg2rad([22, 72]), 0, 10000, alpha=0.3, color='b')
    ax.plot(np.deg2rad([22, 22]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.plot(np.deg2rad([72, 72]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.text(np.deg2rad(35), 1600, r'HiT\&MIS FoV', fontsize=10, color='w', rotation=40)

    for tidx, t in enumerate(ang):
        for ridx, dist in enumerate(alt):
            # col = cmap(1 - ((dist - alt.min()) / alt.max()))
            col = 'k'
            p, r = glow2d_polar.get_local_coords(t, dist + EARTH_RADIUS)
            p = np.pi/2 - p
            ax.scatter(p, r, s=80 if tidx < 3 else 120,
                       marker=markers[tidx],
                       facecolors='w' if p > 0 else col,
                       edgecolors=col, clip_on=True)
            ax.annotate(ttext[tidx] + rtext[ridx], xy=(p, r),
                        color='k' if p > 0 else 'w',
                        weight='heavy', horizontalalignment='center', verticalalignment='center', fontsize=8)

    # np.meshgrid((alt + ofst) / ofst, ang)
    ax.tick_params(labelsize=10)
    # earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    # ax.add_artist(earth)
    ax.set_thetamax(90)
    # ax.set_rscale('symlog')
    ax.set_rorigin(-60)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    fig.suptitle('Distribution of points in local polar coordinates')

    plt.savefig(f'pt_distrib_local.{EXTENSION}', dpi=600, bbox_inches=BBOX)
    if show:
        plt.show()
    else:
        plt.close(fig)

# %% Run model for parallel


def run_model(file_suffix: str, time: datetime, pos: int) -> Tuple[str, xr.Dataset, xr.Dataset]:
    lat, lon = 42.64981361744372, -71.31681056737486
    currrent = mp.current_process()
    tid = currrent._identity[0]
    grobj = glow2d_geo(time, lat, lon, 40, n_pts=100, show_progress=True, tqdm_kwargs={
                       'position': tid, 'desc': f'{pos}: {file_suffix.capitalize()}'})
    bds = grobj.run_model()
    grobj = glow2d_polar(bds, resamp=2)
    iono = grobj.transform_coord()
    return (file_suffix, bds, iono)
# %% Main function


def main(serial: bool = False, show: bool = False, plot_ratio: Optional[Collection] = None, bdss={}, ionos={}, pt_distrib: bool=False) -> None:
    """## Main function to run the GLOW 2D model

    ### Args:
        - `serial (bool, optional)`: 2-D Model level parallelism. Defaults to False.
        - `show (bool, optional)`: Show plots. Defaults to False.
        - `pt_distrib (bool, optional)`: Calculate point distribution. Defaults to False.
    """
    tdict = {
        'dawn': datetime(2022, 3, 22, 6, 10).astimezone(pytz.utc),
        'noon': datetime(2022, 3, 22, 12, 0).astimezone(pytz.utc),
        'dusk': datetime(2022, 3, 22, 18, 50).astimezone(pytz.utc),
        'midnight': datetime(2022, 3, 22, 23, 59).astimezone(pytz.utc)
    }

    lat, lon = 42.64981361744372, -71.31681056737486

    if len(bdss) + len(ionos) != 2*len(tdict):
        # Serial
        if serial:
            sts = perf_counter_ns()
            n_proc = mp.cpu_count()
            for file_suffix, time in tdict.items():
                st = perf_counter_ns()
                with mp.Pool(processes=n_proc) as pool: 
                    grobj = glow2d_geo(time, lat, lon, 40, n_pts=100, mpool=pool, show_progress=True)
                    bds = grobj.run_model()
                end = perf_counter_ns()
                print(f'Time to generate : {(end - st)*1e-6: 8.6f} ms')
                st = perf_counter_ns()
                grobj = glow2d_polar(bds, resamp=2)
                iono = grobj.transform_coord()
                end = perf_counter_ns()
                print(f'Time to convert  : {(end - st)*1e-6: 8.6f} ms')
                bdss[file_suffix] = bds
                ionos[file_suffix] = iono

            ste = perf_counter_ns()

            print(f'Total time to generate (serial): {(ste - sts)*1e-6: 8.6f} ms')
        # Parallel
        else:
            fsfx = []
            ftimes = []
            for file_suffix, time in tdict.items():
                fsfx.append(file_suffix)
                ftimes.append(time)

            sts = perf_counter_ns()
            n_proc = 4
            n_items = len(fsfx)
            with mp.Pool(processes=n_proc) as pool:
                results = pool.starmap(run_model, zip(fsfx, ftimes, range(n_items)))
                for result in results:
                    bdss[result[0]] = result[1]
                    ionos[result[0]] = result[2]
            ste = perf_counter_ns()

            print(f'Total time to generate (parallel): {(ste - sts)*1e-6: 8.6f} ms')
    # Generate the time plots
    for file_suffix in bdss:
        bds = bdss[file_suffix]
        iono = ionos[file_suffix]
        for wl in ('5577', '6300'):
            bds_minmax = get_all_minmax(bdss, 'ver', {'wavelength': wl}, True)
            iono_minmax = get_all_minmax(ionos, 'ver', {'wavelength': wl}, True)
            print(f'[{file_suffix}][{wl}]: ', end='')
            sys.stdout.flush()
            plot_geo(bds, wl, file_suffix, vmin=1e-4, vmax=1e3, show=show)
            print('GEO', end=' ')
            sys.stdout.flush()
            plot_geo_local(bds, wl, file_suffix, vmin=1e-4, vmax=1e3, show=show)
            print('GEO-LOC', end=' ')
            plot_local(iono, wl, file_suffix, vmin=1e-4, vmax=1e3, show=show)
            print('LOC')
        plot_local_ratio(iono, ('5577', '6300'), file_suffix, show=show, colors=('forestgreen', 'r'))
        print('Plotted local ratios.')
    
    if pt_distrib:
        # Generate the coordinate transform point distribution
        from matplotlib import ticker
        fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(6.4, 4.8))
        # np.meshgrid((alt + ofst) / ofst, ang)
        r, t = iono.r.values, iono.za.values
        # print(r.shape, t.shape)
        r, t = np.meshgrid(r, t)
        grobj = glow2d_geo(tdict['dawn'], lat, lon, 40, n_pts=100)
        bds = grobj.run_model()
        grobj = glow2d_polar(bds, resamp=2)
        tt, rr = grobj.get_global_coords(t, r)
        gd2 = grobj.get_jacobian_glob2loc_glob(rr, tt)
        t = np.pi / 2 - t
        # , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
        ticks = None
        levels = int(1e3)
        vmin, vmax = gd2.min(), gd2.max()
        if vmin is not None and vmax is not None:
            levels = np.linspace(np.log10(vmin), np.log10(vmax), levels, endpoint=True).tolist()
            ticks = np.arange(np.round(np.log10(vmin) + 0.1, decimals=1), np.round(np.log10(vmax), decimals=1), 0.5)
        im = ax.contourf(t, r, np.log10(gd2), levels=levels, cmap='gist_ncar_r')
        cbar = fig.colorbar(im, shrink=0.6, ticks=ticks)
        cbar.set_label('Area Scale', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        if ticks is not None:
            cbar.ax.set_yticklabels([r'$10^{%.1f}$' % (tval) for tval in ticks])
        ax.set_thetamax(90)
        ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from observation location (km)',
                rotation=0, ha='center', va='center')
        fig.suptitle('Area element scaling from geocentric to local polar coordinates')
        ax.fill_between(np.deg2rad([22, 72]), 0, 10000, alpha=0.3, color='b')
        ax.plot(np.deg2rad([22, 22]), [0, 10000], lw=0.5, color='k', ls='--')
        ax.plot(np.deg2rad([72, 72]), [0, 10000], lw=0.5, color='k', ls='--')
        ax.text(np.deg2rad(35), 1600, r'HiT\&MIS FoV', fontsize=10, color='w', rotation=40)
        ax.tick_params(labelsize=10)
        ax.text(np.radians(90), r.max()*1.02, '(Zenith)',
                rotation=0, ha='center', va='center', fontdict={'size': 8})
        # earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
        # ax.add_artist(earth)
        # ax.set_thetamax(ang.max()*180/np.pi)
        ax.set_ylim(r.min(), r.max())
        # ax.set_rscale('symlog')
        ax.set_rorigin(-60)
        plt.savefig(f'test_loc_geo_distrib.{EXTENSION}', dpi=600)
        if show:
            plt.show()
        else:
            plt.close(fig)
        print('Plotted point distribution')
    # Plot the brightness comparisons for different number of points
    if plot_ratio is not None and len(plot_ratio) > 0:
        mpool = mp.Pool(processes=mp.cpu_count())
        print('Plotting brightness comparisons for different number of points')
        for num_pts in tqdm(plot_ratio):
            plot_brightness(tdict, num_pts, show=show, mpool=mpool)


# %%
res = None
# %%
if __name__ == '__main__':
    main(serial=False, show=False, plot_ratio=(100,))
    # generate_pt_distrib(show=True)
    # res = plot_ratio(100, mpool=mp.Pool(mp.cpu_count()), show=True, res = res)

# %%
