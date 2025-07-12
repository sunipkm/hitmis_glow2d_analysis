# %% Imports
from __future__ import annotations
from common_funcs import fill_array, geocent_to_geodet, make_color_axis
from settings import Directories, is_interactive_session
import datetime as dt
from typing import Tuple, SupportsFloat as Numeric, Iterable
from skmpython import staticvars
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pytz
import digisondeindices as di
from matplotlib import ticker
import matplotlib
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

dirs = Directories()
COUNTS_DIR = dirs.counts_dir
MODEL_DIR = dirs.model_dir
KEOGRAMS_DIR = dirs.keograms_dir

# %%
sds = xr.load_dataset('keo_scale.nc')
scale_5577 = sds['5577'].values[::-1]
scale_6300 = sds['6300'].values[::-1]
za_min = sds['za_min'].values
za_max = sds['za_max'].values
# %%
# dates = ['20220218']
# %%


def fmt_time(x: Numeric, ofst: dt.datetime) -> str:
    x = dt.timedelta(hours=x) # type: ignore
    res = ofst + x # type: ignore
    return res.strftime('%H:%M')


# %% Dates
filter = True
dates = list(COUNTS_DIR.glob('hitmis_cts_*.nc')) # glob.glob(COUNTS_DIR / f'hitmis_cts_*.nc')
dates = list(map(lambda x: x.name.split('_')[-1].split('.')[0], dates))
if filter:
    dates_ = list(MODEL_DIR.glob('fwdmodel_*.nc'))
    dates_ = list(map(lambda x: x.name.split('_')[-1].split('.')[0], dates_))
    dates = list(set(dates).intersection(dates_))
dates.sort()
# %% Keogram
za_idx = 20
for fidx, date in enumerate(dates):
    ds = xr.load_dataset(COUNTS_DIR / f'hitmis_cts_{date}.nc')
    if filter:
        mds = xr.load_dataset(MODEL_DIR / f'fwdmodel_{date}.nc')
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
    _, imgs_5577, _ = fill_array(imgs_5577, tstamps) # type: ignore
    _, stds_5577, _ = fill_array(stds_5577, tstamps) # type: ignore
    _, imgs_6300, _ = fill_array(imgs_6300, tstamps) # type: ignore
    _, stds_6300, _ = fill_array(stds_6300, tstamps) # type: ignore
    _, imgs_6306, _ = fill_array(imgs_6306, tstamps) # type: ignore
    tstamps, stds_6306, _ = fill_array(stds_6306, tstamps) # type: ignore
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
    ax[0].yaxis.set_major_formatter(fmt2) # type: ignore
    ax[0].locator_params(axis='y', nbins=7) # type: ignore
    ax[1].yaxis.set_major_formatter(fmt2) # type: ignore
    ax[1].locator_params(axis='y', nbins=7) # type: ignore
    ax[2].yaxis.set_major_formatter(fmt2) # type: ignore
    ax[2].locator_params(axis='y', nbins=7) # type: ignore
    for axs in ax: # type: ignore
        axs.set_ylabel('Elevation')
    [ax[i].set_title(wl) for i, wl in enumerate(('5577 Å (Green)', '6300 Å (Red)', '6306 Å (Cloud Indicator)'))] # type: ignore
    im = ax[0].pcolormesh(tx, hy, np.log10(imgs_5577), cmap='Greens')#, vmin=1.5, vmax=4) # type: ignore
    # im = ax[0].imshow(np.log10(imgs_5577), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone') #, vmin=1.5, vmax=4)
    cbar = fig.colorbar(im, cax=cax[0], shrink=0.5, format=fmt) # type: ignore
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Intensity (R)', fontsize=8)
    im = ax[1].pcolormesh(tx, hy, np.log10(imgs_6300), cmap='Reds')#, vmin=1.5, vmax=4) # type: ignore
    # im = ax[1].imshow(np.log10(imgs_6300), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone') #, vmin=1.5, vmax=4)
    cbar = fig.colorbar(im, cax=cax[1], shrink=0.5, format=fmt) # type: ignore
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Intensity (R)', fontsize=8)
    im = ax[2].pcolormesh(tx, hy, np.log10(imgs_6306), cmap='bone', vmin=np.nanpercentile(np.log10(imgs_6306), 1), vmax=np.nanpercentile(np.log10(imgs_6306), 99)) # type: ignore
    # im = ax[2].imshow(np.log10(imgs_6306), aspect='auto', extent=(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, np.rad2deg(height[0]), np.rad2deg(height[-1])), cmap='bone', vmin=np.nanpercentile(np.log10(imgs_6306), 1), vmax=np.nanpercentile(np.log10(imgs_6306), 99))
    cbar=fig.colorbar(im, cax=cax[2], shrink=0.5, format=fmt) # type: ignore
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Intensity (R)', fontsize=8)
    ax[-1].set_xlim(0, 9) # type: ignore
    xticks = np.arange(10).astype(float)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    ax[-1].set_xticklabels(xticks) # type: ignore
    ax[-1].set_xlabel("Local Time") # type: ignore
    if nanfill:
        # 1. create axis
        trange = np.linspace(0, (tstamps[-1] - tstamps[0]).total_seconds()/3600, len(imgs_6300[0, :]), endpoint=True)
        tmin = nanloc[0] - 1
        tmax = nanloc[-1] + 1
        trange = trange[tmin:tmax + 1]
        # 2. Find nan locs
        ax[0].text((trange[-1] + trange[0])*0.5, np.mean(height_ang), 'Unavailable', ha='center', va='center', fontsize=8, rotation='vertical', color='r') # type: ignore
        ax[1].text((trange[-1] + trange[0])*0.5, np.mean(height_ang), 'Unavailable', ha='center', va='center', fontsize=8, rotation='vertical', color='r') # type: ignore
        ax[2].text((trange[-1] + trange[0])*0.5, np.mean(height_ang), 'Unavailable', ha='center', va='center', fontsize=8, rotation='vertical', color='r') # type: ignore
    plt.savefig(KEOGRAMS_DIR / f'hitmis_keo_{date}.pdf')
    if is_interactive_session():
        plt.show()
    else:
        plt.close(fig)

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
    ax: plt.Axes = ax # type: ignore
    ds = xr.load_dataset(COUNTS_DIR / f'hitmis_cts_{date}.nc')
    mds = xr.load_dataset(MODEL_DIR / f'fwdmodel_{date}.nc')
    ds = ds.loc[dict(tstamp=mds.tstamp.values)]
    tstamps = ds.tstamp.values
    lat, lon = 42.64981361744372, -71.31681056737486
    if (len(tstamps) == 0):
        continue
    # tecsrc = get_gps_tec(tstamps.astype(int)*1e-9, [lat], [lon], [0])
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
    _, imgs_5577, _ = fill_array(imgs_5577, tstamps) # type: ignore
    _, stds_5577, _ = fill_array(stds_5577, tstamps) # type: ignore
    _, imgs_6300, _ = fill_array(imgs_6300, tstamps) # type: ignore
    _, stds_6300, _ = fill_array(stds_6300, tstamps) # type: ignore
    _, imgs_6306, _ = fill_array(imgs_6306, tstamps) # type: ignore
    _, stds_6306, _ = fill_array(stds_6306, tstamps) # type: ignore
    _, mds_5577, _ = fill_array(mds_5577, tstamps) # type: ignore
    _, mds_ap, _ = fill_array(mds_ap[:, None], tstamps, axis=0) # type: ignore
    tstamps, mds_6300, _ = fill_array(mds_6300, tstamps) # type: ignore
    # _, mds_ap, _, _, _ = get_smoothed_geomag(tstamps)

    start = tstamps[0].astimezone(pytz.timezone('US/Eastern'))
    start = pd.to_datetime(start).round('1h').to_pydatetime()
    # start = dt.datetime(start.year, start.month, start.day,
    #                     start.hour, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    end = tstamps[-1].astimezone(pytz.timezone('US/Eastern'))
    end = pd.to_datetime(end).round('1h').to_pydatetime()
    ttstamps = [(t.timestamp() - start.timestamp()) / 3600 for t in tstamps]
    # gps_tstamp = tecsrc.timestamps.values.copy()
    # gps_tstamp = [pd.to_datetime(t*1e9).to_pydatetime() for t in gps_tstamp]
    # gps_tstamp = [(t.timestamp() - start.timestamp()) /
    #               3600 for t in gps_tstamp]
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
    ax.text(v[0], np.mean(ylim), 'Data Unavailable', ha='center', # type: ignore
            va='top', fontsize=8, color='r', rotation='vertical')

for ax in axes.flatten()[-2:]:
    xticks = np.asarray(ax.get_xticks())
    xticks = np.round(xticks, decimals=1)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlabel("Local Time (UTC$-$05:00)")
fig.savefig(KEOGRAMS_DIR / 'fwdmodel_lowell.pdf',
            dpi=600, bbox_inches='tight')
if is_interactive_session():
    plt.show()
else:
    plt.close(fig)
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
    ax: plt.Axes = ax # type: ignore
    ds = xr.load_dataset(COUNTS_DIR / f'hitmis_cts_{date}.nc')
    mds = xr.load_dataset(MODEL_DIR / f'fwdmodel_{date}.nc')
    ds = ds.loc[dict(tstamp=mds.tstamp.values)]
    tstamps = ds.tstamp.values
    lat, lon = 42.64981361744372, -71.31681056737486
    if (len(tstamps) == 0):
        continue
    # tecsrc = get_gps_tec(tstamps.astype(int)*1e-9, [lat], [lon], [0])
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
    _, imgs_5577, _ = fill_array(imgs_5577, tstamps) # type: ignore
    _, stds_5577, _ = fill_array(stds_5577, tstamps) # type: ignore
    _, imgs_6300, _ = fill_array(imgs_6300, tstamps) # type: ignore
    _, stds_6300, _ = fill_array(stds_6300, tstamps) # type: ignore
    _, imgs_6306, _ = fill_array(imgs_6306, tstamps) # type: ignore
    _, stds_6306, _ = fill_array(stds_6306, tstamps) # type: ignore
    _, mds_5577, _ = fill_array(mds_5577, tstamps) # type: ignore
    _, mds_ap, _ = fill_array(mds_ap[:, None], tstamps, axis=0) # type: ignore
    _, hmf, _ = fill_array(dds['hmF'].values[:, None], tstamps, axis=0) # type: ignore
    tstamps, mds_6300, _ = fill_array(mds_6300, tstamps) # type: ignore
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

    tax: plt.Axes = ax.twinx() # type: ignore
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
    lobjs = [(l_55, f_55), m_55, (l_63, f_63), m_63]  # , l_ap] # type: ignore
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
    ax.text(v[0], 0.5, 'Data Unavailable', ha='center', # type: ignore
            va='center', fontsize=8, color='r', rotation='vertical')

for ax in axes.flatten()[-2:]:
    xticks = np.asarray(ax.get_xticks())
    xticks = np.round(xticks, decimals=1)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlabel("Local Time (UTC$-$05:00)")
fig.savefig(KEOGRAMS_DIR / 'hmf_variation.pdf', dpi=600, bbox_inches='tight')
if is_interactive_session():
    plt.show()
else:
    plt.close(fig)
# %%
