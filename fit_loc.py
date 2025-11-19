# %% Imports
from __future__ import annotations
from common_funcs import fill_array, get_date, make_color_axis, get_smoothed_geomag
from settings import Directories, is_interactive_session
from collections.abc import Iterable
import datetime as dt
from pathlib import Path
from typing import Sequence, SupportsFloat as Numeric
from tzlocal import get_localzone
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pytz
import matplotlib
import pandas as pd
from dateutil.parser import parse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
INTERACTIVE = is_interactive_session()
# %% Functions


def fmt_time(x: Numeric, ofst: dt.datetime) -> str:
    x = dt.timedelta(hours=x)  # type: ignore
    res = ofst + x  # type: ignore
    return res.strftime('%H:%M')
# %%


def plot_density(iono: xr.Dataset, keys: Sequence[str], exkeys: Sequence[str], outputdir: Path, file_prefix: str, *, vmin=None, vmax=None, log=False, cmap='bone'):
    dtime = parse(iono.time).astimezone(
        get_localzone()) - dt.timedelta(hours=8)
    day = dtime.strftime('%Y-%m-%d')
    tstamps = iono.tstamp.values
    tstamps = list(map(lambda t: pd.to_datetime(t).to_pydatetime(), tstamps))
    vals = {}
    for key in keys[:-1]:
        arr = iono[key].values
        _, arr, _ = fill_array(arr, tstamps, axis=0)  # type: ignore
        vals[key] = arr
    tstamps, vals[keys[-1]], nanfill = fill_array(  # type: ignore
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
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)  # type: ignore
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
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)  # type: ignore
            cbar.ax.locator_params('y')
            cbar.ax.tick_params(labelsize=8)
            ticks = np.asarray(cbar.ax.get_yticks())
            if np.log10(ticks.max() - ticks.min()) > np.log10(3e3):
                cbar.formatter.set_powerlimits((0, 0))  # type: ignore
                # to get 10^3 instead of 1e3
                cbar.formatter.set_useMathText(True)  # type: ignore
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
    os.makedirs(outputdir, exist_ok=True)
    fig.savefig(
        outputdir / f'{file_prefix}_{day.replace("-", "")}.pdf', dpi=600)
    if is_interactive_session():
        fig.show()
    else:
        plt.close(fig)

# %%


def plot_density2(iono: xr.Dataset, keys: Sequence[str], exkeys: Sequence[str], outputdir: Path, file_prefix: str, *, vmin: Numeric | Iterable = None, vmax: Numeric | Iterable = None, log: bool = False, cmap: str = 'bone', alt_min: Numeric | Iterable = None, alt_max: Numeric | Iterable = None):  # type: ignore
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
                {'wavelength': key, 'alt_km': slice(
                    alt_min[idx], alt_max[idx])}  # type: ignore
            ).values
            key_ver[key] = True
        except ValueError:
            arr = iono[key].sel(
                {'alt_km': slice(alt_min[idx], alt_max[idx])}  # type: ignore
            ).values
            key_ver[key] = False
        # _, arr = fill_array(arr, tstamps, axis=0)
        vals[key] = arr
    for key in keys[:-1]:
        _, arr, _ = fill_array(
            vals[key].copy(), tstamps, axis=0)  # type: ignore
        vals[key] = arr
    tstamps, vals[keys[-1]], nanfill = fill_array(  # type: ignore
        vals[keys[-1]].copy(),
        tstamps,
        axis=0
    )

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
        ax: plt.Axes = ax  # type: ignore
        title = exkeys[idx]
        if ' ' not in title:
            ax.set_title(rf'{title}')
        else:
            ax.set_title(title)
        val = vals[key].T.copy()
        val[np.where(np.isnan(val))] = 1e-4
        alt_km = iono.alt_km.sel(
            {'alt_km': slice(alt_min[idx], alt_max[idx])}  # type: ignore
        ).values
        tx, hy = np.meshgrid(ttstamps, alt_km)
        if log:
            im = ax.pcolormesh(
                tx, hy, np.log10(val), cmap=cmap,
                vmin=(vmin[idx]), vmax=vmax[idx],  # type: ignore
            )
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)  # type: ignore
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
            im = ax.pcolormesh(
                tx, hy, val, cmap=cmap,
                vmin=vmin[idx], vmax=vmax[idx],  # type: ignore
            )
            cbar = fig.colorbar(im, cax=cax[idx], shrink=0.5)  # type: ignore
            cbar.ax.locator_params('y')
            cbar.ax.tick_params(labelsize=8)
            ticks = np.asarray(cbar.ax.get_yticks())
            if np.log10(ticks.max() - ticks.min()) > np.log10(3e3):
                cbar.formatter.set_powerlimits((0, 0))  # type: ignore
                # to get 10^3 instead of 1e3
                cbar.formatter.set_useMathText(True)  # type: ignore
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
            cbar.ax.set_ylabel(
                fr'{arr.attrs["long_name"].title()} (${arr.attrs["units"]}$)',
                fontsize=8,
            )
        else:
            cbar.ax.set_ylabel(
                fr'VER (${arr.attrs["units"]}$)',
                fontsize=8,
            )
    xticks = np.asarray(axs[-1, 0].get_xticks())
    xticks = np.round(xticks, decimals=1)
    xticks = list(map(lambda x: fmt_time(x, start), xticks))
    axs[-1, 0].set_xticklabels(xticks)
    axs[-1, 1].set_xticklabels(xticks)
    axs[-1, 0].set_xlabel('Local Time (UTC$-$5:00)')
    axs[-1, 1].set_xlabel('Local Time (UTC$-$5:00)')
    os.makedirs(outputdir, exist_ok=True)
    ofile = outputdir / f'{file_prefix}_{day.replace("-", "")}.pdf'
    fig.savefig(ofile, dpi=600, bbox_inches='tight')
    print(f'Saved {ofile}')
    if is_interactive_session():
        plt.show()
    else:
        plt.close(fig)

# %%


def runner(settings: Directories):
    files = list(settings.model_dir.glob('fitres*.xz'))
    files.sort(key=get_date)

    dates = list(map(get_date, files))
    lat, lon = 42.64981361744372, -71.31681056737486

    for date, _ in zip(dates, files):
        if not (settings.model_dir / f'vert_{date}.nc').exists():
            raise FileNotFoundError(
                f'File {settings.model_dir / f"vert_{date}.nc"} not found. Please run generate_vert first.')
        iono = xr.load_dataset(settings.model_dir / f'vert_{date}.nc')
        # keys = ['O', 'O+', 'O2+']
        # exkeys = ['O', 'O^+', 'O_2^+']
        # plot_density(iono, keys, exkeys, vertprops_dir, 'fitprops_vert', vmin=2, log=False, cmap='gist_ncar_r')
        # keys = ['NS', 'N2D', 'NeIn']
        # exkeys = ['N(2S)', 'N(2D)', 'e^-']
        # plot_density(iono, keys, exkeys, vertprops_dir, 'fitprops_vert', vmin=2, log=False, cmap='gist_ncar_r')
        # keys = ['O', 'O+', 'O2+'] + ['NS', 'N2D', 'NeIn']
        keys = ['O', 'O+', 'O2+'] + ['O2', 'N2', 'NeIn']  # , '5577', '6300']
        # exkeys = ['O', 'O^+', 'O_2^+'] + ['N(2S)', 'N(2D)', 'e^-']
        exkeys = ['O', 'O$^+$', 'O$_2^+$'] + \
            ['O$_2$', 'N$_2$', 'e$^-$']  # , '5577 Å', '6300 Å']
        altmin = [70, 100, 70, 60, 60, 100]
        altmax = [200, 800, 400, 100, 100, 800]
        plot_density2(
            iono, keys, exkeys, settings.vertprops_dir, 'all_den',
            # , 1e-4, 1e-4]
            vmin=[2, 2, 2, 2, 2, 2], log=False, cmap='gist_ncar_r',
            alt_min=altmin, alt_max=altmax
        )
        keys = ['Tn', 'Ti', 'Te']
        exkeys = [
            'Neutral Temperature',
            'Ion Temperature',
            'Electron Temperature'
        ]
        plot_density(
            iono, keys, exkeys, settings.vertprops_dir,
            'temps', vmin=100, log=False, cmap='hot'
        )


# %%
if not INTERACTIVE:
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate vertical profiles from fitres files.')
    parser.add_argument('suffix', type=str, default=None, nargs='*',
                        help='Suffix for the output files.')
    args = parser.parse_args()
    suffixes = list(map(lambda x: x.strip(), args.suffix))
    suffixes = list(filter(lambda x: len(x) > 0, suffixes))
    if len(suffixes) == 0:
        suffixes = [None]
    for suffix in suffixes:
        print(f'Processing suffix: {suffix}')
        dirs = Directories(suffix=suffix)
        runner(dirs)
# %%
