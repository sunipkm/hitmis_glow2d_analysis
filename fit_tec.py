# %% Imports
from __future__ import annotations
from itertools import repeat
from pathlib import Path
from typing import SupportsFloat as Numeric, Tuple

from common_funcs import geocent_to_geodet, get_date, get_gps_tec, get_tec
from settings import Directories, is_interactive_session

from matplotlib.axes import Axes
import datetime as dt
import lzma
import multiprocessing
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pytz
from glowpython import no_precipitation
import matplotlib
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# %% Functions


def fmt_time(x: Numeric, ofst: dt.datetime) -> str:
    x = dt.timedelta(hours=x)  # type: ignore
    res = ofst + x  # type: ignore
    return res.strftime('%H:%M')
# %% For each day


def generate_vert(model_dir: Path, date: str, file: Path):
    print(f'Processing {date}...')
    lat, lon = 42.64981361744372, -71.31681056737486
    if os.path.exists(model_dir / f'vert_{date}.nc'):
        ionos = xr.load_dataset(model_dir / f'vert_{date}.nc')
        return
    ionos = []
    with lzma.open(file, 'rb') as f:
        fitres = pickle.load(f)
    # Get the model data
    tstamps = [x[0] for x in fitres]
    _, ap, f107, f107a, f107p = get_smoothed_geomag(tstamps)  # type: ignore
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
    ionos.to_netcdf(model_dir / f'vert_{date}.nc')
    return ionos


# type: ignore
def plot_tec(settings: Directories, date: str, ax: Axes) -> Tuple[dt.datetime, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    import digisondeindices as di
    iono = xr.load_dataset(settings.model_dir / f'vert_{date}.nc')
    tstamps = iono.tstamp.values
    print(tstamps[0], tstamps[-1])
    lat, lon = 42.64981361744372, -71.31681056737486
    dlat = geocent_to_geodet(lat)
    gpsstart = int(tstamps[0])*1e-9 - 600
    gpsstop = int(tstamps[-1])*1e-9 + 600
    gpstec = get_gps_tec(gpsstart, gpsstop, latrange=slice(
        dlat-0.5, lat+0.5), lonrange=slice(lon-0.5, lon+0.5))  # type: ignore
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
    ds = di.get_indices(tstamps_, 'MHJ45')  # type: ignore
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
if not is_interactive_session():
    import argparse
    parser = argparse.ArgumentParser(
        description='Fit TEC from GLOW model and GPS data')
    parser.add_argument('suffix', type=str, default=None, nargs='?',
                        help='Suffix for the directories')
    args = parser.parse_args()
    if args.suffix is None or args.suffix.strip() == '':
        args.suffix = None
    dirs = Directories(suffix=args.suffix)

    # glob.glob(f'{MODEL_DIR}/fitres*.xz')
    files = list(dirs.model_dir.glob('fitres*.xz'))
    files.sort(key=get_date)

    dates = list(map(get_date, files))
    lat, lon = 42.64981361744372, -71.31681056737486

    with multiprocessing.Pool(4) as pool:
        res = pool.starmap(generate_vert, zip(
            repeat(dirs.model_dir), dates, files))
    try:
        ionos = res[0]
    except IndexError:
        print(f'No data found for {dirs.model_dir}. Please run the model first.')
        exit(1)

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

    with open(dirs.fitprops_dir / f'tec_correlation.csv', 'w') as csvout, open(dirs.fitprops_dir / f'tec_correlation.tex', 'w') as texout:
        csvout.write('Date,Digisonde Correlation,GPS Correlation')
        texout.write(
            r"""
\begin{tabular}{ccc}
\hline
Date & Digisonde Correlation & GNSS Correlation \\
\hline"""
        )
        for date, ax in zip(dates, axes.flatten()):
            start, _, _, _, _, digicorr, gpscorr = plot_tec(dirs, date, ax)
            tot_digicorr *= digicorr
            tot_gpscorr *= gpscorr
            csvout.write(f'\n{start:%Y-%m-%d},{digicorr:.2f},{gpscorr:.2f}')
            texout.write(
                f'\n{start:%Y-%m-%d} & {digicorr:.2f}% & {gpscorr:.2f}% \\\\')

        tot_digicorr = tot_digicorr**(1/len(dates))
        tot_gpscorr = tot_gpscorr**(1/len(dates))
        csvout.write(f'\nGeomean,{tot_digicorr:.2f},{tot_gpscorr:.2f}')
        texout.write(
            f'\n\\hline\nGeomean & {tot_digicorr:.2f}% & {tot_gpscorr:.2f}% \\\\')
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
    plt.savefig(f'{dirs.fitprops_dir}/tec_profile.pdf',
                dpi=600, bbox_inches='tight')
    if is_interactive_session():
        plt.show()
    else:
        plt.close(fig)
# %%
