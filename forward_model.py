# %% Imports
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, SupportsFloat as Numeric, Tuple
import xarray as xr
import numpy as np
import pytz

import geomagdata as gi
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from multiprocessing import Pool, cpu_count

import glow2d
from tqdm import tqdm

import pandas as pd

from settings import Directories, is_interactive_session, ROOT_DIR
from common_funcs import get_smoothed_geomag

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# %%
sds = xr.load_dataset(ROOT_DIR / 'keo_scale.nc')
scale_5577 = sds['5577'].values[::-1]
scale_6300 = sds['6300'].values[::-1]
za_min = sds['za_min'].values
za_max = sds['za_max'].values
# %%


class GLOWFwd:
    def __init__(self, time: dt.datetime, lat: Numeric, lon: Numeric, heading: Numeric, geomag_params: Dict[str, Numeric], za_min: np.ndarray, za_max: np.ndarray, za_idx: int, tec: xr.Dataset, m_pool=None):
        self._time = time
        self._lat = lat
        self._lon = lon
        self._heading = heading
        self._geopar = geomag_params
        self._zamin = za_min
        self._zamax = za_max
        self._zaidx = za_idx
        self._bright = None
        self._pool = m_pool
        self._tec = tec

        self._update()

    @property
    def emission(self):
        return self._bright

    @property
    def tecscale(self):
        return self._tecscale

    def _update(self):
        iono = glow2d.polar_model(
            self._time, self._lat, self._lon, self._heading, n_pts=20,
            geomag_params=self._geopar, tec=self._tec, mpool=self._pool
        )  # type: ignore
        self._tecscale = iono['tecscale'].copy()  # type: ignore
        ec5577 = glow2d.glow2d_polar.get_emission(  # type: ignore
            # type: ignore
            iono, feature='5577', za_min=self._zamin, za_max=self._zamax)[::-1]
        ec6300 = glow2d.glow2d_polar.get_emission(  # type: ignore
            # type: ignore
            iono, feature='6300', za_min=self._zamin, za_max=self._zamax)[::-1]
        self._bright = [ec5577[::-1], ec6300[::-1]]


# %%
# dates = ['20220209']
def runner(model_dir: Path, counts_dir: Path):
    tec = xr.open_dataset(ROOT_DIR / 'gpstec_lowell.nc')
    with Pool(6) as m_pool:
        dates = [
            '20220126', '20220209', '20220215', '20220218',
            '20220219', '20220226', '20220303', '20220304'
        ]
        za_idx = 20
        for date in dates:
            outfile = model_dir / f'fwdmodel_{date}.nc'
            if Path(outfile).exists():
                date = dt.datetime.strptime(
                    date, '%Y%m%d').strftime('%Y-%m-%d')
                print(f'{outfile} exists. Skipping {date}')
                continue
            ds = xr.load_dataset(counts_dir / f'hitmis_cts_{date}.nc')
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

            lat, lon = 42.64981361744372, -71.31681056737486
            _, ap, f107, f107a, f107p = get_smoothed_geomag(
                tstamps)  # type: ignore
            br6300 = np.zeros((len(ds.tstamp), len(ds.height)), dtype=float)
            br5577 = np.zeros((len(ds.tstamp), len(ds.height)), dtype=float)
            pbar = tqdm(range(len(ds.tstamp.values)))

            for idx in pbar:
                geomag_params = {
                    'ap': float(ap[idx]),
                    'f107': float(f107[idx]),
                    'f107a': float(f107a[idx]),
                    'f107p': float(f107p[idx]),
                }
                minf = GLOWFwd(
                    # type: ignore
                    tstamps[idx], lat, lon, 40, geomag_params=geomag_params, za_min=za_min,
                    za_max=za_max, za_idx=za_idx, tec=tec, m_pool=m_pool
                )
                out = minf.emission
                br5577[idx, :] += out[0]  # type: ignore
                br6300[idx, :] += out[1]  # type: ignore
                pbar.set_description(
                    f'[{pd.to_datetime(ds.tstamp.values[idx]).to_pydatetime():%Y-%m-%d %H:%M}]', refresh=True)

            kds = xr.Dataset(
                data_vars={
                    '5577': (('tstamp', 'height'), br5577),
                    '6300': (('tstamp', 'height'), br6300),
                    'ap': (('tstamp'), ap),
                    'f107a': (('tstamp'), f107a),
                    'f107': (('tstamp'), f107),
                    'f107p': (('tstamp'), f107p),
                    'lat': (('tstamp'), [lat]*len(tstamps)),
                    'lon': (('tstamp'), [lon]*len(tstamps)),
                    'to_r': 1/(dheight * 4*np.pi*1e-6)
                },
                coords={
                    'tstamp': ds.tstamp.values,
                    'height': ds.height.values,
                }
            )
            unit_desc = {
                '5577': ('cm^{-2} s^{-1} rad^{-1}', '5577 Brightness'),
                '6300': ('cm^{-2} s^{-1} rad^{-1}', '6300 Brightness'),
                'ap': ('', 'Planetary ap index (3 hour UTC)'),
                'f107a': ('sfu', '81-day rolling average of F10.7 solar flux'),
                'f107': ('sfu', 'F10.7 solar flux on present day'),
                'f107p': ('sfu', 'F10.7 solar flux on previous day'),
                'lat': ('deg', 'Latitude'),
                'lon': ('deg', 'Longitude'),
                'to_r': ('R rad^{-1}', 'Convert brightness to Rayleigh')
            }
            _ = list(map(lambda x: kds[x].attrs.update(
                {'units': unit_desc[x][0], 'description': unit_desc[x][1]}), unit_desc.keys()))
            kds.to_netcdf(outfile)


# %%
if not is_interactive_session():
    dirs = Directories()
    model_dir = dirs.model_dir
    counts_dir = dirs.counts_dir
    runner(model_dir, counts_dir)
