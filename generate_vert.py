# %% Imports
from __future__ import annotations
from itertools import repeat
from common_funcs import get_date, get_smoothed_geomag
from settings import Directories, is_interactive_session
import lzma
import multiprocessing
from pathlib import Path
import pickle
import xarray as xr
import os
import pytz
from glowpython2 import no_precipitation
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
INTERACTIVE = is_interactive_session()

# %%


def generate_vert(model_dir: Path, date: str, file: Path, newmodel: bool = False, overwrite: bool = False):
    print(f'Processing {date}...')
    lat, lon = 42.64981361744372, -71.31681056737486
    if newmodel:
        magmodel = 'IGRF14'
        version = 'MSIS21_IRI20'
    else:
        magmodel = 'POGO68'
        version = 'MSIS00_IRI90'
    if not overwrite and os.path.exists(model_dir / f'vert_{date}.nc'):
        ionos = xr.load_dataset(model_dir / f'vert_{date}.nc')
    ionos = []  # type: ignore
    with lzma.open(file, 'rb') as f:
        fitres = pickle.load(f)
    # Get the model data
    tstamps = [x[0] for x in fitres]
    _, ap, f107, f107a, f107p = get_smoothed_geomag(tstamps)  # type: ignore
    # pbar = tqdm(range(len(tstamps)))
    pbar = range(len(tstamps))
    ionos = []  # type: ignore
    for idx in pbar:
        geomag_params = {
            'Ap': float(ap[idx]),
            'f107': float(f107[idx]),
            'f107a': float(f107a[idx]),
            'f107p': float(f107p[idx]),
        }
        res = fitres[idx][1]
        if res is None:
            print('None')
        else:
            time = pd.to_datetime(
                tstamps[idx]).to_pydatetime().astimezone(pytz.utc)
            density_pert = (res.x[0], res.x[1], res.x[2],
                            res.x[3], res.x[4], 1, res.x[5])
            iono = no_precipitation(
                time, lat, lon, 100, density_pert,
                geomag_params=geomag_params,
                magmodel=magmodel, version=version
            )
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
    ionos: xr.Dataset = xr.concat(ionos, pd.Index(  # type: ignore
        tstamps, name='tstamp'))  # type: ignore
    ionos.to_netcdf(model_dir / f'vert_{date}.nc')
# %%


def runner(settings: Directories, newmodel: bool = False, overwrite: bool = False):
    files = list(settings.model_dir.glob('fitres*.xz'))
    files.sort(key=get_date)

    dates = list(map(get_date, files))
    lat, lon = 42.64981361744372, -71.31681056737486

    with multiprocessing.Pool(4) as pool:
        res = pool.starmap(
            generate_vert,
            zip(
                repeat(settings.model_dir), dates, files,
                repeat(newmodel), repeat(overwrite)
            )
        )


# %%
if not INTERACTIVE:
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate vertical profiles from fitres files.')
    parser.add_argument('suffix', type=str, default=None, nargs='*',
                        help='Suffix for the output files.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files.')
    parser.add_argument('--newmodel', action='store_true',
                        help='Use new model settings.')
    args = parser.parse_args()
    suffixes = list(map(lambda x: x.strip(), args.suffix))
    suffixes = list(filter(lambda x: len(x) > 0, suffixes))
    if len(suffixes) == 0:
        suffixes = [None]
    for suffix in suffixes:
        print(
            f'Processing suffix: {suffix}, newmodel={args.newmodel}, overwrite={args.overwrite}')
        dirs = Directories(suffix=suffix)
        runner(
            dirs,
            newmodel=args.newmodel,
            overwrite=args.overwrite
        )
