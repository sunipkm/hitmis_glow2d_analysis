# %% Imports
from __future__ import annotations
from functools import partial
import matplotlib
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import numpy as np
import datetime as dt
import pandas as pd
import pytz
import geomagdata as gi
from pathlib import Path
from skmpython import staticvars
import xarray as xr
from typing import List, Optional, Tuple, Iterable, SupportsFloat as Numeric
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %% MPL settings


usetex = False
if not usetex:
    # computer modern math text
    matplotlib.rcParams.update({'mathtext.fontset': 'cm'})

matplotlib.rc(
    'font',
    **{
        'family': 'serif',
        'serif': ['Times' if usetex else 'Times New Roman']
    }
)
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
matplotlib.rc('text', usetex=usetex)
# %% Interpolate + Smoothing


def do_interp_smoothing(x: np.ndarray, xp: np.ndarray, yp: np.ndarray, sigma: int | float = 22.5, round: int = None):  # type: ignore
    y = interp1d(
        xp, yp, kind='nearest-up',
        fill_value='extrapolate'  # type: ignore
    )(x)
    y = gaussian_filter1d(y, sigma=sigma)
    if round is not None:
        y = np.round(y, decimals=round)
    return y


def get_smoothed_geomag(tstamps: np.ndarray, tzaware: bool = False) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            [td - dt.timedelta(days=1), td],  # type: ignore
            81, tzaware=tzaware  # type: ignore
        )
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

    return tdtime, ap, f107, f107a, f107p  # type: ignore


def get_date(filename: Path) -> str:
    """Get the date from a filename."""
    base = filename.name
    return base.rsplit('.')[0].rsplit('_')[-1]

# %%


def fill_array(arr: np.ndarray, tstamps: List[dt.datetime], axis: int = 1) -> Tuple[List[dt.datetime], np.ndarray, bool]:
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
        return tstamps, arr, False
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
    return (tstamps, out, True)
# %%
# %%


def fill_array_1d(arr: np.ndarray, tstamps: List[dt.datetime]) -> Tuple[List[dt.datetime], np.ndarray, Optional[np.ndarray]]:
    if arr.ndim != 1:
        raise ValueError('Array must be 1 dim')
    ts = np.asarray(list(map(lambda t: t.timestamp(), tstamps)), dtype=float)
    dts = np.diff(ts)
    t_delta = dts.min()
    gaps = dts[np.where(dts > t_delta)[0]]
    gaps = np.asarray(gaps // t_delta, dtype=int)
    dts = np.diff(dts)
    oidx = np.where(dts < 0)[0]
    if len(oidx) == 0:
        return tstamps, arr, None
    tstamps = []
    tlen = int((ts[-1] - ts[0]) // t_delta) + 1
    for idx in range(tlen):
        tstamps.append(dt.datetime.fromtimestamp(
            ts[0] + t_delta*idx).astimezone(pytz.utc))
    out = np.full((tlen), dtype=arr.dtype, fill_value=np.nan)
    start = 0
    dstart = 0
    nanlocs = []
    for idx, oi in enumerate(oidx):
        out[start:oi+1] = arr[dstart:oi+1]

        start = oi + gaps[idx]
        nanlocs.append(oi)
        dstart = oi + 1
        if idx == len(oidx) - 1:  # end
            out[start:] = arr[dstart:]
            nanlocs.append(oi + gaps[idx])
    nanlocs = None if len(nanlocs) == 0 else np.asarray(nanlocs, dtype=int)
    return (tstamps, out, nanlocs)

# %%


def make_color_axis(ax: Axes | Iterable, position: str = 'right', size: str = '1.5%', pad: float = 0.05) -> Axes | list:
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
    assert (a > 0 and b > 0)  # type: ignore
    return np.rad2deg(np.arctan2(a*np.tan(np.deg2rad(lat)), b))  # type: ignore


def geodet_to_geocent(lat: Numeric, ell: Tuple[Numeric, Numeric] = (6378137.0, 6356752.3142)) -> Numeric:
    """Converts geodetic latitude to geocentric latitude

    Args:
        lat (Numeric): Geodetic latitude (degrees)
        ell (Tuple[Numeric, Numeric], optional): Semi-major and semi-minor axes. Defaults to WGS84(6378137.0, 6356752.3142).

    Returns:
        Numeric: Geocentric latitude (degrees)
    """
    a, b = ell
    assert (a > 0 and b > 0)  # type: ignore
    return np.rad2deg(np.arctan(b*np.tan(np.deg2rad(lat))/a))  # type: ignore


@staticvars(gpstec=None)
def get_gps_tec(tstart: Numeric, tstop: Numeric, latrange: slice = None, lonrange: slice = None, *, fname: str = 'gpstec_lowell.nc') -> xr.Dataset:  # type: ignore
    if get_gps_tec.gpstec is None:  # type: ignore
        get_gps_tec.gpstec = xr.open_dataset(fname)  # type: ignore
    gpstec: xr.Dataset = get_gps_tec.gpstec  # type: ignore
    gpstec = gpstec.sel(timestamps=slice(tstart, tstop))  # type: ignore
    if latrange is not None:
        gpstec = gpstec.sel(gdlat=latrange)
    if lonrange is not None:
        gpstec = gpstec.sel(glon=lonrange)
    return gpstec
# %%


def get_tec(iono: xr.Dataset) -> np.ndarray:
    try:
        from scipy.integrate import trapezoid as trapz
    except ImportError:
        from scipy.integrate import trapz  # type: ignore
    ne = iono['NeOut'].values.copy()
    ne = np.nan_to_num(ne, nan=0)
    alt = iono['alt_km'].values
    tsh, _ = ne.shape
    tec = np.zeros(tsh)
    for idx in range(tsh):
        tec[idx] += 2*trapz(ne[idx, :], alt)
    tec *= 1e9  # convert to m^-2
    return tec


# %% Line styles
LINESTYLE_STR = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
    ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),    # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'

LINESTYLE_DICT = {
    'loosely dotted':      (0, (1, 10)),
    'dotted':              (0, (1, 1)),
    'densely dotted':      (0, (1, 1)),
    'long dash with offset': (5, (10, 3)),
    'loosely dashed':      (0, (5, 10)),
    'dashed':              (0, (5, 5)),
    'densely dashed':      (0, (5, 1)),
    'dashdot':             (0, (3, 5, 1, 5)),
    'loosely dashdotted':  (0, (3, 10, 1, 10)),
    'dashdotted':          (0, (3, 5, 1, 5)),
    'densely dashdotted':  (0, (3, 1, 1, 1)),
    'dashdotdotted':       (0, (3, 5, 1, 5, 1, 5)),
    'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}
