# %% Imports
from __future__ import annotations
from common_funcs import get_date
from settings import Directories, is_interactive_session
import datetime as dt
from io import TextIOWrapper
import lzma
from pathlib import Path
import pickle
from typing import List, SupportsFloat as Numeric, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pytz
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# %% Functions


def fmt_time(x: Numeric, ofst: dt.datetime) -> str:
    x = dt.timedelta(hours=x) # type: ignore
    res = ofst + x # type: ignore
    return res.strftime('%H:%M')
# %% Line styles
linestyle_str = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
    ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),    # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_dict = {
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
# %% For each day


def generate_vert(output: Path, date: str, file: Path, ofile: TextIOWrapper, tfile: TextIOWrapper, keys: List[str], save_figs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    with lzma.open(file, 'rb') as f:
        fitres = pickle.load(f)
    tstamps = [x[0] for x in fitres]
    start = pd.to_datetime(
        tstamps[0]).to_pydatetime()
    end = pd.to_datetime(
        tstamps[-1]).to_pydatetime()
    print(f'Processing {start:%Y-%m-%d}')
    # Get the model data
    # pbar = tqdm(range(len(tstamps)))
    pbar = range(len(tstamps))
    den_part = np.full((len(tstamps), 6), np.nan)
    for idx in pbar:
        res = fitres[idx][1]
        den_part[idx, :] = (res.x[0], res.x[1], res.x[2],
                            res.x[3], res.x[4], res.x[5])
    den_o = np.array(den_part[:, 0])
    den_o2 = np.array(den_part[:, 1])
    den_n2 = np.array(den_part[:, 2])
    den_no = np.array(den_part[:, 3])
    den_n4s = np.array(den_part[:, 4])
    den_e = np.array(den_part[:, 5])
    tstamps = np.asarray(tstamps, dtype=int)
    tstamps = tstamps.astype(float)
    tstamps *= 1e-9  # convert to seconds
    sstart = dt.datetime.fromtimestamp(tstamps[0])  # start
    sstart = dt.datetime(sstart.year, sstart.month,
                         sstart.day, sstart.hour, 0, 0)
    tstamps -= sstart.timestamp()
    sstart = sstart.astimezone(pytz.utc)
    """Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-
    """
    stats = {}
    stats['O'] = (np.nanmean(den_o), np.nanstd(den_o), np.nanmedian(
        den_o), np.nanmin(den_o), np.nanmax(den_o))
    stats['O2'] = (np.nanmean(den_o2), np.nanstd(den_o2), np.nanmedian(
        den_o2), np.nanmin(den_o2), np.nanmax(den_o2))
    stats['N2'] = (np.nanmean(den_n2), np.nanstd(den_n2), np.nanmedian(
        den_n2), np.nanmin(den_n2), np.nanmax(den_n2))
    stats['NO'] = (np.nanmean(den_no), np.nanstd(den_no), np.nanmedian(
        den_no), np.nanmin(den_no), np.nanmax(den_no))
    stats['N4S'] = (np.nanmean(den_n4s), np.nanstd(den_n4s), np.nanmedian(
        den_n4s), np.nanmin(den_n4s), np.nanmax(den_n4s))
    stats['e-'] = (np.nanmean(den_e), np.nanstd(den_e),
                   np.nanmedian(den_e), np.nanmin(den_e), np.nanmax(den_e))
    _, den_o = fill_array(den_o, tstamps) # type: ignore
    _, den_o2 = fill_array(den_o2, tstamps) # type: ignore
    _, den_n2 = fill_array(den_n2, tstamps) # type: ignore
    _, den_no = fill_array(den_no, tstamps) # type: ignore
    _, den_n4s = fill_array(den_n4s, tstamps) # type: ignore
    tstamps, den_e = fill_array(den_e, tstamps) # type: ignore
    tstamps = np.asarray(tstamps, dtype=float)
    tstamps /= 3600  # convert to hours
    fig = plt.figure(figsize=(4.8, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(tstamps, den_o,
            label='O', color='blue',
            linewidth=0.75, linestyle=linestyle_dict['dotted'])
    ax.plot(tstamps, den_o2, label='O$_2$', color='red',
            linewidth=0.75, linestyle=linestyle_dict['loosely dashed'])
    ax.plot(tstamps, den_n2, label='N$_2$', color='green',
            linewidth=0.75, linestyle=linestyle_dict['dashdot'])
    ax.plot(tstamps, den_no, label='NO', color='purple',
            linewidth=0.75, linestyle=linestyle_dict['densely dashdotted'])
    ax.plot(tstamps, den_n4s, label='N$(^4S)$', color='orange',
            linestyle=linestyle_dict['dashdotdotted'], linewidth=0.75)
    ax.plot(tstamps, den_e, label='e$^-$', color='black',
            linewidth=0.75)
    ax.set_xlabel('Local Time (Hours)')
    ax.set_ylabel('Density Perturbation')
    ax.set_xlim(0, 9)
    xticks = ax.get_xticks()
    xticklabels = list(map(lambda x: fmt_time(x, sstart), xticks))
    ax.set_xticklabels(xticklabels)
    ax.legend()
    ax.set_ylim(0.25, 3)
    ax.set_title(f'{sstart:%Y-%m-%d} {start:%H:%M} - {end:%H:%M} (UTC-05:00)')
    if save_figs:
        fig.savefig(output / f'fit_den_{date}.png',
                    dpi=600, bbox_inches='tight')
    if is_interactive_session():
        print(f'Showing figure for {sstart:%Y-%m-%d}')
        plt.show()
    plt.close(fig)

    ofile.write('\n')
    ofile.write(f'{sstart:%Y-%m-%d},\t')
    tfile.write(f'{sstart:%Y-%m-%d} ')
    for key in keys:
        vals = stats[key]
        ofile.write(
            f'{vals[0]:.3f}+/-{vals[1]:.3f},\t{vals[3]:.3f},\t{vals[4]:.3f},\t')
        tfile.write(f'& ${vals[0]:.2f}^{{{vals[4]:.2f}}}_{{{vals[3]:.2f}}}$ ')
    tfile.write(r'\\' + '\n')
    return tstamps, den_part

# %% Runner


def runner(settings: Directories, save_figs: bool = True):
    model_dir = settings.model_dir
    vertprops_dir = settings.vertprops_dir
    files = list(model_dir.glob('fit_den_*.xz'))
    files.sort(key=get_date)
    keys = ['O', 'O2', 'N2', 'NO', 'N4S', 'e-']
    dates = list(map(get_date, files))
    with open(vertprops_dir / 'fit_den_stats.csv', 'w') as ofile, open(vertprops_dir / 'fit_den_tex.tex', 'w') as tfile:
        ofile.write('Date,\t')
        for key in keys:
            ofile.write(f'{key} Mean,\t{key} Min,\t{key} Max,\t')
        tfile.write(
            r"""
\begin{tabular}{r c c c c c c}
    \hline
    Date & O & O$_2$ & N$_2$ & NO & N($^4S$) & e$^-$ \\
    \hline
    """
        )
        for date, file in zip(dates, files):
            generate_vert(vertprops_dir, date, file, ofile,
                          tfile, keys, save_figs=save_figs)
        tfile.write(r"""
    \hline
\end{tabular}
        """)


# %% Main
if not is_interactive_session():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate vertical properties from GLOW model fits.')
    parser.add_argument('suffix', type=str, default=None, nargs='?',
                        help='Suffix for the output files.')
    parser.add_argument('--save_figs', action='store_true', default=True,
                        help='Save fit figures to disk.')
    args = parser.parse_args()
    if args.suffix is None or args.suffix.strip() == '':
        args.suffix = None
    dirs = Directories(suffix=args.suffix)
    runner(dirs, save_figs=args.save_figs)

# %%
