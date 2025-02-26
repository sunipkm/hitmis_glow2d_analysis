# %%
from glob import glob
import os
import re
from tqdm import tqdm
import xarray as xr
# %%
def get_key(fname: str) -> int:
    out = os.path.basename(fname).split('.')[0]
    return int(re.sub('\D', '', out))

# Download the GNSS TEC data from the OpenMadrigal database
# This script is used to concatenate the data into a single file
# Date range: 2022-01-23 to 2022-03-07

files = glob('*.hdf5') # really are netcdf files, and should be downloaded as such
files.sort(key=get_key)

ds_master = None
for idx, file in enumerate(tqdm(files)):
    with xr.load_dataset(file) as ds:
        if ds_master is None:
            ds_master = ds
        else:
            ds_master = xr.concat([ds_master, ds], dim='timestamps')
# %%
encoding = {
    'timestamps': {'dtype': 'float64', 'zlib': True},
    'gdlat': {'dtype': 'float64', 'zlib': True},
    'glon': {'dtype': 'float64', 'zlib': True},
    'dtec': {'dtype': 'float64', 'zlib': True},
    'tec': {'dtype': 'float64', 'zlib': True},
}
# %%
ds_master.to_netcdf('gpstec_lowell.nc', encoding=encoding)
# %%
