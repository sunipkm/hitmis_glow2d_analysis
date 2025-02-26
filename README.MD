# Data and scripts
## Modeling
1. `glow_fit.py`: Fits measured 5577 and 6300 intensities (`keocounts/`) to the model to retrieve neutral and electron densities.
2. `forward_model.py`: Evaluates the GLOW model for the dates where 5577 and 6300A intensities exist.

## Plotting
1. `keo_plot.py`: Compares intensities between model fit for retrieval and measured intensities.
2. `fwdmodel_analysis.py`: Compares forward model intensities to measured intensities. Compares residual 5577 variation to hmF2 variation.
3. `fit_tec.py`: Compares TEC from retrieved e density profile to Digisonde and GNSS TEC.
4. `fit_loc.py`: Generates vertical temperature and density profiles.

## Data Aggregation
1. `tec_concat.py`: Concatenates GNSS data from the OpenMadrigal Database into one file for use with this codebase.