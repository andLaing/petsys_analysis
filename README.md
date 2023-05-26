# petsys_analysis

Code for the analysis of data from the PETSys ASIC.
Provides functions for decoding the data coming from
TBPET.

## Requirements
The code has the following minimum requirements:
 - python 3.10
 - scipy 1.10.0
 - numpy 1.23.5
 - matplotlib 3.7.1
 - pyyaml 6.0
 - pytest 7.1.2
 - pandas 1.5.3
 - docopt 0.6.2
 - configparser 5.0.2
 - cython 0.29.32
 - pyarrow 8.0.0

 A script for environment set-up using `miniconda` is provided for Unix systems:
 `make_condaENV.sh` which checks if conda installed in the `$HOME` folder. Running with `conda` already
 activated will also create or update the environment. If no miniconda is found the user will be prompted
 to either indicate a miniconda/anaconda installation or to allow the script to download and install
 miniconda.

 For windows the file `windEnv-crystal_cal.yml` is provided but installation of anaconda/miniconda is
 left to the user at the moment.

 ## Structure
 The main code elements are structured as follows
 ```
 |- pet_code
 |  |- scripts/
 |  |   |- cal_monitor.py -- Apply a given energy calibration and plot results.
 |  |   |- checks_channels.py -- Check calibration for expected channels.
 |  |   |- ctr.py -- CTR monitor script.
 |  |   |- flood_maps.py -- Basic script to produce supermodule floodmaps.
 |  |   |- grp_channel_specs.py -- Get energy peak position for each channel (calibration/equalization).
 |  |   |- make_listmode.py -- Make a selected LM binary from a PETsys output file.
 |  |   |- make_map.py -- Provides tools to generate a map file for a setup.
 |  |   |- mm_hits.py -- Text output of coincidence hits per mini module.
 |  |   |- negatives.py -- Checks a PETsys file for negative energies.
 |  |   |- raw_channels.py -- Checks a raw ldat file for expected channels and negatives.
 |  |   |- skew_calc.py -- Estimate skew iterating over reference channel data.
 |  |   |- slab_spectra.py -- Make energy and time difference spectra for slabs.
 |  |   |- thresholds.py -- Generate a PETsys threshold file based on relative calibration.
 |  |- src/
 |  |   |- filters.py -- Functions for filtering of PETsys events.
 |  |   |- fits.py -- Functions for distribution fitting.
 |  |   |- io.py -- Input output functions.
 |  |   |- plots.py -- Plotting functions.
 |  |   |- util.py -- Utilities.
 ```
 Tests are provided for the functions and, assuming all requirements are met and the environment is activated, can be run with `pytest -v`
