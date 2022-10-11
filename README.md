# petsys_analysis

Code for the analysis of data from the PETSys ASIC.
Provides functions for decoding the data coming from
TBPET.

## Requirements
The code has the following minimum requirements:
 - python 3.8
 - scipy 1.7.1
 - numpy 1.21.2
 - matplotlib 3.5
 - pyyaml 6.0
 - natsort 7.1
 - pytest 6.2.5
 - pandas 1.3.5

 A script for environment set-up using `miniconda` is provided: `make_condaENV.sh` which currently assumes that miniconda is already installed in the `$HOME` folder.

 ## Structure
 The main code elements are structured as follows
 ```
 |- pet_code
 |  |- scripts/
 |  |   |- flood_maps.py -- Basic script to produce floodmaps
 |  |   |- mm_hits.py -- Text output of coincidence hits per mini module
 |  |- src/
 |  |   |- fits.py -- functions for distribution fitting
 |  |   |- io.py -- input output functions
 |  |   |- plots.py -- plotting functions
 |  |   |- util.py -- utilities including event filters.
 ```
 Tests are provided for the functions and, assuming all requirements are met and the environment is activated, can be run with `pytest -v`
