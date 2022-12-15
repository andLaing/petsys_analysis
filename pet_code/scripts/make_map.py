#!/usr/bin/env python3

"""Make a dataframe with mapping info and save

Usage: make_map.py [-s NSM] [-f NFEM] [-t NTCHAN] [-e NECHAN] MAPYAML

Arguments:
    MAPYAML  File name with time and energy channel info.

Options:
    -s=NSM     Number of Super modules [default: 4]
    -f=NFEM    Number of channels per Supermodule [default: 256]
    -t=NTCHAN  Number of time channels per mini-module [default: 8]
    -e=NECHAN  Number of energy channels per mini-module [default: 8]
"""

import yaml

from docopt import docopt

import pandas as pd

from pet_code.src.util import slab_x, slab_y, slab_z


def row_gen(nsuper, nFEM, chan_per_mm, channels, chan_type):
    for i in range(nsuper):
        for j, chan in enumerate(channels):
            id = chan + i * nFEM
            mm = j // chan_per_mm
            x  = slab_x(j // )
        yield 


if __name__ == '__main__':
    args = docopt(__doc__)
    nsuper = args['-s']
    nFEM   = args['-f']
    ntchan = args['-t']

    with open(args['MAPYAML']) as map_buffer:
        channel_map = yaml.safe_load(map_buffer)

    time_df = pd.DataFrame()