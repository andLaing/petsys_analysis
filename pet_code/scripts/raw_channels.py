#!/usr/bin/env python3

"""Checks the channels that give data in a file and compares to expectation

Usage: raw_channels.py (--map MAPFILE) INPUT

Arguments:
    INPUT  Input file name (PETsys .ldat format)

Required:
    --map=MAPFILE  Name of the map file
"""

from docopt import docopt

import numpy as np

from pet_code.src.io import ChannelMap
from pet_code.src.io import read_petsys_filebyfile


if __name__ == '__main__':
    args       = docopt(__doc__)
    input_file = args['INPUT']
    map_file   = args['--map']

    ch_map = ChannelMap(map_file)

    reader   = read_petsys_filebyfile(ch_map.ch_type, singles='coinc' not in input_file)
    evt_negs = {}
    for evt in reader(input_file):
        for sm in evt:
            if sm:
                for imp in sm:
                    try:
                        evt_negs[imp[0]].append(imp[3] <= 0)
                    except KeyError:
                        evt_negs[imp[0]] = [imp[3] <= 0]
    for id in ch_map.mapping.index:
        if id not in evt_negs.keys():
            print(f'Channel {id} has no data. SM{ch_map.get_supermodule(id)}, mM{ch_map.get_minimodule(id)}.')
            continue
        prop_negative = np.count_nonzero(evt_negs[id]) / len(evt_negs[id])
        if prop_negative > 0.01:
            print(f'More than 1% ({round(prop_negative*100, 3)}%) of energy values are <= 0 for channel {id}. SM{ch_map.get_supermodule(id)}, mM{ch_map.get_minimodule(id)}.')
