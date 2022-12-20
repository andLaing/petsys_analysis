#!/usr/bin/env python3

"""Make a dataframe with mapping info (for 2 SM IMAS!) and save

Usage: make_map.py [-f NFEM] MAPYAML

Arguments:
    MAPYAML  File name with time and energy channel info.

Options:
    -s=NSM     Number of Super modules [default: 4]
    -f=NFEM    Number of channels per Supermodule [default: 256]
"""

import yaml

from docopt import docopt

import pandas as pd

from pet_code.src.util import slab_x, slab_y, slab_z


def echan_x(rc_num):
    mm_wrap = round(0.3 * (rc_num // 8), 2)
    return round(1.75 + mm_wrap + 3.2 * rc_num, 2)


def echan_y(sm, row):
    if sm == 0:
        return round(-25.9 * (0.5 + (3 - row % 4)), 2)
    return round(-25.9 * (0.5 + row % 4), 2)


def row_gen(nFEM, chan_per_mm, tchans, echans):
    mM_energyMapping = {1:1,  2:5,  3:9 ,  4:13,  5:2,  6:6,  7:10,  8:14,
                        9:3, 10:7, 11:11, 12:15, 13:4, 14:8, 15:12, 16:16}
    for i in (0, 2):
        for j, (tch, ech) in enumerate(zip(tchans, echans)):
            id = tch + i * nFEM
            mm = j // chan_per_mm + 1
            x  = slab_x(j // 32)
            y  = slab_y(j %  32, i)
            z  = slab_z(i)
            # Position for floodmap (review!)
            pp = round(1.6 + 3.2 * (j % 32), 2)
            yield id, 'TIME', mm, x, y, z, pp
            id = ech + i * nFEM
            mm = mM_energyMapping[mm]
            x  = echan_x(j % 32)
            y  = echan_y(i, j // 32)
            pp = round(1.6 + 3.2 * (31 - j % 32), 2)
            yield id, 'ENERGY', mm, x, y, z, pp


if __name__ == '__main__':
    args = docopt(__doc__)
    # nsuper = args['-s']
    nFEM = int(args['-f'])

    with open(args['MAPYAML']) as map_buffer:
        channel_map = yaml.safe_load(map_buffer)

    df = pd.DataFrame((row for row in row_gen(nFEM, 8, channel_map['time_channels'], channel_map['energy_channels'])),
                      columns=['id', 'type', 'minimodule', 'X', 'Y', 'Z', 'PLOTP'])
    df.to_feather('pet_code/test_data/twoSM_IMAS_map.feather')