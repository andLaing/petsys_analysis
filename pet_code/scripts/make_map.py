#!/usr/bin/env python3

"""Make a dataframe with mapping info for TBPET type SM and save

Usage: make_map.py [-f NFEM] [-g GEOM] [-o OUT] MAPYAML

Arguments:
    MAPYAML  File name with time and energy channel info.

Options:
    -f=NFEM  Number of channels per Supermodule [default: 256]
    -g=GEOM  Geometry: 2SM, 1ring [default: 1ring]
    -o=OUT   Path for output file [default: 1ring_map]
"""

import yaml

from docopt import docopt

import numpy  as np
import pandas as pd

from scipy.spatial.transform import Rotation as R

from pet_code.src.util import ChannelType
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
            # Make the recID the same as id for now
            yield id, 'TIME', i, mm, x, y, z, pp, id
            id = ech + i * nFEM
            mm = mM_energyMapping[mm]
            x  = echan_x(j % 32)
            y  = echan_y(i, j // 32)
            pp = round(1.6 + 3.2 * (31 - j % 32), 2)
            # Make the recID the same as id for now
            yield id, 'ENERGY', i, mm, x, y, z, pp, id


def channel_sm_coordinate(mm_rowcol, ch_indx, type):
    """
    mm_rowcol : Either row or col depending on type: TIME row, ENERGY col.
    ch_indx   : index along row/column
    """
    mm_spacing   = 0.3 * ((31 - ch_indx) // 8)
    local_fine   = round(1.75 + mm_spacing + 3.2 * (31 - ch_indx), 2)
    local_coarse = round(25.9 * (3.5 - mm_rowcol), 2)
    if type is ChannelType.TIME:
        # local y course, local_x fine
        return local_fine, local_coarse
    return local_coarse, local_fine


def sm_centre_pos():
    """
    Gives polar angle position (and r; fixed)
    for the centres of the SMs.
    Currently set for ring positions of 
    center inner face of TBPET supermodules.
    """
    SM_r  = 410.0
    # This shouldn't be hardwired!!
    # Note order Y, X for easy use with arctan2
    SM_yx = { 0: ( -53.5157,  406.4924),  1: (-156.9002,  378.7906),
              2: (-249.5922,  325.2749),  3: (-325.2749,  249.5922),
              4: (-378.7906,  156.9002),  5: (-406.4924,   53.5157),
              6: (-406.4924,  -53.5157),  7: (-378.7906, -156.9002),
              8: (-325.2749, -249.5922),  9: (-249.5922, -325.2749),
             10: (-156.9002, -378.7906), 11: ( -53.5157, -406.4924),
             12: (  53.5157, -406.4924), 13: ( 156.9002, -378.7906),
             14: ( 249.5922, -325.2749), 15: ( 325.2749, -249.5922),
             16: ( 378.7906, -156.9002), 17: ( 406.4924,  -53.5157),
             18: ( 406.4924,   53.5157), 19: ( 378.7906,  156.9002),
             20: ( 325.2749,  249.5922), 21: ( 249.5922,  325.2749),
             22: ( 156.9002,  378.7906), 23: (  53.5157,  406.4924)}
    def get_rtheta(SM):
        return SM_r, np.arctan2(*SM_yx[SM])
    return get_rtheta


def sm_gen(nFEM, chan_per_mm, tchans, echans, mm_emap):
    def _sm_gen(sm_no):
        for i, (tch, ech) in enumerate(zip(tchans, echans)):
            id = tch + sm_no * nFEM
            mm = i // chan_per_mm
            loc_x, loc_y = channel_sm_coordinate(i // 32, i % 32, ChannelType.TIME)
            yield id, 'TIME', mm, loc_x, loc_y
            id = ech + sm_no * nFEM
            mm = mm_emap[mm]
            loc_x, loc_y = channel_sm_coordinate(i // 32, i % 32, ChannelType.ENERGY)
            yield id, 'ENERGY', mm, loc_x, loc_y
    return _sm_gen


def single_ring(nFEM, chan_per_mm, tchans, echans):
    mM_energyMapping = {0:0,  1:4,  2:8 ,  3:12,  4:1,  5:5,  6:9 ,  7:13,
                        8:2,  9:6, 10:10, 11:14, 12:3, 13:7, 14:11, 15:15}
    sm_angle         = sm_centre_pos()
    superm_gen       = sm_gen(nFEM, chan_per_mm, tchans, echans, mM_energyMapping)
    # Hardwired, fix.
    SM_half_len      = 103.6 / 2
    coords           = ['X', 'Y', 'Z']
    def ring_gen():
        local_cols = ['id', 'type', 'minimodule', 'local_x', 'local_y']
        for sm in range(24):
            sm_local                = pd.DataFrame((ch for ch in superm_gen(sm)),
                                                    columns = local_cols        )
            sm_local['supermodule'] = sm

            sm_r, sm_ang     = sm_angle(sm)
            ## Translate to XYZ relative to SM centre at X = R, Y = Z = 0.
            sm_local['X']    =  sm_r
            sm_local['Y']    =  sm_local.local_x - SM_half_len
            sm_local['Z']    = -sm_local.local_y + SM_half_len
            ## Rotate to supermodule angular position.
            sm_rot           = R.from_euler('z', sm_ang)
            sm_local[coords] = sm_local[coords].apply(sm_rot.apply, axis=1,
                                                      result_type='broadcast')
            yield sm_local
    return pd.concat((sm for sm in ring_gen()), ignore_index=True)


if __name__ == '__main__':
    args    = docopt(__doc__)
    nFEM    = int(args['-f'])
    geom    =     args['-g']
    outname =     args['-o']

    if '.feather' not in outname:
        outname += '.feather'

    with open(args['MAPYAML']) as map_buffer:
        channel_map = yaml.safe_load(map_buffer)

    if geom == '2SM':
        df = pd.DataFrame((row for row in row_gen(nFEM, 8, channel_map['time_channels'], channel_map['energy_channels'])),
                        columns=['id', 'type', 'supermodule', 'minimodule', 'X', 'Y', 'Z', 'PLOTP', 'recID'])
    elif geom == '1ring':
        df = single_ring(nFEM, 8, channel_map['time_channels'], channel_map['energy_channels'])
    else:
        print('Geometry not recognised')
        exit()

    df.to_feather(outname)