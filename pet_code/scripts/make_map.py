#!/usr/bin/env python3

"""Make a dataframe with mapping info for TBPET type SM and save

Usage: make_map.py [-f NFEM] [-g GEOM] [-c CONF] [-o OUT] MAPYAML

Arguments:
    MAPYAML  File name with time and energy channel info.

Options:
    -f=NFEM  Number of channels per Supermodule [default: 256]
    -g=GEOM  Geometry: 2SM, 1ring, nring [default: 1ring]
    -c=CONF  YAML with ring z position info (only used with GEOM=nring)
    -o=OUT   Path for output file [default: 1ring_map]
"""

import yaml

from docopt import docopt

import numpy  as np
import pandas as pd

from scipy.spatial.transform import Rotation as R

from pet_code.src.util import ChannelType
from pet_code.src.util import slab_x, slab_y, slab_z


# Global values for module placement in mm
mm_spacing       =  0.205
mm_edge          = 25.805
slab_width       =  3.2
mM_energyMapping = {0:0,  1:4,  2:8 ,  3:12,  4:1,  5:5,  6:9 ,  7:13,
                    8:2,  9:6, 10:10, 11:14, 12:3, 13:7, 14:11, 15:15}


def echan_x(rc_num):
    mm_wrap = round(0.3 * (rc_num // 8), 2)
    return round(1.75 + mm_wrap + 3.2 * rc_num, 2)


def echan_y(sm, row):
    if sm == 0:
        return round(-25.9 * (0.5 + (3 - row % 4)), 2)
    return round(-25.9 * (0.5 + row % 4), 2)


def brain_map(nFEM, chan_per_mm, tchans, echans):
    chan_per_sec = 4 * chan_per_mm
    chan_per_col = 8 * chan_per_mm
    for i in (0, 2):
        for j, (tch, ech) in enumerate(zip(tchans, echans)):
            id    = tch + i * nFEM
            mm    = j // chan_per_mm
            ## Will need to add corrections for spacing.
            half  =  j // chan_per_col
            # row   = (j %  chan_per_col) // chan_per_mm
            if half == 1:
                tindx    = 15 - j % chan_per_mm
                half_sec = (j - chan_per_col) // chan_per_sec
                eindx    = 31 - j %  chan_per_sec if half_sec == 0 else 63 - j %  chan_per_sec
                if half_sec == 0:
                    row = 3 - j // chan_per_mm +  8
                else:
                    row = 7 - j // chan_per_mm + 12
            else:
                tindx    = j % chan_per_mm
                eindx    = j % chan_per_col
                row      = (j %  chan_per_col) // chan_per_mm
            # indx  = j % chan_per_mm if half == 0 else 15 - j % chan_per_mm
            loc_x = round(1.6 + 3.2 * tindx, 3)
            loc_y = round(mm_edge * (0.5 + row), 3)
            z     = 0 if i == 0 else 10#dummy
            # Will need i dependent rotation for true global xyz
            yield id, 'TIME', i, mm, loc_x, loc_y, loc_x, loc_y, z
            id    = ech + i * nFEM
            loc_x = round(mm_edge * (0.5 + (j // chan_per_col)), 3)
            # if half == 1:
            #     half_sec = (j - chan_per_col) // chan_per_sec
            #     indx = 31 - j % chan_per_sec if half_sec == 0 else 63 - j % chan_per_sec
            # else:
            #     indx = j % chan_per_col
            loc_y = round(1.6 + 3.2 * eindx, 3)
            yield id, 'ENERGY', i, mm, loc_x, loc_y, loc_x, loc_y, z


def row_gen(nFEM, chan_per_mm, tchans, echans):
    superm_gen = sm_gen(nFEM, chan_per_mm, tchans, echans, mM_energyMapping)
    for i in (0, 2):
        for j, (id, typ, mm, loc_x, loc_y) in enumerate(superm_gen(i)):
            if typ == 'TIME':
                ind = j // 2
                x = slab_x(ind // 32)
                y = slab_y(ind %  32, i)
            else:
                ind = j // 2
                x = echan_x(ind % 32)
                y = echan_y(i, ind // 32)
            z  = slab_z(i)
            yield id, typ, i, mm, loc_x, loc_y, x, y, z


def channel_sm_coordinate(mm_rowcol, ch_indx, type):
    """
    mm_rowcol : Either row or col depending on type: TIME row, ENERGY col.
    ch_indx   : index along row/column
    """
    mm_shift     = mm_spacing * ((31 - ch_indx) // 8)
    ch_start     = (slab_width + mm_spacing) / 2
    local_fine   = round(ch_start + mm_shift + slab_width * (31 - ch_indx), 3)
    local_coarse = round(mm_edge * (3.5 - mm_rowcol), 3)
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


def local_translation(df, sm_r, sm_half_len):
    """
    Translate local coordinates to coordinates
    centred on supermodule.

    Assumes y coordinate needs to be inverted.
    """
    coords = np.vstack((np.full(df.shape[0], sm_r),
                         df.local_x - sm_half_len ,
                        -df.local_y + sm_half_len ))
    return coords.T


def single_ring(nFEM, chan_per_mm, tchans, echans):
    sm_angle         = sm_centre_pos()
    superm_gen       = sm_gen(nFEM, chan_per_mm, tchans, echans, mM_energyMapping)
    # Hardwired, fix.
    SM_half_len      = mm_edge * 2
    coords           = ['X', 'Y', 'Z']
    def ring_gen():
        local_cols = ['id', 'type', 'minimodule', 'local_x', 'local_y']
        for sm in range(24):
            sm_local                = pd.DataFrame((ch for ch in superm_gen(sm)),
                                                    columns = local_cols        )
            sm_local['supermodule'] = sm

            sm_r, sm_ang     = sm_angle(sm)
            ## Translate to XYZ relative to SM centre at X = R, Y = Z = 0.
            sm_local[coords] = local_translation(sm_local, sm_r, SM_half_len)
            ## Rotate to supermodule angular position.
            sm_rot           = R.from_euler('z', sm_ang)
            sm_local[coords] = sm_local[coords].apply(sm_rot.apply, axis=1,
                                                      result_type='broadcast')
            yield sm_local
    return pd.concat((sm for sm in ring_gen()), ignore_index=True)


def n_rings(ring_pos, nFEM, chan_per_mm, tchans, echans):
    """
    Generate len(ring_pos) 24 sm rings.

    ring_pos : list_like
                Axial position of centre of the rings
    """
    sm_per_ring  = 24
    ids_per_ring = sm_per_ring * nFEM
    def ring_at_z(ring_no, ring_z):
        sm_correction     = ring_no * sm_per_ring
        id_correction     = ring_no * ids_per_ring
        df                = single_ring(nFEM, chan_per_mm, tchans, echans)
        df['supermodule'] = df.supermodule + sm_correction
        df['id']          = df.id          + id_correction
        df['Z']           = df.Z           + ring_z
        return df

    return pd.concat((ring_at_z(i, rngz) for i, rngz in enumerate(ring_pos)), ignore_index=True)


if __name__ == '__main__':
    args    = docopt(__doc__)
    nFEM    = int(args['-f'])
    geom    =     args['-g']
    outname =     args['-o']

    if '.feather' not in outname:
        outname += '.feather'

    with open(args['MAPYAML']) as map_buffer:
        channel_map = yaml.safe_load(map_buffer)

    if   geom == '2SM'  :
        df = pd.DataFrame(row_gen(nFEM, 8, channel_map['time_channels'], channel_map['energy_channels']),
                          columns=['id', 'type', 'supermodule', 'minimodule', 'local_x', 'local_y', 'X', 'Y', 'Z'])
    elif geom == '1ring':
        df = single_ring(nFEM, 8, channel_map['time_channels'], channel_map['energy_channels'])
    elif geom == 'nring':
        with open(args['-c']) as ringZ:
            ring_z = yaml.safe_load(ringZ)
        df = n_rings(ring_z['Z'], nFEM, 8, channel_map['time_channels'], channel_map['energy_channels'])
    elif geom == 'brain':
        df = pd.DataFrame(brain_map(nFEM, 8, channel_map['time_channels'], channel_map['energy_channels']),
                          columns=['id', 'type', 'supermodule', 'minimodule', 'local_x', 'local_y', 'X', 'Y', 'Z'])
    else:
        print('Geometry not recognised')
        exit()

    df.to_feather(outname)