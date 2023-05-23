#!/usr/bin/env python3

"""Make a dataframe with mapping info for TBPET type SM and save

Usage: make_map.py [-f NFEM] [-g GEOM] [-o OUT] MAPYAML

Arguments:
    MAPYAML  File name with time and energy channel info.

Options:
    -f=NFEM  Number of channels per Supermodule [default: 256]
    -g=GEOM  Geometry: 2SM, 1ring, nring [default: 1ring]
    -o=OUT   Path for output file [default: 1ring_map]
"""

import yaml

from docopt import docopt
from typing import Callable, Iterator

import numpy  as np
import pandas as pd

from scipy.spatial.transform import Rotation as R

from pet_code.src.util import ChannelType
from pet_code.src.util import slab_x, slab_y, slab_z


# Global values for module placement in mm
class TbpetGeom:
    def __init__(self) -> None:
        self.sm_per_ring      = 24
        self.mm_spacing       =  0.205
        self.mm_edge          = 25.805
        self.slab_width       =  3.2
        self.mM_energyMapping = {0:0,  1:4,  2:8 ,  3:12,  4:1,  5:5,  6:9 ,  7:13,
                                 8:2,  9:6, 10:10, 11:14, 12:3, 13:7, 14:11, 15:15}
        self.sm_edge_x        = 4 * self.mm_edge
        self.sm_edge_y        = 4 * self.mm_edge


class BrainGeom:
    def __init__(self) -> None:
        self.sm_per_ring = 20
        self.mm_spacing  =  0.205
        self.mm_edge     = 25.805
        self.slab_width  =  3.2
        self.sm_edge_x   = 2 * self.mm_edge
        self.sm_edge_y   = 8 * self.mm_edge


def echan_x(rc_num: int) -> float:
    mm_wrap = round(0.3 * (rc_num // 8), 2)
    return round(1.75 + mm_wrap + 3.2 * rc_num, 2)


def echan_y(sm: int, row: int) -> float:
    if sm == 0:
        return round(-25.9 * (0.5 + (3 - row % 4)), 2)
    return round(-25.9 * (0.5 + row % 4), 2)


def brain_map(nFEM: int, chan_per_mm: int, tchans: list, echans: list) -> Iterator:
    chan_per_sec = 4 * chan_per_mm
    chan_per_col = 8 * chan_per_mm
    geom         = BrainGeom()
    for i in (0, 3):
        for j, (tch, ech) in enumerate(zip(tchans, echans)):
            id    = tch + i * nFEM
            mm    = j // chan_per_mm
            ## Will need to add corrections for spacing.
            half  =  j // chan_per_col
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
            loc_x = round(1.6 + 3.2 * tindx, 3)
            loc_y = round(geom.mm_edge * (0.5 + row), 3)
            z     = 0 if i == 0 else 10#dummy
            # Will need i dependent rotation for true global xyz
            yield id, 'TIME', i if i == 0 else 2, mm, loc_x, loc_y, loc_x, loc_y, z
            id    = ech + i * nFEM
            loc_x = round(geom.mm_edge * (0.5 + (j // chan_per_col)), 3)
            loc_y = round(1.6 + 3.2 * eindx, 3)
            yield id, 'ENERGY', i if i == 0 else 2, mm, loc_x, loc_y, loc_x, loc_y, z


def row_gen(nFEM: int, chan_per_mm: int, tchans: list, echans: list) -> Iterator:
    superm_gen = sm_gen(nFEM, chan_per_mm, tchans, echans, {0: [0, 0, 0], 2: [0, 0, 2]})
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


def channel_sm_coordinate(mm_rowcol: int        ,
                          ch_indx  : int        ,
                          chtype   : ChannelType,
                          geom     : TbpetGeom
                          ) -> tuple[float, float]:
    """
    mm_rowcol : Either row or col depending on type: TIME row, ENERGY col.
    ch_indx   : index along row/column
    """
    mm_shift     = geom.mm_spacing * ((31 - ch_indx) // 8)
    ch_start     = (geom.slab_width + geom.mm_spacing) / 2
    local_fine   = round(ch_start + mm_shift + geom.slab_width * (31 - ch_indx), 3)
    local_coarse = round(geom.mm_edge * (3.5 - mm_rowcol), 3)
    if chtype is ChannelType.TIME:
        # local y course, local_x fine
        return local_fine, local_coarse
    return local_coarse, local_fine


def sm_centre_pos(SM_r: float, SM_yx: dict) -> Callable:
    """
    Gives polar angle position (and r; fixed)
    for the centres of the SMs.
    Currently set for ring positions of
    center inner face of TBPET supermodules.
    SM_r  : Radius of ring
    SM_yx : YX values of the SM centres.
    """
    def get_rtheta(SM: int) -> tuple[float, float]:
        return SM_r, np.arctan2(*SM_yx[SM])
    return get_rtheta


def brain_sm_gen(nFEM         : int ,
                 chan_per_mm  : int ,
                 tchans       : list,
                 echans       : list,
                 sm_to_febport: dict
                 ) -> Callable:
    """
    Generate local coordinates with electronics ids for
    brain type supermodules.
    """
    chan_per_sec = 4 * chan_per_mm
    chan_per_col = 8 * chan_per_mm
    geom         = BrainGeom()
    def _sm_gen(sm_no: int) -> Iterator:
        portID, slaveID, febport = sm_to_febport[sm_no]
        sm_min_chan = 131072 * portID + 4096 * slaveID + nFEM * febport
        for i, (tch, ech) in enumerate(zip(tchans, echans)):
            id   = tch + sm_min_chan
            mm   = i // chan_per_mm
            ## Will need to add corrections for spacing.
            half =  i // chan_per_col
            if half == 1:
                tindx    = 15 - i % chan_per_mm
                half_sec = (i - chan_per_col) // chan_per_sec
                eindx    = 31 - i % chan_per_sec if half_sec == 0 else 63 - i % chan_per_sec
                if half_sec == 0:
                    row = 3 - i // chan_per_mm +  8
                else:
                    row = 7 - i // chan_per_mm + 12
            else:
                tindx    = i % chan_per_mm
                eindx    = i % chan_per_col
                row      = (i %  chan_per_col) // chan_per_mm
            loc_x = round(1.6 + 3.2 * tindx, 3)
            loc_y = round(geom.mm_edge * (0.5 + row), 3)
            # Will need i dependent rotation for true global xyz
            yield id, 'TIME', mm, loc_x, loc_y
            id    = ech + sm_min_chan
            loc_x = round(geom.mm_edge * (0.5 + (i // chan_per_col)), 3)
            loc_y = round(1.6 + 3.2 * eindx, 3)
            yield id, 'ENERGY', mm, loc_x, loc_y
    return _sm_gen


def sm_gen(nFEM         : int ,
           chan_per_mm  : int ,
           tchans       : list,
           echans       : list,
           sm_to_febport: dict
           ) -> Callable:
    """
    Generate local coordinates with electronics ids for
    tbpet type supermodules.
    nFEM          : Channels per Supermodule (FEM)
    chan_per_mm   : Channels per minimodule
    tchans        : Time channel ids at FEM level
    echans        : Energy channel ids at FEM level
    mm_emap       : time minimodule to energy minimodule
    sm_to_febport : Dict with [slaveID, FEBport] for each supermodule
    """
    geom = TbpetGeom()
    def _sm_gen(sm_no: int) -> Iterator:
        portID, slaveID, febport = sm_to_febport[sm_no]
        sm_min_chan = 131072 * portID + 4096 * slaveID + nFEM * febport
        for i, (tch, ech) in enumerate(zip(tchans, echans)):
            id = tch + sm_min_chan#sm_no * nFEM
            mm = i // chan_per_mm
            loc_x, loc_y = channel_sm_coordinate(i // 32, i % 32, ChannelType.TIME, geom)
            yield id, 'TIME', mm, loc_x, loc_y
            id = ech + sm_min_chan#sm_no * nFEM
            mm = geom.mM_energyMapping[mm]
            loc_x, loc_y = channel_sm_coordinate(i // 32, i % 32, ChannelType.ENERGY, geom)
            yield id, 'ENERGY', mm, loc_x, loc_y
    return _sm_gen


def local_translation(df         : pd.DataFrame,
                      sm_r       : float       ,
                      sm_half_len: float
                      ) -> np.ndarray:
    """
    Translate local coordinates to coordinates
    centred on supermodule.

    Assumes y coordinate needs to be inverted.
    """
    coords = np.vstack((np.full(df.shape[0], sm_r),
                         df.local_x - sm_half_len ,
                        -df.local_y + sm_half_len ))
    return coords.T


def single_ring(nFEM       : int                                ,
                chan_per_mm: int                                ,
                tchans     : list                               ,
                echans     : list                               ,
                ring_r     : float                              ,
                ring_yx    : dict                               ,
                sm_feb     : dict                               ,
                first_sm   : int                   =           0,
                sm_geom    : BrainGeom | TbpetGeom = TbpetGeom()
                ) -> Callable:
    sm_angle    = sm_centre_pos(ring_r, ring_yx)
    if isinstance(sm_geom, TbpetGeom):
        superm_gen = sm_gen(nFEM, chan_per_mm, tchans, echans, sm_feb)
        SM_half_len = sm_geom.sm_edge_y / 2
    else:
        superm_gen = brain_sm_gen(nFEM, chan_per_mm,
                                  tchans, echans, sm_feb)
        SM_half_len = sm_geom.sm_edge_y / 2 #need to know what axes used since not symmetric
    coords      = ['X', 'Y', 'Z']
    def ring_gen() -> Iterator:
        local_cols = ['id', 'type', 'minimodule', 'local_x', 'local_y']
        for sm in range(first_sm, first_sm + sm_geom.sm_per_ring):
            sm_local                = pd.DataFrame((ch for ch in superm_gen(sm)),
                                                    columns = local_cols        )
            sm_local['supermodule'] = sm

            sm_r, sm_ang     = sm_angle(sm % sm_geom.sm_per_ring)
            ## Translate to XYZ relative to SM centre at X = R, Y = Z = 0.
            sm_local[coords] = local_translation(sm_local, sm_r, SM_half_len)
            ## Rotate to supermodule angular position.
            sm_rot           = R.from_euler('z', sm_ang)
            sm_local[coords] = sm_local[coords].apply(sm_rot.apply, axis=1,
                                                      result_type='broadcast')
            yield sm_local
    return pd.concat((sm for sm in ring_gen()), ignore_index=True)


def n_rings(ring_pos   : list ,
            nFEM       : int  ,
            chan_per_mm: int  ,
            tchans     : dict ,
            echans     : dict ,
            ring_r     : float,
            ring_yx    : dict ,
            sm_feb     : dict
            ) -> pd.DataFrame:
    """
    Generate len(ring_pos) 24 tbpet sm rings.

    ring_pos : list_like
                Axial position of centre of the rings
    """
    geom = TbpetGeom()
    def ring_at_z(ring_no: int, ring_z: float) -> pd.DataFrame:
        df      = single_ring(nFEM  , chan_per_mm, tchans ,
                              echans, ring_r     , ring_yx,
                              sm_feb, ring_no * geom.sm_per_ring, geom)
        df['Z'] = df.Z + ring_z
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
        df = single_ring(nFEM                          ,
                         8                             ,
                         channel_map[  'time_channels'],
                         channel_map['energy_channels'],
                         channel_map[        'ring_r' ],
                         channel_map[        'ring_yx'],
                         channel_map[     'sm_feb_map'])
    elif geom == 'nring':
        df = n_rings(channel_map[         'ring_z'],
                     nFEM                          ,
                     8                             ,
                     channel_map[  'time_channels'],
                     channel_map['energy_channels'],
                     channel_map[        'ring_r' ],
                     channel_map[        'ring_yx'],
                     channel_map[     'sm_feb_map'])
    elif geom == 'brain':
        df = pd.DataFrame(brain_map(nFEM, 8, channel_map['time_channels'], channel_map['energy_channels']),
                          columns=['id', 'type', 'supermodule', 'minimodule', 'local_x', 'local_y', 'X', 'Y', 'Z'])
    elif geom == 'brainring':
        df = single_ring(nFEM                          ,
                         8                             ,
                         channel_map[  'time_channels'],
                         channel_map['energy_channels'],
                         channel_map[        'ring_r' ],
                         channel_map[        'ring_yx'],
                         channel_map[     'sm_feb_map'],
                         sm_per_ring = 20              ,
                         sm_geom     = BrainGeom()     )
    else:
        print('Geometry not recognised')
        exit()

    df.to_feather(outname)
