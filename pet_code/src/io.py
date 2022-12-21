import struct
import yaml

import numpy  as np
import pandas as pd

from itertools import chain, islice, repeat

from . util    import ChannelType
from . util    import slab_indx, slab_x, slab_y, slab_z
from . io_util import coinc_evt_loop

from typing import List, Tuple, Union # Once upgraded to python 3.9 not necessary


def read_petsys(type_func, sm_filter=lambda x, y: True, singles=False):
    """
    Reader for petsys output for a list of input files.
    All files yielded in single generator.
    type_func : Lookup function for the channel id
                to ChannelType.
    singles   : Is the file singles mode? default False.
    sm_filter : Function taking a tuple of sm to filter the module data.
    returns
    petsys_event: Fn, loops over input file list and yields
                      event information.
    """
    def petsys_event(file_list):
        """
        file_list: List{String}
                   List of strings with the paths to the files of interest.
        """
        for fn in file_list:
            yield from _read_petsys_file(fn, type_func, sm_filter, singles)
    return petsys_event


def read_petsys_filebyfile(type_func, sm_filter=lambda x, y: True, singles=False):
    """
    Reader for petsys output for a list of input files.
    type_func : Lookup function for the channel id
                to ChannelType.
    singles   : Is the file singles mode? default False.
    sm_filter : Function taking a tuple of smto filter the module data.
    returns
    petsys_event: Fn, loops over input file list and yields
                      event information.
    """
    def petsys_event(file_name):
        """
        Read a single file:
        file_name  : String
                     The path to the file to be read.
        """
        yield from _read_petsys_file(file_name, type_func, sm_filter, singles)
    return petsys_event


def read_petsys_singles(file_name, type_func):
    """
    Read a petsys singles mode file which
    contains only channel by channel time
    and energy info with no impact (singles)
    nor coincidence grouping.
    file_name : String
                ldat file name with petsys singles
    type_func : Lookup function for the channel id
                to ChannelType.
    returns a generator for line info [id, mm, tstp, eng]
    """
    line_struct = '<qfi'
    def petsys_event():
        with open(file_name, 'rb') as fbuff:
            for line in struct.iter_unpack(line_struct, fbuff.read()):
                yield line[2], type_func(line[2]), line[0], line[1]
    return petsys_event


def _read_petsys_file(file_name, type_func, sm_filter, singles=False):
    """
    Read all events from a single petsys
    file yielding those meeting sm_filter
    conditions.
    """
    line_struct = '<BBqfi'         if singles else '<BBqfiBBqfi'
    evt_loop    = singles_evt_loop if singles else coinc_evt_loop
    with open(file_name, 'rb') as fbuff:
        b_iter = struct.iter_unpack(line_struct, fbuff.read())
        for first_line in b_iter:
            sm1, sm2 = evt_loop(first_line, b_iter, type_func)
            if sm_filter(sm1, sm2):
                yield sm1, sm2


def singles_evt_loop(first_line, line_it, type_func):
    """
    Loop through the lines for an event
    of singles data.
    Needs to be optimised/tested
    Should be for what PETSys calls 'grouped'
    which seems more like a PET single.
    """
    nlines = first_line[0] - 1
    return list(map(unpack_supermodule                          ,
                    chain([first_line], islice(line_it, nlines)),
                    repeat(type_func)                           )), []


def coincidences_evt_loop(first_line, line_it, type_func):
    """
    Loop through the lines for an event
    of coincidence data.
    """
    sm1    = []
    sm2    = []
    ch_sm1 = set()
    ch_sm2 = set()
    nlines = first_line[0] + first_line[5] - 2
    for evt in chain([first_line], islice(line_it, nlines)):
        if evt[4] not in ch_sm1:
            sm1.append(unpack_supermodule(evt[:5], type_func))
            ch_sm1.add(evt[4])
        if evt[-1] not in ch_sm2:
            sm2.append(unpack_supermodule(evt[5:], type_func))
            ch_sm2.add(evt[-1])
    return sm1, sm2


def unpack_supermodule(sm_info, type_func):
    """
    For a super module readout line give the
    channel, module, time and energy info.
    """
    id      =       sm_info[4]
    ch_type = type_func(id)
    tstp    =       sm_info[2]
    eng     = float(sm_info[3])
    return [id, ch_type, tstp, eng]


def _read_yaml_file(mapping_file):
    """
    Read and return yaml mapping file.
    """
    try:
        with open(mapping_file) as map_buffer:
            channel_map = yaml.safe_load(map_buffer)
    except yaml.YAMLError:
        raise RuntimeError('Mapping file not readable.')
    if type(channel_map) is not dict or "time_channels" not in channel_map.keys()\
        or "energy_channels" not in channel_map.keys():
        raise RuntimeError('Mapping file not correct format.')
    return channel_map


def read_ymlmapping(mapping_file):
    """
    Read channel mapping information from
    yaml file.

    This version for 2 super module setup for IMAS.
    """
    channel_map = _read_yaml_file(mapping_file)

    ALLSM_time_ch    = set()
    ALLSM_energy_ch  = set()
    mM_mapping       = {}
    centroid_mapping = {}
    slab_positions   = {}
    ## A lot of hardwired numbers here, maybe not stable
    mM_energyMapping = {1:1,  2:5,  3:9 ,  4:13,  5:2,  6:6,  7:10,  8:14,
                        9:3, 10:7, 11:11, 12:15, 13:4, 14:8, 15:12, 16:16}
    FEM_num_ch = 256
    no_sm      =   4
    for sm in range(no_sm):
        for i, (tch, ech) in enumerate(zip(channel_map["time_channels"], channel_map["energy_channels"])):
            absolut_tch = tch + sm * FEM_num_ch
            absolut_ech = ech + sm * FEM_num_ch
            ALLSM_time_ch  .add(absolut_tch)
            ALLSM_energy_ch.add(absolut_ech)

            mM_num    = i // 8 + 1
            mM_num_en = mM_energyMapping[mM_num]
            mM_mapping      [absolut_tch] = mM_num
            mM_mapping      [absolut_ech] = mM_num_en
            # The centroid mapping 'x, y' is set for floodmap display
            # So that the first module is in the top left. Doesn't
            # represent true positions. Should be reviewed.
            centroid_mapping[absolut_tch] = (0, round(1.6 + 3.2 * (     i % 32), 2))
            centroid_mapping[absolut_ech] = (1, round(1.6 + 3.2 * (31 - i % 32), 2))  #establish 0 reference at the botom left of the floodmap
            # slab_positions  [absolut_tch] = (slab_x(rc_num, sm), slab_y(row), slab_z(sm))
            slab_positions  [absolut_tch] = (slab_x(i // 32), slab_y(i % 32, sm), slab_z(sm))

    return ALLSM_time_ch, ALLSM_energy_ch, mM_mapping, centroid_mapping, slab_positions


def read_ymlmapping_brain(mapping_file):
    """
    Read channel mapping information from
    yaml file.

    This version for 2 super module setup for IBRAIN.
    """
    channel_map = _read_yaml_file(mapping_file)

    ALLSM_time_ch    = set()
    ALLSM_energy_ch  = set()
    mM_mapping       = {}
    centroid_mapping = {}
    slab_positions   = {}

    FEM_num_ch = 256
    no_sm      =   4
    for sm in range(no_sm):
        mM_num   = 1
        slab_num = 1
        half     = 0
        r_num    = 0
        c_num    = 0
        for tch, ech in zip(channel_map["time_channels"], channel_map["energy_channels"]):
            absolut_tch = tch + sm * FEM_num_ch
            absolut_ech = ech + sm * FEM_num_ch
            ALLSM_time_ch  .append(absolut_tch)
            ALLSM_energy_ch.append(absolut_ech)

            mM_mapping[absolut_tch] = mM_num            
            mM_mapping[absolut_ech] = mM_num
            if   half == 0:
                centroid_mapping[absolut_tch] = (0, round(1.6 + 3.2 *       c_num , 2))
                centroid_mapping[absolut_ech] = (1, round(1.6 + 3.2 *       r_num , 2))
            elif half == 1:
                centroid_mapping[absolut_tch] = (0, round(1.6 + 3.2 * (15 - c_num), 2))
                centroid_mapping[absolut_ech] = (1, round(1.6 + 3.2 * (63 - r_num), 2))
            r_num += 1
            c_num += 1
            if slab_num %  8 == 0:
                mM_num += 1  
                c_num   = 0              
            if slab_num % 64 == 0:
                half  = 1
                r_num = 0
            slab_num += 1
            ## Need to add physical positions!
    return ALLSM_time_ch, ALLSM_energy_ch, mM_mapping, centroid_mapping, slab_positions


class ChannelMap:
    def __init__(self, map_file: str, ch_fem: int = 256) -> None:
        """
        Initialize Channel map type reading from feather
        mapping file with optional setting of channels
        per FEM.
        """
        self.mapping    = pd.read_feather(map_file).set_index('id')
        if 'gain' not in self.mapping.columns:
            ## Uncalibrated map.
            self.mapping['gain'] = 1.0
        self.ch_per_fem = ch_fem

    def get_channel_type(self, id: int) -> ChannelType:
        return ChannelType[self.mapping.at[id, 'type']]

    def get_supermodule(self, id: int) -> int:
        return id // self.ch_per_fem

    def get_minimodule(self, id: int) -> int:
        return self.mapping.at[id, 'minimodule']

    def get_minimodule_channels(self, sm: int, mm: int) -> np.ndarray:
        sm_mask = (self.mapping.index >=  sm      * self.ch_per_fem) &\
                  (self.mapping.index <  (sm + 1) * self.ch_per_fem)
        return self.mapping.index[sm_mask & (self.mapping.minimodule == mm)].values

    def get_channel_gain(self, id: int) -> float:
        return self.mapping.at[id, 'gain']

    def get_gains(self, ids: Union[List, Tuple, np.ndarray]) -> np.ndarray:
        return self.mapping.loc[ids, 'gain'].values

    def get_plot_position(self, id: int) -> float:
        """
        Pseudo position for floodmap plotting.
        """
        return self.mapping.at[id, 'PLOTP']

    def get_channel_position(self, id: int) -> np.ndarray:
        return self.mapping.loc[id, ['X', 'Y', 'Z']].values.astype('float')


def write_event_trace(file_buffer, centroid_map, mm_map):
    """
    Writer for text output of mini-module
    information as tab separated list of:
    8 * time channels 8 * energy channels, module number
    """
    def write_minimod(mm_trace):
        channels = np.zeros(16)
        for imp in mm_trace:
            en_t, pos = centroid_map[imp[0]]
            indx      = slab_indx(pos)
            channels[indx + 8 * en_t] = imp[3]
        file_buffer.write('\t'.join("{:.6f}".format(round(val, 6)) for val in channels))
        file_buffer.write('\t' + str(mm_map(mm_trace[0][0])) + '\n')
    return write_minimod
        