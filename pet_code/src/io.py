import struct
import yaml

import numpy as np

from itertools import islice

from . util import slab_position


def read_petsys(mod_mapping, sm_filter=lambda x: True, singles=False):
    """
    Reader for petsys output.
    mod_mapping: Lookup table for the channel id
                 to mini module numbering.
    energy_ch  : energy channels.
    singles    : Is the file singles mode? default False.
    sm_filter  : Function taking a tuple of smto filter the module data.
    returns
    petsys_event: Fn, loops over input file list and yields
                      event information.
    """
    line_struct = '<BBqfiBBqfi'
    if singles:
        line_Struct = line_struct[:6]
    def petsys_event(file_list):

        for fn in file_list:
            with open(fn, 'rb') as fbuff:
                b_iter = struct.iter_unpack(line_struct, fbuff.read())
                for first_line in b_iter:
                    sm1 = []
                    sm2 = []
                    last_ch = [-99, -99]
                    evt_lines = first_line[0] + first_line[5] - 2
                    for evt in [first_line] + list(islice(b_iter, evt_lines)):
                        if evt[4] != last_ch[0]:
                            sm1.append(unpack_supermodule(evt[:5], mod_mapping))
                            last_ch[0] = evt[4]
                        if not singles:
                            if evt[-1] != last_ch[1]:
                                sm2.append(unpack_supermodule(evt[5:], mod_mapping))
                                last_ch[1] = evt[-1]
                    if sm_filter(sm1, sm2):
                        yield sm1, sm2
    return petsys_event


def unpack_supermodule(sm_info, mod_mapping):
    """
    For a super module readout line give the
    channel, module, time and energy info.
    """
    id       =       sm_info[4]
    mini_mod = mod_mapping[id]
    tstp     =       sm_info[2]
    eng      = float(sm_info[3])
    return [id, mini_mod, tstp, eng]


def read_ymlmapping(mapping_file):
    """
    Read channel mapping information from
    yaml file.
    """
    try:
        with open(mapping_file) as map_buffer:
            channel_map = yaml.safe_load(map_buffer)
    except yaml.YAMLError:
        raise RuntimeError('Mapping file not readable.')
    if type(channel_map) is not dict or "time_channels" not in channel_map.keys()\
        or "energy_channels" not in channel_map.keys():
        raise RuntimeError('Mapping file not correct format.')

    ALLSM_time_ch    = set()
    ALLSM_energy_ch  = set()
    mM_mapping       = {}
    centroid_mapping = {}
    ## A lot of hardwired numbers here, maybe not stable
    mM_energyMapping = {1:1,  2:5,  3:9 ,  4:13,  5:2,  6:6,  7:10,  8:14,
                        9:3, 10:7, 11:11, 12:15, 13:4, 14:8, 15:12, 16:16}
    FEM_num_ch = 256
    slab_num   =   1
    rc_num     =   0
    no_sm      =   4
    for sm in range(no_sm):
        mM_num = 1
        for tch, ech in zip(channel_map["time_channels"], channel_map["energy_channels"]):
            absolut_tch = tch + sm * FEM_num_ch
            absolut_ech = ech + sm * FEM_num_ch
            ALLSM_time_ch  .add(absolut_tch)
            ALLSM_energy_ch.add(absolut_ech)

            mM_num_en = mM_energyMapping[mM_num]
            mM_mapping      [absolut_tch] = mM_num
            mM_mapping      [absolut_ech] = mM_num_en
            centroid_mapping[absolut_tch] = (0, round(1.6 + 3.2 *     rc_num , 2))
            centroid_mapping[absolut_ech] = (1, round(1.6 + 3.2 * (31-rc_num), 2))  #establish 0 reference at the botom left of the floodmap

            rc_num += 1
            if slab_num%8 == 0:
                mM_num += 1
            if slab_num%32 == 0:
                rc_num = 0
            slab_num += 1
    return ALLSM_time_ch, ALLSM_energy_ch, mM_mapping, centroid_mapping


def write_event_trace(file_buffer, centroid_map):
    """
    Writer for text output of mini-module
    information as tab separated list of:
    8 * time channels 8 * energy channels, module number
    """
    def write_minimod(mm_trace):
        channels = np.zeros(16)
        for imp in mm_trace:
            en_t, pos = centroid_map[imp[0]]
            indx      = slab_position(pos)
            channels[indx + 8 * en_t] = imp[3]
        file_buffer.write('\t'.join("{:.6f}".format(round(val, 6)) for val in channels))
        file_buffer.write('\t' + str(mm_trace[0][1]) + '\n')
    return write_minimod
        