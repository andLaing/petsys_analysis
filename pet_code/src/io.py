import struct
import yaml

import numpy as np

from itertools import chain, islice

from . util import slab_indx, slab_x, slab_y, slab_z


def read_petsys(mod_mapping, sm_filter=lambda x, y: True, singles=False):
    """
    Reader for petsys output for a list of input files.
    All files yielded in single generator.
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
        line_struct = line_struct[:6]
    def petsys_event(file_list):
        """
        file_list: List{String}
                   List of strings with the paths to the files of interest.
        """
        for fn in file_list:
            yield from _read_petsys_file(fn         ,
                                         line_struct,
                                         mod_mapping,
                                         sm_filter  ,
                                         singles    )
    return petsys_event


def read_petsys_filebyfile(file_name, mod_mapping, sm_filter=lambda x, y: True, singles=False):
    """
    Reader for petsys output for a list of input files.
    file_name  : String
                 The path to the file to be read.
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
        line_struct = line_struct[:6]
    def petsys_event():
        yield from _read_petsys_file(file_name  ,
                                     line_struct,
                                     mod_mapping,
                                     sm_filter  ,
                                     singles    )
    return petsys_event


def _read_petsys_file(file_name      ,
                      line_struct    ,
                      mod_mapping    ,
                      sm_filter      ,
                      singles = False):
    """
    Read all events from a single petsys
    file yielding those meeting sm_filter
    conditions.
    """
    evt_loop = singles_evt_loop if singles else coincidences_evt_loop
    with open(file_name, 'rb') as fbuff:
        b_iter = struct.iter_unpack(line_struct, fbuff.read())
        for first_line in b_iter:
            sm1, sm2 = evt_loop(first_line, b_iter, mod_mapping)
            if sm_filter(sm1, sm2):
                yield sm1, sm2


def singles_evt_loop(first_line, line_it, mod_mapping):
    """
    Loop through the lines for an event
    of singles data.
    Needs to be optimised/tested
    Should be for what PETSys calls 'grouped'
    which seems more like a PET single.
    """
    nlines = first_line[0]
    return list(map(unpack_supermodule                              ,
                    chain([first_line], islice(line_it, nlines - 1)),
                    [mod_mapping] * nlines                          )), []


def coincidences_evt_loop(first_line, line_it, mod_mapping):
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
            sm1.append(unpack_supermodule(evt[:5], mod_mapping))
            ch_sm1.add(evt[4])
        if evt[-1] not in ch_sm2:
            sm2.append(unpack_supermodule(evt[5:], mod_mapping))
            ch_sm2.add(evt[-1])
    return sm1, sm2


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

    This version for 2 super module setup.
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
    slab_positions   = {}
    ## A lot of hardwired numbers here, maybe not stable
    mM_energyMapping = {1:1,  2:5,  3:9 ,  4:13,  5:2,  6:6,  7:10,  8:14,
                        9:3, 10:7, 11:11, 12:15, 13:4, 14:8, 15:12, 16:16}
    FEM_num_ch = 256
    slab_num   =   1
    rc_num     =   0
    no_sm      =   4
    for sm in range(no_sm):
        mM_num = 1
        row    = 0
        for tch, ech in zip(channel_map["time_channels"], channel_map["energy_channels"]):
            absolut_tch = tch + sm * FEM_num_ch
            absolut_ech = ech + sm * FEM_num_ch
            ALLSM_time_ch  .add(absolut_tch)
            ALLSM_energy_ch.add(absolut_ech)

            mM_num_en = mM_energyMapping[mM_num]
            mM_mapping      [absolut_tch] = mM_num
            mM_mapping      [absolut_ech] = mM_num_en
            # The centroid mapping 'x, y' is set for floodmap display
            # So that the first module is in the top left. Doesn't
            # represent true positions. Should be reviewed.
            centroid_mapping[absolut_tch] = (0, round(1.6 + 3.2*(rc_num), 2))
            centroid_mapping[absolut_ech] = (1, round(1.6 + 3.2 * (31 - rc_num), 2))  #establish 0 reference at the botom left of the floodmap
            # slab_positions  [absolut_tch] = (slab_x(rc_num, sm), slab_y(row), slab_z(sm))
            slab_positions  [absolut_tch] = (slab_x(row), slab_y(rc_num, sm), slab_z(sm))

            rc_num += 1
            if slab_num%8 == 0:
                mM_num += 1
            if slab_num%32 == 0:
                rc_num = 0
                row   += 1
            slab_num += 1
    return ALLSM_time_ch, ALLSM_energy_ch, mM_mapping, centroid_mapping, slab_positions


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
            indx      = slab_indx(pos)
            channels[indx + 8 * en_t] = imp[3]
        file_buffer.write('\t'.join("{:.6f}".format(round(val, 6)) for val in channels))
        file_buffer.write('\t' + str(mm_trace[0][1]) + '\n')
    return write_minimod
        