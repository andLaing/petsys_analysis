#!/usr/bin/env python3

"""Save min-module energy information for NN training

Usage: mm_hits.py (--conf CONFFILE) INFILE ...

Arguments:
    INFILE  File(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt import docopt

from pet_code.src.filters import filter_event_by_impacts_noneg
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.io      import write_event_trace
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import select_module


if __name__ == '__main__':
    args = docopt(__doc__)
    conf = configparser.ConfigParser()
    conf.read(args['--conf'])

    map_file   = conf.get   ('mapping',   'map_file')
    control_sm = conf.getint('mapping', 'control_sm')
    valid_sm   = {0, 2}
    if control_sm not in valid_sm:
        print("Invalid control super module. Use 0 or 2")
        exit()
    control_indx = 0 if control_sm == 2 else 1
    file_list  = args['INFILE']

    chan_map = ChannelMap(map_file)

    min_chan   = conf.getint('filter', 'min_channels')
    evt_filter = filter_event_by_impacts_noneg(min_chan)

    out_fldr = conf.get('output',  'out_dir', fallback='')
    out_name = conf.get('output', 'out_name', fallback='all_impacts')

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal)

    for fn in file_list:
        in_parts = os.path.normpath(fn).split(os.sep)
        if out_fldr:
            out_dir = os.path.join(out_fldr, out_name)
        else:
            if in_parts[0] == '':
                in_parts[0] = os.sep
            out_dir = os.path.join(*in_parts[:-1], out_name)
        out_file = out_dir + in_parts[-1].replace('.ldat', '_NN.txt')

        # Need to protect from overwrite? Will add output folder when using docopt/config or both
        sm_map = chan_map.mapping[chan_map.mapping.supermodule == control_sm]
        mod_select = select_module(chan_map.get_minimodule)
        with open(out_file, 'w') as tout:
            writer = write_event_trace(tout, sm_map, chan_map.get_minimodule)
            reader = read_petsys_filebyfile(chan_map.ch_type, evt_filter)
            for evt in reader(fn):
                sel_mods = mod_select(cal_func(evt)[control_indx])
                writer(sel_mods)

