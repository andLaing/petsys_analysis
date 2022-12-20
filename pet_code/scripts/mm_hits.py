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

from pet_code.src.io   import read_petsys_filebyfile, read_ymlmapping
from pet_code.src.io   import write_event_trace
from pet_code.src.util import calibrate_energies
from pet_code.src.util import filter_event_by_impacts_noneg
from pet_code.src.util import select_module


# Is there a better way?
def sort_and_write_mm(writer, sm_num):
    """
    Get info for each mini module in
    super module of interest and write
    to file.
    """
    def sort_and_write(evt):
        mm_dict = {}
        for hit in evt[sm_num]:
            try:
                mm_dict[hit[1]].append(hit)
            except KeyError:
                mm_dict[hit[1]] = [hit]
        for vals in mm_dict.values():
            writer(vals)
    return sort_and_write


if __name__ == '__main__':
    args = docopt(__doc__)
    conf = configparser.ConfigParser()
    conf.read(args['--conf'])

    map_file   = conf.get   ('mapping',   'map_file')
    control_sm = conf.getint('mapping', 'control_sm')
    valid_sm   = {1, 3}
    if control_sm not in valid_sm:
        print("Invalid control super module. Use 1 or 3")
        exit()
    control_indx = 0 if control_sm == 3 else 1
    file_list  = args['INFILE']

    time_ch, eng_ch, mm_map, centroid_map, _ = read_ymlmapping(map_file)

    min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
    evt_filter = filter_event_by_impacts_noneg(eng_ch, *min_chan)

    out_fldr = conf.get('output',  'out_dir', fallback='')
    out_name = conf.get('output', 'out_name', fallback='all_impacts')

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(time_ch, eng_ch, time_cal, eng_cal)

    mm_check = 0
    all_evt  = 0
    for fn in file_list:
        in_parts = os.path.normpath(fn).split(os.path)
        if out_fldr:
            out_dir = os.path.join(out_fldr, out_name)
        else:
            if in_parts[0] == '':
                in_parts[0] = os.sep
            out_dir = os.path.join(*in_parts[:-1], out_name)
        out_file = out_dir + in_parts[-1].replace('.ldat', '_NN.txt')

        # Need to protect from overwrite? Will add output folder when using docopt/config or both
        with open(out_file, 'w') as tout:
            sort_writer = sort_and_write_mm(write_event_trace(tout, centroid_map), control_indx)
            reader      = read_petsys_filebyfile(mm_map, evt_filter)
            for evt in reader():
                all_evt += 1
                sel_mods = tuple(map(select_module, cal_func(evt)))
                n_mm     = len(set(x[1] for x in sel_mods[control_indx]))
                if n_mm > 1:
                    mm_check += 1
                sort_writer(sel_mods)
    print("Proportion of events with multihit in sm (highest charge MM selected): ", mm_check / all_evt)

