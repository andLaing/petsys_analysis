#!/usr/bin/env python3

"""Calculate and save mini-module level energy spectra and floodmaps

Usage: flood_maps.py (--conf CONFFILE) INFILE

Arguments:
    INFILE  File name to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt    import docopt
from itertools import repeat

from pet_code.src.io    import ChannelMap
from pet_code.src.io    import read_petsys_filebyfile
from pet_code.src.io    import read_ymlmapping
from pet_code.src.plots import mm_energy_spectra
from pet_code.src.util  import calibrate_energies
from pet_code.src.util  import centroid_calculation
from pet_code.src.util  import filter_event_by_impacts
from pet_code.src.util  import filter_event_by_impacts_noneg
from pet_code.src.util  import filter_impacts_one_minimod
from pet_code.src.util  import mm_energy_centroids
from pet_code.src.util  import select_module


import time
if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])
    
    start    = time.time()
    map_file = conf.get('mapping', 'map_file')#'pet_code/test_data/SM_mapping_corrected.yaml' # shouldn't be hardwired
    infile   = args['INFILE']

    # time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    chan_map  = ChannelMap(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    # Should improve with an enum or something
    if 'Impacts'  in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_event_by_impacts(*min_chan)
    elif 'OneMod' in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_impacts_one_minimod(*min_chan, chan_map.get_minimodule)
    elif 'NoNeg'  in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_event_by_impacts_noneg(*min_chan)
    else:
        print('No valid filter found, fallback to 4, 4 minimum energy channels')
        evt_select = filter_event_by_impacts(4, 4)

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal)

    pet_reader      = read_petsys_filebyfile(chan_map.ch_type, evt_select)
    filtered_events = list(map(cal_func, pet_reader(infile)))
    end_r           = time.time()
    print("Time enlapsed reading: {} s".format(int(end_r - start)))
    print("length check: ", len(filtered_events))
    ## Should we be filtering the events with multiple mini-modules in one sm?
    c_calc = centroid_calculation(chan_map.get_plot_position)
    # ## Must be a better way but...
    max_sel   = select_module(chan_map.get_minimodule) if conf.getboolean('filter', 'sel_max_mm') else lambda x: x
    mod_dicts = mm_energy_centroids(filtered_events, c_calc, chan_map.get_minimodule, mod_sel=max_sel)

    out_dir    = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nsigma = conf.getint('output', 'nsigma', fallback=2)
    out_base   = os.path.join(out_dir, conf.get('output', 'out_file', fallback=infile.split('/')[-1]))#'test_floods/' + file_list[0].split('/')[-1]
    photo_peak = list(map(mm_energy_spectra, mod_dicts, [1, 2], repeat(out_base), repeat(100), repeat((0, 300)), repeat(nsigma)))
    end_p = time.time()
    print("Time enlapsed plotting: {} s".format(int(end_p - end_r)))
            


