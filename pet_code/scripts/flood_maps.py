#!/usr/bin/env python3

"""Calculate and save mini-module level energy spectra and floodmaps

Usage: flood_maps.py (--conf CONFFILE) INFILES ...

Arguments:
    INFILES  File name(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt    import docopt
from itertools import repeat

from pet_code.src.filters  import filter_event_by_impacts
from pet_code.src.filters  import filter_event_by_impacts_noneg
from pet_code.src.filters  import filter_impacts_one_minimod
from pet_code.src.io       import ChannelMap
from pet_code.src.io       import read_petsys_filebyfile
from pet_code.src.io       import read_ymlmapping
from pet_code.src.plots    import mm_energy_spectra
from pet_code.src.util     import calibrate_energies
from pet_code.src.util     import centroid_calculation
from pet_code.src.util     import mm_energy_centroids
from pet_code.src.util     import select_module


import time
if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])
    
    start    = time.time()
    map_file = conf.get('mapping', 'map_file')#'pet_code/test_data/SM_mapping_corrected.yaml' # shouldn't be hardwired
    infiles  = args['INFILES']

    # time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    chan_map  = ChannelMap(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    # Should improve with an enum or something
    if 'Impacts'  in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_event_by_impacts(min_chan)
    elif 'OneMod' in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_impacts_one_minimod(min_chan, chan_map.get_minimodule)
    elif 'NoNeg'  in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_event_by_impacts_noneg(min_chan)
    else:
        print('No valid filter found, fallback to 4 minimum energy channels')
        evt_select = filter_event_by_impacts(4)

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal)

    c_calc   = centroid_calculation(chan_map.plotp)
    max_sel  = select_module(chan_map.get_minimodule) if conf.getboolean('filter', 'sel_max_mm') else lambda x: x
    out_dir  = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nsigma   = conf.getint('output', 'nsigma', fallback=2)
    sm_setup = 'ebrain' if 'brain' in map_file else 'tbpet'
    mm_ecent = mm_energy_centroids(c_calc, chan_map.get_supermodule,
                                   chan_map.get_minimodule, mod_sel=max_sel)
    try:
        out_form = conf.get('output', 'out_file')
    except configparser.NoOptionError:
        out_form = None

    pet_reader = read_petsys_filebyfile(chan_map.ch_type, evt_select)
    end_sec    = time.time()
    print(f'Time elapsed in setups: {end_sec - start} s')
    start_sec  = end_sec
    for fn in infiles:
        print(f'Reading file {fn}')
        filtered_events = list(map(cal_func, pet_reader(fn)))
        end_sec         = time.time()
        print(f'Time enlapsed reading: {end_sec - start_sec} s')
        print("length check: ", len(filtered_events))

        start_sec = end_sec
        mod_dicts = mm_ecent(filtered_events)

        cal     = 'cal' if time_cal else 'noCal'
        set_end = f'_{cal}_filt{filt_type}.ldat'
        if out_form is None:
            fbase    = fn.split('/')[-1]
            out_base = os.path.join(out_dir, fbase.replace('.ldat', set_end))
        else:
            fbase    = out_form + fn.split('/')[-1].replace('.ldat', set_end)
            out_base = os.path.join(out_dir, fbase)
        plotter  = mm_energy_spectra(sm_setup, out_base, 100, nsigma=nsigma)
        photo_peak = {i: plotter(i, vals) for i, vals in mod_dicts.items()}
        # photo_peak = dict(map(lambda i: i[0]: plotter(*i), mod_dicts.items()))
        end_p = time.time()
        print("Time enlapsed plotting: {} s".format(int(end_p - start_sec)))
            


