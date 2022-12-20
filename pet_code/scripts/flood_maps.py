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

    time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    # Should improve with an enum or something
    if 'Impacts'  in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_event_by_impacts(eng_ch, *min_chan)
    elif 'OneMod' in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_impacts_one_minimod(eng_ch, *min_chan)
    elif 'NoNeg'  in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_event_by_impacts_noneg(eng_ch, *min_chan)
    else:
        print('No valid filter found, fallback to 4, 4 minimum energy channels')
        evt_select = filter_event_by_impacts(eng_ch, 4, 4)

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(time_ch, eng_ch, time_cal, eng_cal)

    pet_reader      = read_petsys_filebyfile(mm_map, evt_select)
    filtered_events = [cal_func(tpl) for tpl in pet_reader(infile)]
    end_r           = time.time()
    print("Time enlapsed reading: {} s".format(int(end_r - start)))
    print("length check: ", len(filtered_events))
    ## Should we be filtering the events with multiple mini-modules in one sm?
    c_calc = centroid_calculation(centroid_map)
    # ## Must be a better way but...
    if conf.getboolean('filter', 'sel_max_mm'):
        def wrap_mmsel(eng_ch):
            def sel(sm):
                return select_module(sm, eng_ch)
            return sel
        msel = wrap_mmsel(eng_ch)
    else:
        msel = lambda x: x
    mod_dicts = mm_energy_centroids(filtered_events, c_calc, eng_ch, mod_sel=msel)

    out_dir    = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nsigma = conf.getint('output', 'nsigma', fallback=2)
    out_base   = os.path.join(out_dir, conf.get('output', 'out_file', fallback=infile.split('/')[-1]))#'test_floods/' + file_list[0].split('/')[-1]
    photo_peak = list(map(mm_energy_spectra, mod_dicts, [1, 2], repeat(out_base), repeat(100), repeat((0, 300)), repeat(nsigma)))
            


