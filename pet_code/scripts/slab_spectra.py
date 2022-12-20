#!/usr/bin/env python3

"""Calculate and save time channel level energy spectra

Usage: slab_spectra.py (--conf CONFFILE) INFILE

Arguments:
    INFILE  File name to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt    import docopt
from itertools import repeat

import matplotlib.pyplot as plt
import numpy  as np

from pet_code.src.io    import read_petsys_filebyfile
from pet_code.src.io    import read_ymlmapping
from pet_code.src.plots import slab_energy_spectra
from pet_code.src.util  import calibrate_energies
from pet_code.src.util  import centroid_calculation
from pet_code.src.util  import filter_event_by_impacts
from pet_code.src.util  import filter_impacts_one_minimod
from pet_code.src.util  import filter_impacts_specific_mod
from pet_code.src.util  import slab_energy_centroids

if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    map_file = conf.get('mapping', 'map_file')#'pet_code/test_data/SM_mapping_corrected.yaml' # shouldn't be hardwired
    infile   = args['INFILE']

    time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    # Should improve with an enum or something
    if 'Impacts'    in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_event_by_impacts(eng_ch, *min_chan)
    elif 'OneMod'   in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_impacts_one_minimod(eng_ch, *min_chan)
    elif 'Specific' in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        # Still needs to be improved, re: index not obvious maybe
        ref_indx   = conf.getint('filter', 'supermod_indx')
        mm_num     = conf.getint('filter',   'mini_module')
        evt_select = filter_impacts_specific_mod(ref_indx, mm_num, eng_ch, *min_chan)
    else:
        print('No valid filter found, fallback to 4, 4 minimum energy channels')
        evt_select = filter_event_by_impacts(eng_ch, 4, 4)

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(time_ch, eng_ch, time_cal, eng_cal)

    pet_reader = read_petsys_filebyfile(mm_map, evt_select)
    filtered_events = list(map(cal_func, pet_reader(infile)))
    ## Should we be filtering the events with multiple mini-modules in one sm?
    c_calc     = centroid_calculation(centroid_map)
    slab_dicts = slab_energy_centroids(filtered_events, c_calc, time_ch)

    out_dir    = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    min_peak   = conf.getint('output', 'min_peak_fit', fallback=100)
    bin_edges  = np.arange(*tuple(map(float, conf.get('output', 'binning', fallback='9,25,0.2').split(','))))
    out_base   = os.path.join(out_dir, conf.get('output', 'out_file', fallback=infile.split('/')[-1]))
    photo_peak = list(map(slab_energy_spectra, slab_dicts, repeat(out_base), repeat(min_peak), repeat(bin_edges)))

    # reco_dt = group_times_slab(filtered_events, photo_peak, time_ch, ref_indx)
    # # Source pos hardwired for tests, extract from data file?
    # flight_time = time_of_flight(np.array([38.4, 38.4, 22.5986]))
    # for ref_ch, tstps in reco_dt.items():
    #     time_arr   = np.array(tstps)
    #     dt_th      = flight_time(slab_map[ref_ch]) - np.fromiter((flight_time(slab_map[id]) for id in time_arr[:, 0]), float)
    #     tstp_diff  = np.diff(time_arr[:, 1:], axis=1).flatten()
    #     plt.hist(tstp_diff, bins=300, range=[-10000, 10000], histtype='step', fill=False, label = f"Ref ch {ref_ch}")
    #     plt.hist(tstp_diff - dt_th, bins=300, range=[-10000, 10000], histtype='step', fill=False, label = f"Ref ch {ref_ch} theory corr dt")
    #     plt.xlabel(f'tstp ch {ref_ch} - tstp coinc (ps)')
    #     plt.legend()
    #     ## Temp hardwire!!
    #     out_name = 'test_plots/slab_filt/' + file_list[0].split('/')[-1].replace('.ldat', '_timeCoincRef' + str(ref_ch) + '.png')
    #     plt.savefig(out_name)
    #     plt.clf()
