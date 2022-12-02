#!/usr/bin/env python3

"""Calculate and save at slab level: Energy Res., Centroid, CTR, min. and max. cog in monolithic direction

Usage: slab_performance.py (--conf CONFFILE) INFILE...

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
import numpy as np
import math
from scipy.signal import find_peaks


from pet_code.src.io    import read_petsys_filebyfile
from pet_code.src.io    import read_ymlmapping
from pet_code.src.plots import mm_energy_spectra
from pet_code.src.util  import calibrate_energies
from pet_code.src.util  import centroid_calculation
from pet_code.src.util  import filter_event_by_impacts
from pet_code.src.util  import filter_impacts_one_minimod
from pet_code.src.util  import mm_energy_centroids
from pet_code.src.util  import select_module
from pet_code.src.util  import select_array_range
from pet_code.src.util  import get_supermodule_eng
from pet_code.src.fits  import fit_gaussian



import time


def sm_slab_specs(sm):
    """Input: 
    sm: supermodule in dictionary form.
    Returns: 
    slab_params: dictionary with specs per slab: [compresion in monolithic (ymin, ymax), compresion
    in pixelated (xmin, xmax), mu, ER].
    roi_mm_dict: dictionary with the ROI per slab to be plotted in the floodmap."""

    slab_params    = {}
    roi_mm_dict    = {}
    mm_compression = {}
    for mm in sm:        
            #if mm == 1: 
            x_array          = np.array(sm[mm]["x"])
            y_array          = np.array(sm[mm]["y"])
            e_array          = np.array(sm[mm]["energy"])

            average_y        = np.average(sm[mm]["y"])
            cut_y_center_mm  = select_array_range(average_y - 2, average_y + 2)(y_array)

            profile_x        = x_array[cut_y_center_mm]
        
            n, bins_x, _     = plt.hist(profile_x, bins = 250, range = [0, 108])
            peaks_x, _       = find_peaks(n, height = max(n)/3, distance = 4, prominence=1) 
            #plt.plot(bins_x[peaks_x], n[peaks_x], 'rv', markersize=15, label="Peak finder")
            #plt.show()            
            plt.clf()
            roi_mm_dict[mm]  = []
            slab_params[mm]  = {}
            for slab, p in enumerate(peaks_x):
                cut_x_slab   = select_array_range(bins_x[p-1], bins_x[p+1])(x_array) 
                profile_y    = y_array[cut_x_slab]          
                n, bins_y, _ = plt.hist(profile_y, bins = 200, range = [0, 108])
                
                peaks_y, _   = find_peaks(n, height = max(n)/3, distance = 5)
                #plt.plot(bins_y[peaks_y], n[peaks_y], 'rv', markersize=15, label="Peak finder")
                #plt.show()
                plt.clf()
                roi_mm_dict[mm].append([bins_x[p-1], bins_x[p+1], bins_y[peaks_y[0] - 4], bins_y[peaks_y[-1] + 4]])

                cut_y_slab   = select_array_range(bins_y[peaks_y[0] - 4], bins_y[peaks_y[-1] + 4])(y_array)

                cut_roi_slab = np.logical_and(cut_x_slab, cut_y_slab)
                e_slab = e_array[cut_roi_slab]
                n, bins, _   = plt.hist(e_slab, bins = 150, range = [0, 150])
                try:
                    bcent, gvals, pars, _ = fit_gaussian(n, bins, cb=6, min_peak=max(n)/2)
                    FWHM_e                = round(pars[2]*math.sqrt( 8 * math.log( 2 ) ) , 3)
                    ER                    = round(pars[2]*math.sqrt( 8 * math.log( 2 ) ) / pars[1]*100, 2)
                    mu_e                  = round(pars[1], 3)
                    slab_params[mm][slab] = [round(bins_y[peaks_y[0]], 2), round(bins_y[peaks_y[-1]], 2), mu_e, ER]
                    if mm not in mm_compression.keys():
                        mm_compression[mm] = (round(bins_x[peaks_x[0]],2 ), round(bins_x[peaks_x[-1]], 2))
                except RuntimeError:
                    print("Error fitting slab energy ...")
                plt.clf()

    return slab_params, roi_mm_dict, mm_compression


def CTR_spec(events_sm, mod_sm, eng_ch, ctr_name):
    mm_en_limit = [{}, {}]
    mm_en       = [{key: value["energy"] for (key, value) in sm.items()} for sm in mod_sm]
    for n_sm, sm in enumerate(mm_en):
        for mm in sm:
            n, bins, _            = plt.hist(sm[mm], bins = 200, range = [0, 300])
            try:
                bcent, gvals, pars, _ = fit_gaussian(n, bins, cb=6, min_peak=max(n)/2)            
                mu_e                  = round(pars[1], 3)
                #plt.plot(bcent, gvals, label=f'fit $\mu$ = {mu_e},  $ER$ = {ER}')
                #plt.legend(loc = 0)
                #plt.show()
                plt.clf()
                mm_en_limit[n_sm][mm] = (pars[1] - 2*pars[2], pars[1] + 2*pars[2])
            except RuntimeError:
                print("Error fitting minimodule energy ...")
                plt.show()
                continue

    dt_dict = {}
    for sm in events_sm:
        _, en_sm0 = get_supermodule_eng(sm[0], eng_ch)
        _, en_sm1 = get_supermodule_eng(sm[1], eng_ch)
        mm0       = sm[0][0][1]
        mm1       = sm[1][0][1]
        mm0_minE  = mm_en_limit[0][mm0][0]
        mm0_maxE  = mm_en_limit[0][mm0][1]
        mm1_minE  = mm_en_limit[1][mm1][0]
        mm1_maxE  = mm_en_limit[1][mm1][1]
        if (en_sm0 > mm0_minE and en_sm0 < mm0_maxE) and (en_sm1 > mm1_minE and en_sm1 < mm1_maxE):
            alltch_sm0 = list(filter(lambda x: x[0] not in eng_ch, sm[0]))
            tch_sm0    = alltch_sm0[0][0]
            t_sm0      = alltch_sm0[0][2]
            alltch_sm1 = list(filter(lambda x: x[0] not in eng_ch, sm[1]))
            tch_sm1    = alltch_sm1[0][0]
            t_sm1      = alltch_sm1[0][2] 
            dt         = t_sm0 - t_sm1
            pair_ch    = str(tch_sm0) + "-" + str(tch_sm1)
            try:
                dt_dict[pair_ch].append(dt)
            except KeyError:
                dt_dict[pair_ch] = []

    
    CTR_skew_corr = []
    for pair in dt_dict:        
        num_events = len(dt_dict[pair])
        std_CTR    = np.std(dt_dict[pair])
        if std_CTR > 2000 or num_events < 20:
            continue
        try:
            n, bins, _            = plt.hist(dt_dict[pair], bins = 250, range = [-5000, 5000])            
            bcent, gvals, pars, _ = fit_gaussian(n, bins, cb=6, min_peak=max(n)/2)
            FWHM_t                = round(pars[2]*math.sqrt( 8 * math.log( 2 ) ) , 3)
            mu_t                  = round(pars[1], 3)
            #plt.plot(bcent, gvals, label=f'fit $\mu$ = {mu_e},  $CTR$ = {FWHM_t}, pair - {pair}')
            #plt.legend(loc = 0)
            #print("Length pair and CTR {}: {} - {} ps".format(pair, num_events, FWHM_t))
            #plt.show()
            plt.clf()
            CTR_skew_corr.extend(np.array(dt_dict[pair]) - pars[1])
        except RuntimeError:
            print("Error fitting time pair ...")
            CTR_skew_corr.extend(np.array(dt_dict[pair]) - 0)
            continue

    n, bins, _                   = plt.hist(CTR_skew_corr, bins = 250, range = [-5000, 5000])            
    bcent, gvals, pars, _        = fit_gaussian(n, bins, cb=6, min_peak=max(n)/2)
    FWHM_t = round(pars[2]*math.sqrt( 8 * math.log( 2 ) ) , 3)
    mu_t = round(pars[1], 3)
    plt.plot(bcent, gvals, label=f'fit $\mu$ = {mu_t},  $Full CTR$ = {FWHM_t}')
    plt.legend(loc = 0)
    print("Length full CTR: {} - {} ps".format(len(CTR_skew_corr), FWHM_t))
    out_name                     = ctr_name.replace(".ldat","_FullCTR.png")
    plt.savefig(out_name)
    plt.clf()
    return FWHM_t


def perf_to_file(slab_dict, compress_dict, ctr_meas, file_name):
    out_name = file_name.replace(".ldat","_performance.txt")
    with open(out_name, 'w') as perf_out:
        perf_out.write('SM_ID\tmM_ID\tslab_ID\tmin_y\tmax_y\tmu_e\tER\tmin_x\tmax_x\tCTR\n')
        for n_sm, sm in enumerate(slab_dict):
            for mm in sorted(sm):
                slabs = min(len(sm[mm]), 8)
                for n_sl, sl in enumerate(sm[mm]):                
                    perf_out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(n_sm, mm, sl,
                    sm[mm][sl][0], sm[mm][sl][1], sm[mm][sl][2], sm[mm][sl][3], compress_dict[n_sm][mm][0], 
                    compress_dict[n_sm][mm][1], ctr_meas))
                    if n_sl == slabs - 1:
                        break
                if slabs < 8:
                    for zero_slabs in range(slabs, 8):
                        perf_out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(n_sm, mm, zero_slabs,
                        0, 0, 0, 0, 0, 0, 0))


if __name__ == '__main__':
    args      = docopt(__doc__)
    conf      = configparser.ConfigParser()
    conf.read(args['--conf'])
    
    start     = time.time()
    map_file  = conf.get('mapping', 'map_file')#'pet_code/test_data/SM_mapping_corrected.yaml' # shouldn't be hardwired
    infiles   = args['INFILE']
    nsigma    = conf.getint('output', 'nsigma', fallback=2)

    time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    # Should improve with an enum or something
    if 'Impacts'  in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_event_by_impacts(eng_ch, *min_chan)
    elif 'OneMod' in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_impacts_one_minimod(eng_ch, *min_chan)
    else:
        print('No valid filter found, fallback to 4, 4 minimum energy channels')
        evt_select = filter_event_by_impacts(eng_ch, 4, 4)

    time_cal   = conf.get('calibration',   'time_channels', fallback='')
    eng_cal    = conf.get('calibration', 'energy_channels', fallback='')
    if time_cal or eng_cal:
        cal_func = calibrate_energies(time_ch, eng_ch, time_cal, eng_cal)
        out_cal = "Cal"
    else:
        cal_func = lambda x: x
        out_cal = "WoCal"
    end_r      = time.time()
    print("Time enlapsed configuring: {} s".format(int(end_r - start)))
    for f_in in infiles:
        out_dir             = conf.get('output', 'out_dir')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        f_in_name = f_in.split(os.sep)[-1]
        f_in_name_split = f_in_name.split(".")
        f_in_cal_name = [".".join(f_in_name_split[0:-1]) + out_cal, f_in_name_split[-1]]        
        out_base        = os.path.join(out_dir, ".".join(f_in_cal_name))
        
        start           = time.time()
        print("Reading file: {}...".format(f_in))
        pet_reader      = read_petsys_filebyfile(f_in, mm_map, evt_select)
        filtered_events = [cal_func(tpl) for tpl in pet_reader()]
        end_r           = time.time()
        print("Time enlapsed reading: {} s".format(int(end_r - start)))
        print("length check: ", len(filtered_events))
        start           = time.time()
        ## Should we be filtering the events with multiple mini-modules in one sm?
        c_calc          = centroid_calculation(centroid_map)
        # ## Must be a better way but...
        if conf.getboolean('filter', 'sel_max_mm'):
            def wrap_mmsel(eng_ch):
                def sel(sm):
                    return select_module(sm, eng_ch)
                return sel
            msel = wrap_mmsel(eng_ch)
        else:
            msel = lambda x: x
        mod_dicts           = mm_energy_centroids(filtered_events, c_calc, eng_ch, mod_sel=msel)
                
        CTR_meas            = CTR_spec(filtered_events, mod_dicts, eng_ch, out_base)

        roi_mm_dict_list    = [{}, {}]   #ROI per slab of each SM - key (mm): value ([Rxmin, Rxmax, Rymin, Rymax]) 
        slab_params_list    = [{}, {}]   #Slab parameters for each SM - key (mm): key(slab) : value ([ymin, ymax, mu, ER])
        mm_compression_list = [{}, {}]   #Minimodule compresion (in x direction) SM - key (mm): value ((xmin, xmax))
        
        for n_sm, sm in enumerate(mod_dicts):        
            slab_params_list[n_sm], roi_mm_dict_list[n_sm], mm_compression_list[n_sm] = sm_slab_specs(sm)
            
        photo_peak          = list(map(mm_energy_spectra, mod_dicts, [1, 2], repeat(out_base), repeat(100), repeat((0, 300)), repeat(nsigma), roi_mm_dict_list))

        perf_to_file(slab_params_list, mm_compression_list, CTR_meas, out_base)
        end_r               = time.time()
        print("Time enlapsed processing: {} s".format(int(end_r - start)))

    
