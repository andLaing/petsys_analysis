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
from pet_code.src.fits  import fit_gaussian


import time


def sm_slab_specs(sm):
    """Takes the sm event by event information and retrieve slab performance: compresion,
    energy resolution (ER) and centroid of the energy distribution per miniModule.
    Input: 
    sm: supermodule in dictionary form.
    Returns: 
    slab_params: dictionary with specs per slab: [compresion in monolithic (ymin, ymax), compresion
    in pixelated (xmin, xmax), mu, ER].
    roi_mm_dict: dictionary with the ROI per slab to be plotted in the floodmap.
    """

    slab_params    = {}
    roi_mm_dict    = {}
    mm_compression = {}
    for mm in sm:        
            #if mm == 4: 
            x_array          = np.array(sm[mm]["x"])
            y_array          = np.array(sm[mm]["y"])
            e_array          = np.array(sm[mm]["energy"])
            """
            fig = plt.figure(8)
            plt.hist(x_array, bins = 250, range = [0, 108]) 
            fig = plt.figure(9)
            plt.hist(y_array, bins = 250, range = [0, 108]) 
            fig = plt.figure(10)
            plt.hist2d(x_array, y_array, bins = 500, range=[[0, 104], [0, 104]], cmap="Greys", cmax=50)        
            plt.show()
            """
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


def perf_to_file(slab_dict, compress_dict, file_name):
    """Output slab performance to txt file to be plotted. 
    Inputs:
    slab_dict:     dictionary with slab characteristics. 
    compress_dict: dictionary with compression parameters per miniModule.
    file_name:     output file name for the txt datafile
    Returns:
    no return
    """
    out_name = file_name.replace(".ldat","_performance.txt")
    with open(out_name, 'w') as perf_out:
        perf_out.write('SM_ID\tmM_ID\tslab_ID\tmin_y\tmax_y\tmu_e\tER\tmin_x\tmax_x\tCTR\n')    
        for mm in sorted(slab_dict):
            slabs = min(len(slab_dict[mm]), 8)
            for n_sl, sl in enumerate(slab_dict[mm]):                
                perf_out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(1, mm, sl,
                slab_dict[mm][sl][0], slab_dict[mm][sl][1], slab_dict[mm][sl][2], slab_dict[mm][sl][3], compress_dict[mm][0], 
                compress_dict[mm][1]))
                if n_sl == slabs - 1:
                    break
            if slabs < 8:
                for zero_slabs in range(slabs, 8):
                    perf_out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(1, mm, zero_slabs,
                    0, 0, 0, 0, 0, 0))


if __name__ == '__main__':
    args      = docopt(__doc__)
    conf      = configparser.ConfigParser()
    conf.read(args['--conf'])
    
    start     = time.time()
    map_file  = conf.get('mapping', 'map_file')#'pet_code/test_data/SM_mapping_corrected.yaml' # shouldn't be hardwired
    infiles   = args['INFILE']
    nsigma    = conf.getint('output', 'nsigma', fallback=2)
    singles_flag = conf.get('filter', 'singles', fallback='False')

    time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    # Should improve with an enum or something
    if 'Impacts'  in filt_type:
        min_chan   = tuple(map(int, conf.get('filter', 'min_channels').split(',')))
        evt_select = filter_event_by_impacts(eng_ch, *min_chan, singles = singles_flag)
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
        out_cal  = "Cal"
    else:
        cal_func = lambda x: x
        out_cal  = "WoCal"
    end_r      = time.time()
    print("Time enlapsed configuring: {} s".format(int(end_r - start)))
    for f_in in infiles:
        out_dir = conf.get('output', 'out_dir')
        png_dir = out_dir + "/pngs"
        txt_dir = out_dir + "/txts"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)                    
            os.makedirs(png_dir)
            os.makedirs(txt_dir)
        
        f_in_name       = f_in.split(os.sep)[-1]
        f_in_name_split = f_in_name.split(".")
        f_in_cal_name   = [".".join(f_in_name_split[0:-1]) + out_cal, f_in_name_split[-1]]  
        out_base        = os.path.join(out_dir, ".".join(f_in_cal_name))      
        out_base_png    = os.path.join(png_dir, ".".join(f_in_cal_name))
        out_base_txt    = os.path.join(txt_dir, ".".join(f_in_cal_name))
        
        start           = time.time()
        print("Reading file: {}...".format(f_in))
        pet_reader      = read_petsys_filebyfile(f_in, mm_map, evt_select, singles = singles_flag)        
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
        #print(filtered_events[0])
        mod_dicts           = mm_energy_centroids(filtered_events, c_calc, eng_ch, mod_sel=msel)

        roi_mm_dict_list    = {}   #ROI per slab of each SM - key (mm): value ([Rxmin, Rxmax, Rymin, Rymax]) 
        slab_params_list    = {}   #Slab parameters for each SM - key (mm): key(slab) : value ([ymin, ymax, mu, ER])
        mm_compression_list = {}   #Minimodule compresion (in x direction) SM - key (mm): value ((xmin, xmax))
        
                
        slab_params_list, roi_mm_dict_list, mm_compression_list = sm_slab_specs(mod_dicts[0])
            
        photo_peak          = list(mm_energy_spectra(mod_dicts[0], 1, out_base_png, 100, (0, 300), nsigma, roi_mm_dict_list))

        perf_to_file(slab_params_list, mm_compression_list, out_base_txt)
        end_r               = time.time()
        print("Time enlapsed processing: {} s".format(int(end_r - start)))

    
