#!/usr/bin/env python3

"""Calculate the skew for each channel in a two super module set-up

Usage: skew_calc.py (--conf CONFFILE) [-n NCORE] [--it NITER] [--firstit FITER] [-r] INFILES ...

Arguments:
    INFILES File(s) to be processed.

Required:
    --conf=CONFFILE  Configuration file for run.

Options:
    -n=NCORE         Number of cores for processing [default: 2]
    --it=NITER       Number of iterations over data [default: 3]
    --firstit=FITRE  Iteration at which to start    [default: 0]
    -r               Recalculate from 0 assuming dt files available.
"""

import os
import configparser

from docopt    import docopt
# from glob      import glob
# from itertools import repeat

import numpy  as np
import pandas as pd

from functools import partial

# import matplotlib.pyplot as plt

# import multiprocessing as mp
# from multiprocessing import cpu_count, get_context

from pet_code.src.filters  import filter_impacts_specific_mod
from pet_code.src.fits     import fit_gaussian
from pet_code.src.fits     import mean_around_max
from pet_code.src.io       import ChannelMap
from pet_code.src.io       import read_petsys_filebyfile
# from pet_code.src.io       import read_ymlmapping
from pet_code.src.plots    import corrected_time_difference
# from pet_code.src.plots    import group_times_list
# from pet_code.src.plots    import slab_energy_spectra
from pet_code.src.util     import ChannelType
from pet_code.src.util     import calibrate_energies
# from pet_code.src.util     import centroid_calculation
from pet_code.src.util     import time_of_flight
from pet_code.src.util     import select_energy_range
from pet_code.src.util     import select_module
# from pet_code.src.util     import slab_energy_centroids


# Source positions. Improve!!
# def source_position(sm_num, mm_num):
#     """
#     Hard wired source positions!
#     """
#     x = 103.6 - 25.9 * (0.5 + (mm_num - 1) // 4)
#     if sm_num == 3:
#         y = -25.9 * 0.5 + (3 - (mm_num - 1) % 4) * 25.9
#         z = 123.7971 - 31.8986
#         return np.array([x, y, z])
#     y = -25.9 * 0.5 - (mm_num - 1) % 4 * 25.9
#     z = 31.8986
#     return np.array([x, y, z])
pos_to_mm = {1: { 1: 3,  2: 7,  3:11,  4:15,
                  5: 2,  6: 6,  7:10,  8:14,
                  9: 1, 10: 5, 11: 9, 12:13,
                 13: 0, 14: 4, 15: 8, 16:12},
             3: { 1:12,  2: 8,  3: 4,  4: 0,
                  5:13,  6: 9,  7: 5,  8: 1,
                  9:14, 10:10, 11: 6, 12: 2,
                 13:15, 14:11, 15: 7, 16: 3}}
def source_position(sm_num, pos_num):
    """
    Hard wired source positions!
    """
    mm = pos_to_mm[sm_num][pos_num]
    y = -25.9 * (0.5 + (pos_num - 1) // 4)
    if sm_num == 3:
        x = 103.6 - 25.9 * (0.5 + (pos_num - 1) % 4)
        z = 123.7971 - 31.8986
        return mm, np.array([x, y, z])
    x =  25.9 * (0.5 + (pos_num - 1) %  4)
    z =  33.8986
    return mm, np.array([x, y, z])


def get_references(file_name):
    """
    Extract supermodule and minimodule
    numbers for reference channels and
    the source position.
    DUMMY!
    """
    # Get them from the filename?
    # Source pos from some saved lookup!
    file_name_parts = file_name.split(os.sep)[-1].split('_')
    # SM_lab = int(file_name_parts[1][2:])
    SM_lab          = int(file_name_parts[1][ 8:9])
    SM_indx         = 0 if SM_lab == 3 else 1
    # mM_num = int(file_name_parts[2][2:])
    source_posNo    = int(file_name_parts[1][12: ])
    mM_num, s_pos   = source_position(SM_lab, source_posNo)
    # return SM_indx, mM_num, source_position(SM_lab, mM_num)#[38.4, 38.4, 22.5986]
    return SM_indx, mM_num, s_pos


def geom_loc_2sm(file_name, ch_map):
    """
    Get functions for the reference index
    and geometric dt for the 2 SM setup.
    REVIEW, a lot of inversion to adapt!
    """
    SM_indx, mM_num, s_pos = get_references(file_name)
    mm_channels = ch_map.get_minimodule_channels(2 if SM_indx == 0 else 0, mM_num)
    valid_ids   = set(filter(lambda x: ch_map.get_channel_type(x) is ChannelType.TIME, mm_channels))
    def find_indx(evt):
        return 0 if evt[0][0] in valid_ids else 1

    flight_time = time_of_flight(s_pos)
    def geom_dt(ref_id, coinc_id):
        ref_pos   = ch_map.get_channel_position(  ref_id)
        coinc_pos = ch_map.get_channel_position(coinc_id)
        return flight_time(ref_pos) - flight_time(coinc_pos)
    # Return the index and mm number too. Specific to 2SM so not ideal.
    return SM_indx, mM_num, find_indx, geom_dt


# def read_and_select(file_list, config):
#     """
#     Read the binaries corresponding to file_list,
#     filter on the slab spectra, group and return
#     raw channels and timestamps.
#     Will use parallelization but start simple for
#     now.
#     """
#     (time_ch, eng_ch, mm_map,
#      centroid_map, slab_map) = read_ymlmapping(config.get('mapping', 'map_file'))
#     outdir                   = config.get('output', 'out_dir')

#     sm1_minch, sm2_minch = tuple(map(int, config.get('filter', 'min_channels').split(',')))
#     relax                = config.getfloat('filter', 'relax_fact')
#     c_calc               = centroid_calculation(centroid_map)

#     time_cal = config.get('calibration',   'time_channels', fallback='')
#     eng_cal  = config.get('calibration', 'energy_channels', fallback='')
#     cal_func = calibrate_energies(time_ch, eng_ch, time_cal, eng_cal)

#     ebins    = np.arange(*tuple(map(float, config.get('plots', 'ebinning', fallback='5,25,0.2').split(','))))
#     min_efit = config.getint('plots', 'min_estats', fallback=100)

#     all_skews = pd.Series(dtype=float)
#     relax = config.getfloat('filter', 'relax_fact')

#     for fn in file_list:
#         print(f'Processing file {fn}', flush=True)
#         sm_num, mm_num, source_pos = get_references(fn)
#         evt_filter = filter_impacts_specific_mod(sm_num, mm_num, eng_ch, sm1_minch, sm2_minch)
#         evt_reader = read_petsys_filebyfile(mm_map, evt_filter)
#         out_base   = os.path.join(outdir, fn.split('/')[-1])
#         skew_calc  = get_skew(time_of_flight(source_pos), slab_map, plot_output=out_base)

#         sel_evts   = list(map(cal_func, evt_reader(fn)))
#         print(f'Events read {len(sel_evts)}. Proceeding...', flush=True)
#         slab_dicts = slab_energy_centroids(sel_evts, c_calc, time_ch)

#         photo_peak = list(map(slab_energy_spectra, slab_dicts,
#                               repeat(None), repeat(min_efit), repeat(ebins)))

#         deltat_df = pd.DataFrame(group_times_list(sel_evts, photo_peak, sm_num),
#                                  columns=['ref_ch', 'coinc_ch', 'coinc_tstp', 'ref_tstp'])
#         deltat_df.to_pickle(out_base.replace('.ldat', '_dtFrame.pkl'))
#         print(f'Time DataFrame output for {out_base}', flush=True)
 
#         skew_values = deltat_df.groupby('ref_ch', group_keys=False).apply(skew_calc)
#         print(f'Skews calculated for {out_base}', flush=True)
#         all_skews   = pd.concat((all_skews, relax * skew_values))
#     return all_skews


# def time_distributions(file_list, config, skew_values, it):
#     """
#     Read from the pickle files of the
#     coincidence time frames, make the
#     dt distributions correcting for
#     skew and fit for current skew.
#     """
#     corr_skews   = skew_values.copy()
#     outdir       = config.get('output', 'out_dir')
#     *_, slab_map = read_ymlmapping(config.get('mapping', 'map_file'))
#     relax        = config.getfloat('filter', 'relax_fact')
#     for fn in file_list:
#         *_, source_pos = get_references(fn)
#         ## Probably want a function for the name so consistent
#         out_base   = os.path.join(outdir, fn.split('/')[-1])
#         skew_calc  = get_skew(time_of_flight(source_pos),
#                               slab_map                  ,
#                               skew        = skew_values ,
#                               plot_output = out_base    ,
#                               it          = it          )
#         pkl_name   = out_base.replace('.ldat', '_dtFrame.pkl')
#         deltat_df  = pd.read_pickle(pkl_name)
#         biases     = deltat_df.groupby('ref_ch', group_keys=False).apply(skew_calc)
#         corr_skews = corr_skews.add(relax * biases, fill_value=0.0)
#     return corr_skews


# def get_skew(flight_time, slab_map, skew=pd.Series(dtype=float), plot_output=None, it=0):
#     """
#     Get the skew value for a reference
#     channel correcting for previously
#     calculated skew if available.
#     """
#     # Shouldn't be hardwired!!
#     hist_bins = np.linspace(-10000, 10000, 400, endpoint=False)
#     def calc_skew(delta_t):
#         ref_ch    = delta_t.ref_ch.unique()[0]
#         dt_th     = flight_time(slab_map[ref_ch]) - np.fromiter((flight_time(slab_map[id]) for id in delta_t.coinc_ch), float)
#         ref_skew  = skew.get(ref_ch, 0)
#         skew_corr = np.fromiter((skew.get(id, 0) for id in delta_t.coinc_ch), float) - ref_skew
#         dt_reco   = np.diff(delta_t[['coinc_tstp', 'ref_tstp']], axis=1).flatten()

#         if plot_output:
#             bin_vals, bin_edges, _ = plt.hist(dt_reco - dt_th + skew_corr, bins=hist_bins)
#             plt.xlabel(f'$dt_r$ - $dt_t$ for slab {ref_ch} (ps)')
#         else:
#             bin_vals, bin_edges = np.histogram(dt_reco - dt_th + skew_corr, bins=hist_bins)
#         try:
#             bcent, gvals, pars, _ = fit_gaussian(bin_vals, bin_edges, min_peak=100)
#             if plot_output:
#                 plt.plot(bcent, gvals, label=f'fit $\mu$ = {round(pars[1], 3)},  $\sigma$ = {round(pars[2], 3)}')
#                 plt.legend()
#                 plt.savefig(plot_output.replace('.ldat', f'_it{it}_timeCoincRef{ref_ch}.png'))
#                 plt.clf()
#         except RuntimeError:
#             print(f'Ref channel {ref_ch} fit fail', flush=True)
#             plt.clf()
#             peak_mean, *_ = mean_around_max(bin_vals, bin_edges[:-1], 6)
#             return peak_mean if peak_mean else 0
#         return pars[1]
#     return calc_skew


def process_raw_data(config):
    """
    Read the binaries corresponding to file_list,
    filter selecting events in the slab spectra
    limits, calculate geometrically corrected deltaT
    and output to feather files.
    """
    ch_map = ChannelMap(config.get('mapping', 'map_file'))

    ## This needs to be improved. In principle it can be in the map.
    time_cal = config.get('calibration',   'time_channels', fallback='')
    eng_cal  = config.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(ch_map.get_chantype_ids, time_cal, eng_cal)
    sel_mod  = select_module(ch_map.get_minimodule)
    def cal_and_select(evt):
        cal_evt = cal_func(evt)
        return tuple(map(sel_mod, cal_evt))

    elimits  = map(float, config.get('filter', 'elimits').split(','))
    eselect  = select_energy_range(*elimits)

    setup    = config.get('mapping', 'setup', fallback='2SM')
    if setup == '2SM':
        geom_func = geom_loc_2sm
        sm1_minch, sm2_minch = tuple(map(int, config.get('filter', 'min_channels').split(',')))
        evt_filt  = partial(filter_impacts_specific_mod    ,
                            mm_map  = ch_map.get_minimodule,
                            min_sm1 = sm1_minch            ,
                            min_sm2 = sm2_minch            )
    else:
        print(f'Setup {setup} not available')
        exit()

    ## Need filter options and position options for general usage.
    #
    df_cols    = ['ref_ch', 'coinc_ch', 'corr_dt']
    dt_calc    = partial(corrected_time_difference  ,
                         impact_sel = cal_and_select,
                         energy_sel = eselect       )
    pet_reader = partial(read_petsys_filebyfile, type_dict=ch_map.ch_type)
    outdir     = config.get('output', 'out_dir')
    def _process(file_list):
        for fn in file_list:
            sm_no, mm_no, ref_ch, geom_dt = geom_func(fn, ch_map)
            ## Specific filter depends on source position.
            # Not ideal, should be easier with filter update.
            filt   = evt_filt(sm_num=sm_no, mm_num=mm_no)
            dt_gen = map(dt_calc(ref_ch=ref_ch, geom_dt=geom_dt),
                         pet_reader(sm_filter=filt)(fn)         )
            evt_df = pd.DataFrame(filter(lambda x: x, dt_gen), columns=df_cols)

            out_base   = fn.split(os.sep)[-1].replace('.ldat', '.feather')
            feath_name = os.path.join(outdir, out_base)
            evt_df.to_feather(feath_name)
            yield feath_name
    return _process


def calculate_skews(file_list, config, skew_values):
    """
    Read files with channel numbers and calculated
    delta_t - delta_t_geom and calculate the bias
    correcting for any known skew.
    Want monitor plots with iteration number?
    """
    corr_skews = skew_values.copy()
    rel_fact   = config.getfloat('filter', 'relax_fact')
    min_stats  = config.getint  ('filter',  'min_stats')
    hist_bins  = map(float, config.get('filter',  'hist_bins').split(','))
    bias_calc  = peak_position(np.arange(*hist_bins), min_stats, skew_values)
    for fn in file_list:
        dt_df      = pd.read_feather(fn)
        biases     = dt_df.groupby('ref_ch', group_keys=False).apply(bias_calc)
        corr_skews = corr_skews.add(rel_fact * biases, fill_value=0.0)
    return corr_skews


def peak_position(hist_bins, min_stats, skew):
    """
    Calculate the mean position of the delta time distribution
    corrected for theoretical difference and, optionally, skew.
    """
    def calculate_bias(delta_t):
        """
        Do the calculation for every row of delta t.
        Assumes that the ref_ch is the same for all.
        """
        ref_ch    = delta_t.ref_ch.unique()[0]
        ref_skew  = skew.get(ref_ch, 0)
        skew_corr = skew.loc[delta_t.coinc_ch].values - ref_skew

        bin_vals, bin_edges = np.histogram(delta_t.corr_dt.values + skew_corr, bins=hist_bins)
        try:
            *_, pars, _ = fit_gaussian(bin_vals, bin_edges, min_peak=min_stats)
        except RuntimeError:
            print(f'Ref channel {ref_ch} fit fail', flush=True)
            peak_mean, *_ = mean_around_max(bin_vals, bin_edges[:-1], 6)
            return peak_mean if peak_mean else 0
        return pars[1]
    return calculate_bias


if __name__ == '__main__':
    args     = docopt(__doc__)
    ncores   = int(args['-n'       ]) # For parallelization
    niter    = int(args['--it'     ])
    first_it = int(args['--firstit'])

    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    # Try without glob...
    input_files = args['INFILES']
    outdir      = conf.get('output', 'out_dir')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    total_iter  = first_it + niter
    out_skew    = input_files[0].split(os.sep)[-1].split('_')[0] + f'_skew_{total_iter}iter.txt'
    skew_file   = os.path.join(outdir, out_skew)
    if first_it != 0:
        # Probably want to update formats?
        skew_values = pd.read_csv(skew_file).set_index('id')['Skew']
        # Let's not overwrite existing outputs.
        skew_file = skew_file.replace('.txt', '_recalc.txt')
    else:
        all_ids     = pd.read_feather(conf.get('mapping', 'map_file')).id
        skew_values = pd.Series(0, index=all_ids.sort_values())
        if not args['-r']:
            bin_proc    = process_raw_data(conf)
            # Possible to parallelize here?
            input_files = list(bin_proc(input_files))
            print('Binary processing complete')

    # start iterations (will need to do monitor plots using i for iteration):
    for i in range(first_it, total_iter):
        print(f'Starting iteration {i}')
        skew_values = calculate_skews(input_files, conf, skew_values)

    ## Save the skew values, change to format expected for PETsys?
    print('Requested iterations complete, output text file.')
    skew_values = skew_values.reset_index().rename(columns={'index': 'id', 0: 'Skew'})
    skew_values.to_csv(skew_file)


# if __name__ == '__main__':
#     args     = docopt(__doc__)
#     ncores   = int(args['-n'       ]) # For parallelization
#     niter    = int(args['--it'     ])
#     first_it = int(args['--firstit'])

#     conf   = configparser.ConfigParser()
#     conf.read(args['--conf'])

#     # ncpu = mp.cpu_count()
#     ncpu = cpu_count()
#     if ncores > ncpu:
#         print(f'Too many cores requested ({ncores}), only {ncpu} available.')
#         print('Will use half available cores.')
#         ncores = ncpu // 2

#     input_files = glob(args['INBASE'] + '*.ldat')
#     outdir = conf.get('output', 'out_dir')
#     if not os.path.isdir(outdir):
#         os.makedirs(outdir)
#     # print("File Checks: ", input_files)
#     if first_it != 0:
#         skew_file   = os.path.join(conf.get('output', 'out_dir'),
#                                    args['INBASE'].split('/')[-1].split('_')[0]) + '_skew.txt'
#         skew_values = pd.read_csv(skew_file).set_index('Channel_id')['Skew']
#     for i in range(first_it, first_it + niter):
#         print(f'Start iteration {i}')
#         if i == 0:
#             ## Read the ldat binaries and do the first calculation.
#             ## We definitely want to parallelize here.
#             chunk_args = [(file_set, conf) for file_set in np.array_split(input_files, ncores)]
#             # with mp.Pool(ncores) as p:
#             with get_context("spawn").Pool(ncores) as p:
#                 # Run chunks in parallel
#                 skew_chunks = p.starmap(read_and_select, chunk_args)
#             print('Finished parallel read, concatenating results')
#             skew_values = pd.concat(skew_chunks)
#             # skew_values = read_and_select(input_files, conf)
#         else:
#             skew_values = time_distributions(input_files, conf,
#                                              skew_values, i   )
#     ## Save the skew values.
#     print('Requested iterations complete, output text file.')
#     skew_values = skew_values.reset_index().rename(columns={'index': 'Channel_id', 0: 'Skew'})
#     skew_file   = os.path.join(conf.get('output', 'out_dir'), args['INBASE'].split('/')[-1].split('_')[0]) + '_skew.txt'
#     skew_values.to_csv(skew_file)
