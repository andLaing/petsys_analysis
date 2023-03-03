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

import numpy  as np
import pandas as pd

from functools import partial

# import matplotlib.pyplot as plt

# import multiprocessing as mp
    """
    Extract supermodule and minimodule
    numbers for reference channels and
    the source position.
    DUMMY!
    """
    # Get them from the filename?
    # Source pos from some saved lookup!
    file_name_parts = file_name.split(os.sep)[-1].split('_')
    SM_lab          = int(file_name_parts[1][ 8:9])
    SM_indx         = 0 if SM_lab == 3 else 1
    source_posNo    = int(file_name_parts[1][12: ])
    mM_num, s_pos   = source_p(SM_lab, source_posNo)
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


def geom_loc_bar(file_name, ch_map):
    """
    Get functions for the reference index
    and geometric dt for setups with bar source.
    """
    # Need a lookup for bar_xy position
    bar_xy = np.array([280, 280])
    bar_r  = 10#mm
    # Need a lookup for SM/MM groups of interest
    # 'single ring' fixed pos for now
    SM_no = (0,)
    mms   = (0, 1, 4, 5, 8, 9, 12, 13)
    ids   = np.concatenate([ch_map.get_minimodule_channels(*v)
                            for v in np.vstack(np.stack(np.meshgrid(SM_no, mms)).T)])
    s_ids = set(filter(lambda x: ch_map.get_channel_type(x) is ChannelType.TIME, ids))
    def find_indx(evt):
        return 0 if evt[0][0] in s_ids else 1

    geom_dt = bar_source_dt(bar_xy, bar_r, ch_map.get_channel_position)
    return ids, find_indx, geom_dt


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
    if   setup == '2SM'      :
        geom_func = geom_loc_2sm
        sm1_minch, sm2_minch = tuple(map(int, config.get('filter', 'min_channels').split(',')))
        evt_filt  = partial(filter_impacts_specific_mod    ,
                            mm_map  = ch_map.get_minimodule,
                            min_sm1 = sm1_minch            ,
                            min_sm2 = sm2_minch            )
    # elif setup == 'barSource':
    #     geom_func = partial
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


