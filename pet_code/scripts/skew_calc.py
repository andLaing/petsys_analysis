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
import yaml

from docopt import docopt

import numpy  as np
import pandas as pd

from functools import partial

# import matplotlib.pyplot as plt

# import multiprocessing as mp
from multiprocessing import cpu_count, get_context

from pet_code.src.filters import filter_impacts_channel_list
from pet_code.src.fits    import fit_gaussian
from pet_code.src.fits    import mean_around_max
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.plots   import corrected_time_difference
from pet_code.src.util    import ChannelType
from pet_code.src.util    import bar_source_dt
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import time_of_flight
from pet_code.src.util    import get_absolute_id
from pet_code.src.util    import get_electronics_nums
from pet_code.src.util    import select_energy_range
from pet_code.src.util    import select_module


def source_position(pos_conf):
    pos_to_mm  = pos_conf[ 'pos_to_mm']
    source_pos = pos_conf['source_pos']
    def _source_position(sm_num, pos_num):
        mm = pos_to_mm[sm_num][pos_num]
        return mm, np.asarray(source_pos[sm_num][mm])
    return _source_position


def get_references(file_name, source_p, correct=0):
    """
    Extract supermodule and minimodule
    numbers for reference channels and
    the source position.
    correct allows backward compatibility for old numbering
    """
    file_name_parts = file_name.split(os.sep)[-1].split('_')
    SM_lab          = int(file_name_parts[1][ 8:9])
    source_posNo    = int(file_name_parts[1][12: ])
    mM_num, s_pos   = source_p(SM_lab, source_posNo)
    return SM_lab + correct, mM_num, s_pos


def geom_loc_point(file_name, ch_map, source_yml, corr_sm_no=0):
    """
    Get functions for the reference index
    and geometric dt for the 2 SM setup.
    REVIEW, a lot of inversion to adapt!
    """
    (SM_lab,
     mM_num,
     s_pos ) = get_references(file_name, source_position(source_yml), corr_sm_no)
    mm_channels = ch_map.get_minimodule_channels(SM_lab, mM_num)
    valid_ids   = set(filter(lambda x: ch_map.get_channel_type(x) is ChannelType.TIME, mm_channels))
    def find_indx(evt):
        return 0 if evt[0][0] in valid_ids else 1

    flight_time = time_of_flight(s_pos)
    def geom_dt(ref_id, coinc_id):
        ref_pos   = ch_map.get_channel_position(  ref_id)
        coinc_pos = ch_map.get_channel_position(coinc_id)
        return flight_time(ref_pos) - flight_time(coinc_pos)
    return mm_channels, find_indx, geom_dt


def geom_loc_bar(file_name, ch_map, source_yml):
    """
    Get functions for the reference index
    and geometric dt for setups with bar source.
    """
    file_label  = 'SourcePos'#Assume this is followed by position number.
    source_indx = file_name.find(file_label)
    source_pnum = int(file_name[source_indx + len(file_label):].split('_')[0])
    # Need a lookup for bar_xy position
    bar_xy = np.asarray(source_yml['bar_xy'][source_pnum])
    bar_r  =            source_yml['bar_r' ]#mm
    SM_no  =            source_yml['ref_SM'][source_pnum]
    mms    =            source_yml['ref_MM'][source_pnum]
    ids    = np.concatenate([ch_map.get_minimodule_channels(*v)
                             for v in np.vstack(np.stack(np.meshgrid(SM_no, mms)).T)])
    s_ids  = set(filter(lambda x: ch_map.get_channel_type(x) is ChannelType.TIME, ids))
    def find_indx(evt):
        return 0 if evt[0][0] in s_ids else 1

    geom_dt = bar_source_dt(bar_xy, bar_r, ch_map.get_channel_position)
    return ids, find_indx, geom_dt


def process_raw_data(file_list, config, ch_map):
    """
    Read the binaries corresponding to file_list,
    filter selecting events in the slab spectra
    limits, calculate geometrically corrected deltaT
    and output to feather files.
    """
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
    minch    = config.getint('filter', 'min_channels')
    evt_filt = partial(filter_impacts_channel_list   ,
                       min_ch = minch                ,
                       mm_map = ch_map.get_minimodule)
    if   setup == '2SM'      :
        ## For backwards compatibility.
        correct_sm = config.getint('mapping', 'SM_NO_CORR', fallback=0)
        with open(config.get('mapping', 'source_pos')) as s_yml:
            yml_positions = yaml.safe_load(s_yml)
            geom_func     = partial(geom_loc_point            ,
                                    ch_map     = ch_map       ,
                                    source_yml = yml_positions,
                                    corr_sm_no = correct_sm   )
    elif setup == 'barSource':
        with open(config.get('mapping', 'source_pos')) as s_yml:
            yml_positions = yaml.safe_load(s_yml)
            geom_func     = partial(geom_loc_bar              ,
                                    ch_map     = ch_map       ,
                                    source_yml = yml_positions)
    else:
        print(f'Setup {setup} not available')
        exit()

    ## Need filter options and position options for general usage.
    #
    df_cols      = ['ref_ch', 'coinc_ch', 'corr_dt']
    dt_calc      = partial(corrected_time_difference  ,
                           impact_sel = cal_and_select,
                           energy_sel = eselect       )
    pet_reader   = partial(read_petsys_filebyfile, type_dict=ch_map.ch_type)
    outdir       = config.get('output', 'out_dir')
    dt_filenames = []
    for fn in file_list:
        channel_list, ref_ch, geom_dt = geom_func(file_name=fn)
        ## Specific filter depends on source position.
        filt   = evt_filt(valid_channels=channel_list)
        dt_gen = map(dt_calc(ref_ch=ref_ch, geom_dt=geom_dt),
                     pet_reader(sm_filter=filt)(fn)         )
        evt_df = pd.DataFrame(filter(lambda x: x, dt_gen), columns=df_cols)

        out_base   = fn.split(os.sep)[-1].replace('.ldat', '.feather')
        feath_name = os.path.join(outdir, out_base)
        evt_df.to_feather(feath_name)
        dt_filenames.append(feath_name)
    return dt_filenames


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

    # ncpu = mp.cpu_count()
    ncpu = cpu_count()
    if ncores > ncpu:
        print(f'Too many cores requested ({ncores}), only {ncpu} available.')
        print('Will use half available cores.')
        ncores = ncpu // 2

    # Try without glob...
    input_files = args['INFILES']
    outdir      = conf.get('output', 'out_dir')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    total_iter  = first_it + niter
    out_skew    = input_files[0].split(os.sep)[-1].split('_')[0] + f'_skew_{total_iter}iter.txt'
    skew_file   = os.path.join(outdir, out_skew)
    elec_cols   = ['#portID', 'slaveID', 'chipID', 'channelID']
    if first_it != 0:
        # Probably want to update formats?
        skew_vals       = pd.read_csv(skew_file, sep='\t')
        skew_vals['id'] = skew_vals.apply(lambda r: get_absolute_id(*r[elec_cols]),
                                          axis=1).astype('int')
        skew_vals       = skew_vals.set_index('id')['tOffset (ps)']
        # Let's not overwrite existing outputs.
        skew_file = skew_file.replace('.txt', '_recalc.txt')
    else:
        ch_map    = ChannelMap(conf.get('mapping', 'map_file'))
        all_ids   = ch_map.mapping.index
        skew_vals = pd.Series(0                            ,
                              index = all_ids.sort_values(),
                              name  = 'tOffset (ps)'       )
        if not args['-r']:
            chunks = [(file_set, conf, ch_map)
                      for file_set in np.array_split(input_files, ncores)]
            with get_context('spawn').Pool(ncores) as p:
                filename_chunks = p.starmap(process_raw_data, chunks)
            input_files = np.concatenate(filename_chunks)
            print('Binary processing complete')

    # start iterations (will need to do monitor plots using i for iteration):
    for i in range(first_it, total_iter):
        print(f'Starting iteration {i}')
        skew_vals = calculate_skews(input_files, conf, skew_vals)

    ## Save the skew values, change to format expected for PETsys?
    print('Requested iterations complete, output text file.')
    val_col              = ['tOffset (ps)']
    skew_vals            = skew_vals.reset_index().rename(columns={'index': 'id', 0: val_col[0]})
    skew_vals[elec_cols] = np.row_stack(skew_vals.id.map(get_electronics_nums))
    skew_vals[elec_cols + val_col].round(3).to_csv(skew_file, sep='\t')
