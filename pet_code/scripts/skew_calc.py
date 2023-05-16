#!/usr/bin/env python3

"""Calculate the skew for each channel in a two super module set-up

Usage: skew_calc.py (--conf CONFFILE) [-n NCORE] [--it NITER] [--firstit FITER] [--sk SKEW] [-r] INFILES ...

Arguments:
    INFILES File(s) to be processed.

Required:
    --conf=CONFFILE  Configuration file for run.

Options:
    -n=NCORE         Number of cores for processing [default: 2]
    --it=NITER       Number of iterations over data [default: 3]
    --firstit=FITRE  Iteration at which to start    [default: 0]
    --sk=SKEW        Existing skew file to be used if FITEr != 0.
    -r               Recalculate from 0 assuming dt files available.
"""

import os
import configparser
import yaml

from docopt import docopt
from typing import Callable

import numpy  as np
import pandas as pd

from functools import partial

import matplotlib.pyplot as plt

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
from pet_code.src.util    import read_skewfile
from pet_code.src.util    import select_energy_range
from pet_code.src.util    import select_module
from pet_code.src.util    import shift_to_centres


def source_position(pos_conf: dict) -> Callable:
    pos_to_mm  = pos_conf[ 'pos_to_mm']
    source_pos = pos_conf['source_pos']
    def _source_position(pos_num: int) -> tuple[int, list, np.ndarray]:
        sm, mms = pos_to_mm[pos_num]
        return sm, mms, np.asarray(source_pos[pos_num])
    return _source_position


def get_references(file_name: str, source_p: Callable) -> tuple[int, list, np.ndarray]:
    """
    Extract supermodule and minimodule
    numbers for reference channels and
    the source position.
    #correct allows backward compatibility for old numbering
    """
    file_label  = 'SourcePos'#Assume this is followed by position number
    source_indx = file_name.find(file_label)
    if source_indx == -1:
        file_name_parts = file_name.split(os.sep)[-1].split('_')
        SM_lab          = int(file_name_parts[1][ 8:9])
        source_posNo    = int(file_name_parts[1][12: ])
        pos_no          = int(str(SM_lab) + '00' + str(source_posNo).zfill(2))
        sm, mms, s_pos  = source_p(pos_no)
        return sm, mms, s_pos
    source_pnum = int(file_name[source_indx + len(file_label):].split('_')[0])
    SM_lab, mM_nums, s_pos = source_p(source_pnum)
    return SM_lab, mM_nums, s_pos


def geom_loc_point(file_name: str, ch_map: ChannelMap, source_yml: dict) -> tuple:
    """
    Get functions for the reference index
    and geometric dt for the 2 SM setup.
    REVIEW, a lot of inversion to adapt!
    """
    (SM_lab ,
     mM_nums,
     s_pos  ) = get_references(file_name, source_position(source_yml))
    mm_channels = np.concatenate([ch_map.get_minimodule_channels(SM_lab, mm)
                                  for mm in mM_nums])
    valid_ids   = set(filter(lambda x: ch_map.get_channel_type(x) is ChannelType.TIME, mm_channels))
    def find_indx(evt: tuple) -> int:
        return 0 if evt[0][0] in valid_ids else 1

    flight_time = time_of_flight(s_pos)
    def geom_dt(ref_id: int, coinc_id: int) -> float:
        ref_pos   = ch_map.get_channel_position(  ref_id)
        coinc_pos = ch_map.get_channel_position(coinc_id)
        return flight_time(ref_pos) - flight_time(coinc_pos)
    return mm_channels, find_indx, geom_dt


def geom_loc_bar(file_name: str, ch_map: ChannelMap, source_yml: dict) -> tuple:
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
    def find_indx(evt: tuple) -> int:
        return 0 if evt[0][0] in s_ids else 1

    geom_dt = bar_source_dt(bar_xy, bar_r, ch_map.get_channel_position)
    return ids, find_indx, geom_dt


def process_raw_data(file_list: list[str]                ,
                     config   : configparser.ConfigParser,
                     ch_map   : ChannelMap
                     ) -> list[str]:
    """
    Read the binaries corresponding to file_list,
    filter selecting events in the slab spectra
    limits, calculate geometrically corrected deltaT
    and output to feather files.
    """
    ## This needs to be improved. In principle it can be in the map.
    time_cal = config.get     ('calibration',   'time_channels' , fallback='')
    eng_cal  = config.get     ('calibration', 'energy_channels' , fallback='')
    eref     = config.getfloat('calibration', 'energy_reference', fallback=None)
    cal_func = calibrate_energies(ch_map.get_chantype_ids, time_cal, eng_cal, eref=eref)
    sel_mod  = select_module(ch_map.get_minimodule)

    def cal_and_select(evt: tuple) -> tuple:
        cal_evt = cal_func(evt)
        return tuple(map(sel_mod, cal_evt))

    elimits  = map(float, config.get('filter', 'elimits').split(','))
    eselect  = select_energy_range(*elimits)

    setup    = config.get('mapping', 'setup', fallback='pointSource')
    minch    = config.getint('filter', 'min_channels')
    evt_filt = partial(filter_impacts_channel_list   ,
                       min_ch = minch                ,
                       mm_map = ch_map.get_minimodule)
    if   setup == 'pointSource':
        ## For backwards compatibility.
        with open(config.get('mapping', 'source_pos')) as s_yml:
            yml_positions = yaml.safe_load(s_yml)
            geom_func     = partial(geom_loc_point            ,
                                    ch_map     = ch_map       ,
                                    source_yml = yml_positions)
    elif setup == 'barSource'  :
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


def calculate_skews(file_list  : list[str]                ,
                    config     : configparser.ConfigParser,
                    skew_values: pd.Series                ,
                    it         : int=0
                    ) -> pd.Series:
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
    try:
        monitor_id = set(map(int, config.get('output', 'mon_ids').split(',')))
    except (configparser.NoSectionError, configparser.NoOptionError):
        monitor_id = {}
    out_name   = os.path.join(*file_list[0].split(os.sep)[:-1], f'corrected_dt_it{it}_')
    bias_calc  = peak_position(np.arange(*hist_bins), min_stats,
                               skew_values, monitor_id, out_name)
    for fn in file_list:
        dt_df      = pd.read_feather(fn)
        biases     = dt_df.groupby('ref_ch', group_keys=False).apply(bias_calc)
        corr_skews = corr_skews.add(rel_fact * biases, fill_value=0.0)
    return corr_skews


def peak_position(hist_bins: np.ndarray             ,
                  min_stats: int                    ,
                  skew     : pd.Series              ,
                  mon_ids  : dict[int] = {}         ,
                  out_base : str      = 'dt_monitor'
                  ) -> Callable:
    """
    Calculate the mean position of the delta time distribution
    corrected for theoretical difference and, optionally, skew.
    """
    def calculate_bias(delta_t: pd.DataFrame) -> float:
        """
        Do the calculation for every row of delta t.
        Assumes that the ref_ch is the same for all.
        """
        ref_ch    = delta_t.ref_ch.unique()[0]
        ref_skew  = skew.get(ref_ch, 0)
        skew_corr = skew.loc[delta_t.coinc_ch].values - ref_skew

        bin_vals, bin_edges = np.histogram(delta_t.corr_dt.values + skew_corr, bins=hist_bins)
        if ref_ch in mon_ids:
            output_plot(ref_ch, bin_vals, bin_edges, '_'.join((out_base, f'ch{ref_ch}.png')))
        try:
            *_, pars, _, _ = fit_gaussian(bin_vals, bin_edges, min_peak=min_stats)
        except RuntimeError:
            print(f'Ref channel {ref_ch} fit fail', flush=True)
            peak_mean, *_ = mean_around_max(bin_vals, shift_to_centres(bin_edges), 6)
            return peak_mean if peak_mean else 0
        return pars[1]
    return calculate_bias


def output_plot(id         : int       ,
                dt_data    : np.ndarray,
                dt_binedges: np.ndarray,
                plot_file  : str
                ) -> None:
    """
    Plots and saves a corrected dt distribution.
    """
    plt.errorbar(shift_to_centres(dt_binedges), dt_data, yerr=np.sqrt(dt_data))
    plt.xlabel(f'$dt_r$ - $dt_t$ for slab {id} (ps)')
    plt.ylabel('Bin content (AU)')
    plt.savefig(plot_file)
    plt.clf()


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
        skew_vals = read_skewfile(args['--sk'])
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
        skew_vals = calculate_skews(input_files, conf, skew_vals, i)

    ## Save the skew values, change to format expected for PETsys?
    print('Requested iterations complete, output text file.')
    val_col              = ['tOffset (ps)']
    skew_vals            = skew_vals.reset_index().rename(columns={'index': 'id', 0: val_col[0]})
    skew_vals[elec_cols] = np.row_stack(skew_vals.id.map(get_electronics_nums))
    skew_vals[elec_cols + val_col].round(3).to_csv(skew_file, sep='\t')
