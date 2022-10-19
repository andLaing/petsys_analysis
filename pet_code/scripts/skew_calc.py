#!/usr/bin/env python3

"""Calculate the skew for each channel in a two super module set-up

Usage: skew_calc.py (--conf CONFFILE) [-n NCORE] [--it NITER] INBASE

Arguments:
    INBASE File pattern to be matched (+ *.ldat) for input.

Required:
    --conf=CONFFILE  Configuration file for run.

Options:
    -n=NCORE   Number of cores for processing [default: 2]
    --it=NITER Number of iterations over data [default: 3]
"""

from enum import unique
import os
import configparser

from docopt import docopt
from glob   import glob

import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt

import multiprocessing as mp

from pet_code.src.fits  import fit_gaussian
from pet_code.src.io    import read_petsys_filebyfile
from pet_code.src.io    import read_ymlmapping
from pet_code.src.plots import group_times_slab
from pet_code.src.plots import slab_energy_spectra
from pet_code.src.util  import centroid_calculation
from pet_code.src.util  import time_of_flight
from pet_code.src.util  import filter_impacts_specific_mod
from pet_code.src.util  import slab_energy_centroids


# Source positions. Improve!!
def source_position(sm_num, mm_num):
    """
    Hard wired source positions!
    """
    x = 103.6 - 25.9 * (0.5 - (mm_num - 1) // 4)
    if sm_num == 3:
        y = -25.9 + (3 - (mm_num - 1) % 4) * 25.9
        z = 123.7971 - 31.8986
        return np.array([x, y, z])
    y = -25.9 - (mm_num - 1) % 4 * 25.9
    z = 31.8986
    return np.array([x, y, z])


def get_references(file_name):
    """
    Extract supermodule and minimodule
    numbers for reference channels and
    the source position.
    DUMMY!
    """
    # Get them from the filename?
    # Source pos from some saved lookup!
    file_name_parts = file_name.split('/')[-1].split('_')
    SM_lab = int(file_name_parts[1][2:])
    SM_indx = 0 if SM_lab == 3 else 1
    mM_num = int(file_name_parts[2][2:])
    return SM_indx, mM_num, source_position(SM_lab, mM_num)#[38.4, 38.4, 22.5986]


def read_and_select(file_list, config):
    """
    Read the binaries corresponding to file_list,
    filter on the slab spectra, group and return
    raw channels and timestamps.
    Will use parallelization but start simple for
    now.
    """
    (time_ch, eng_ch, mm_map,
     centroid_map, slab_map) = read_ymlmapping(config.get('mapping', 'map_file'))
    outdir = config.get('output', 'out_dir')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    sm1_minch, sm2_minch = tuple(map(int, config.get('filter', 'min_channels').split(', ')))
    c_calc = centroid_calculation(centroid_map)

    all_skews = pd.Series(dtype=float)
    for fn in file_list:
        print(f'Processing file {fn}')
        sm_num, mm_num, source_pos = get_references(fn)
        evt_filter = filter_impacts_specific_mod(sm_num, mm_num, eng_ch, sm1_minch, sm2_minch)
        evt_reader = read_petsys_filebyfile(fn, mm_map, evt_filter)
        out_base   = os.path.join(outdir, fn.split('/')[-1])
        skew_calc  = get_skew(time_of_flight(source_pos), slab_map, plot_output=out_base)

        sel_evts   = [evt for evt in evt_reader()]
        print('Events read. Proceeding...')
        slab_dicts = slab_energy_centroids(sel_evts, c_calc, time_ch)

        photo_peak = list(map(slab_energy_spectra, slab_dicts))

        reco_dt    = group_times_slab(sel_evts, photo_peak, time_ch, sm_num)
        ## Test pandas output at this point. Should facilitate further iterations.
        deltat_df = pd.concat((pd.DataFrame(vals                     ,
                                            index   = [key]*len(vals),
                                            columns = ['coinc_ch'  ,
                                                       'coinc_tstp',
                                                       'ref_tstp'  ] )
                               for key, vals in reco_dt.items()       ))
        deltat_df.reset_index(inplace=True)
        deltat_df.rename(inplace=True, columns={'index': 'ref_ch'})
        deltat_df.to_pickle(out_base.replace('.ldat', '_dtFrame.pkl'))
 
        skew_values = deltat_df.groupby('ref_ch', group_keys=False).apply(skew_calc)
        all_skews = pd.concat((all_skews, skew_values))
    return all_skews


def time_distributions(file_list, config, skew_values, it):
    """
    Read from the pickle files of the
    coincidence time frames, make the
    dt distributions correcting for
    skew and fit for current skew.
    """
    corr_skews = skew_values.copy()
    outdir = config.get('output', 'out_dir')
    *_, slab_map = read_ymlmapping(config.get('mapping', 'map_file'))
    for fn in file_list:
        *_, source_pos = get_references(fn)
        ## Probably want a function for the name so consistent
        out_base  = os.path.join(outdir, fn.split('/')[-1])
        skew_calc = get_skew(time_of_flight(source_pos),
                             slab_map                  ,
                             skew        = skew_values ,
                             plot_output = out_base    ,
                             it          = it          )
        pkl_name  = out_base.replace('.ldat', '_dtFrame.pkl')
        deltat_df = pd.read_pickle(pkl_name)
        corr_skews.add(deltat_df.groupby('ref_ch', group_keys=False).apply(skew_calc), fill_value=0.0)
    return corr_skews


def get_skew(flight_time, slab_map, skew=pd.Series(dtype=float), plot_output=None, it=0):
    """
    Get the skew value for a reference
    channel correcting for previously
    calculated skew if available.
    """
    # Shouldn't be hardwired!!
    hist_bins = np.linspace(-10000, 10000, 400, endpoint=False)
    def calc_skew(delta_t):
        ref_ch    = delta_t.ref_ch.unique()[0]
        dt_th     = flight_time(slab_map[ref_ch]) - np.fromiter((flight_time(slab_map[id]) for id in delta_t.coinc_ch), float)
        ref_skew  = skew.get(ref_ch, 0)
        skew_corr = np.fromiter((skew.get(id, 0) for id in delta_t.coinc_ch), float) - ref_skew
        dt_reco   = np.diff(delta_t[['coinc_tstp', 'ref_tstp']], axis=1).flatten()

        if plot_output:
            bin_vals, bin_edges, _ = plt.hist(dt_reco - dt_th + skew_corr, bins=hist_bins)
            plt.xlabel(f'$dt_r$ - $dt_t$ for slab {ref_ch} (ps)')
        else:
            bin_vals, bin_edges = np.histogram(dt_reco - dt_th + skew_corr, bins=hist_bins)
        try:
            bcent, gvals, pars, _ = fit_gaussian(bin_vals, bin_edges)
            if plot_output:
                plt.plot(bcent, gvals, label=f'fit $\mu$ = {round(pars[1], 3)},  $\sigma$ = {round(pars[2], 3)}')
                plt.legend()
                plt.savefig(plot_output.replace('.ldat', f'_it{it}_timeCoincRef{ref_ch}.png'))
                plt.clf()
        except RuntimeError:
            print("FUCKCKCKCKC")
            return 0
        return pars[1]
    return calc_skew




if __name__ == '__main__':
    args   = docopt(__doc__)
    ncores = int(args['-n'  ]) # For parallelization
    niter  = int(args['--it'])

    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    ncpu = mp.cpu_count()
    if ncores > ncpu:
        print(f'Too many cores requested ({ncores}), only {ncpu} available.')
        print('Will use half available cores.')
        ncores = ncpu // 2

    input_files = glob(args['INBASE'] + '*.ldat')
    print("File Checks: ", input_files)
    for i in range(niter):
        print(f'Start iteration {i}')
        if i == 0:
            ## Read the ldat binaries and do the first calculation.
            ## We definitely want to parallelize here.
            chunk_args = [(file_set, conf) for file_set in np.array_split(input_files, ncores)]
            with mp.Pool(ncores) as p:
                # Run chunks in parallel
                skew_chunks = p.starmap(read_and_select, chunk_args)
            skew_values = pd.concat(skew_chunks)
            # skew_values = read_and_select(input_files, conf)
        else:
            skew_values = time_distributions(input_files, conf,
                                             skew_values, i   )
    ## Save the skew values.
    print('Requested iterations complete, output text file.')
    skew_values = skew_values.reset_index().rename(columns={'index': 'Channel_id', 0: 'Skew'})
    skew_file   = os.path.join(conf.get('output', 'out_dir'), args['INBASE'].split('/')[-1].split('_')[0]) + '_skew.txt'
    skew_values.to_csv(skew_file)
    # (time_ch, eng_ch,
    #  mm_map, centroid_map, slab_map) = read_ymlmapping(conf.get('mapping', 'map_file'))

    # map_file  = 'pet_code/test_data/SM_mapping.yaml' # shouldn't be hardwired

    # file_list = sys.argv[1:]

    # time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)