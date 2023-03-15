#!/usr/bin/env python3

"""Make plots under given (conf) calibration conditions and return plots

Usage: cal_monitor.py (--conf CONFFILE) INPUT ...

Arguments:
    INPUT  File(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt import docopt

import matplotlib.pyplot as plt
import numpy             as np

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.fits    import fit_gaussian
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.plots   import ChannelEHistograms
from pet_code.src.util    import ChannelType
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import select_module
    

def cal_and_sel(cal_func, sel_func):
    def _cal_and_sel(evt):
        return tuple(map(sel_func, cal_func(evt)))
    return _cal_and_sel


def output_time_plots(histos, cal_name, out_dir, file_name, min_stats):
    """
    Make the energy plots for timeslabs.
    """
    print('Checking time channel energy distributions...')
    htype    = ChannelType.TIME
    mu_vals  = []
    sig_vals = []
    slab_sum = np.zeros(histos.nbin_time, int)
    for id, dist in histos.tdist.items():
        slab_sum += dist
        try:
            *_, fit_pars, _ = fit_gaussian(dist, histos.edges[htype], min_peak=min_stats)
            mu_vals .append(fit_pars[1])
            sig_vals.append(fit_pars[2])
        except RuntimeError:
            print(f'Fit failed for channel {id}')
            if __name__ == '__main__':
                plt.errorbar(histos.edges[htype][:-1], dist, yerr=np.sqrt(dist))
                plt.show()
                plt.clf()
    
    ## Fit distributions
    bins = min(mu_vals) - 2, max(mu_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(mu_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(mu_vals)} +- {np.std(mu_vals, ddof=1)}')
    plt.xlabel('Time channel peak positions (keV)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_timeEngMu.png')))
    plt.clf()
    bins = min(sig_vals) - 2, max(sig_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(sig_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(sig_vals)} +- {np.std(sig_vals, ddof=1)}')
    plt.xlabel('Time channel peak sigmas (keV)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_timeEngSig.png')))
    plt.clf()

    bcent, gvals, pars, _ = fit_gaussian(slab_sum, histos.edges[htype])
    plt.errorbar(bcent, slab_sum, yerr=np.sqrt(slab_sum), label='Binned values')
    plt.plot(bcent, gvals, label=f'Fit: mu = {pars[1]}, sigma = {pars[2]}')
    plt.xlabel('Time channel energy (keV)')
    plt.ylabel('AU')
    plt.savefig(os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_timeAllDist.png')))
    plt.clf()
    ##


def output_energy_plots(histos, cal_name, out_dir, file_name, setup, no_super):
    """
    Make the plots for the energy channels.
    """
    fig_cols = 2        if 'brain' in setup else 4
    fig_rows = 8        if 'brain' in setup else 4
    psize    = (20, 15) if 'brain' in setup else (15, 15)
    all_eng  = np.zeros(histos.nbin_sum, int)

    mu_vals  = []
    sig_vals = []
    fig_ax   = {k: plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=psize)
                for k in no_super}
    htype    = 'ESUM'
    txt_file = os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_MMEngPeaks.txt'))
    with open(txt_file, 'w') as peak_out:
        peak_out.write('Supermod\tMinimod\tEnergy Peak\tSigma\n')
        for id, dist in histos.sum_dist.items():
            sm, mm = id

            all_eng += dist
            bcent, gvals, pars, _ = fit_gaussian(dist, histos.edges[htype])
            fig_ax[sm][1].flatten()[mm].errorbar(bcent                ,
                                                 dist                 ,
                                                 yerr  = np.sqrt(dist),
                                                 label = 'dataset'    )
            fig_ax[sm][1].flatten()[mm].plot(bcent, gvals, label=f'Fit: mu = {pars[1]}, sigma = {pars[2]}')
            fig_ax[sm][1].flatten()[mm].set_xlabel(f'mM {mm} energy sum (au)')
            fig_ax[sm][1].flatten()[mm].set_ylabel('AU')
            fig_ax[sm][1].flatten()[mm].legend()
            mu_vals .append(pars[1])
            sig_vals.append(pars[2])
            peak_out.write(f'{sm}\t{mm}\t{pars[1]}\t{pars[2]}\n')

    for i, (fig, _) in fig_ax.items():
        out_name = os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_MMEngs_sm{i}.png'))
        fig.savefig(out_name)
    plt.clf()

    ## Fit distributions
    bins = min(mu_vals) - 2, max(mu_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(mu_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(mu_vals)} +- {np.std(mu_vals, ddof=1)}')
    plt.xlabel('Minimodule energy channel peak positions (au)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_mmEngMu.png')))
    plt.clf()
    bins = min(sig_vals) - 2, max(sig_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(sig_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(sig_vals)} +- {np.std(sig_vals, ddof=1)}')
    plt.xlabel('Minimodule energy channel peak sigmas (au)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_mmEngSig.png')))
    plt.clf()

    bcent, gvals, pars, _ = fit_gaussian(all_eng, histos.edges[htype])
    plt.errorbar(bcent, all_eng, yerr=np.sqrt(all_eng), label='Binned values')
    plt.plot(bcent, gvals, label=f'Fit: mu = {pars[1]}, sigma = {pars[2]}')
    plt.xlabel('All MM sum energy (keV)')
    plt.ylabel('AU')
    plt.savefig(os.path.join(out_dir, file_name.split('/')[-1].replace('.ldat', f'{cal_name}_engAllDist.png')))
    plt.clf()
    ##


if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    infiles  = args['INPUT']

    map_file = conf.get('mapping', 'map_file')
    chan_map = ChannelMap(map_file)
    sm_nums  = chan_map.mapping.supermodule.unique()

    min_chan   = conf.getint('filter', 'min_channels')
    singles    = 'coinc' not in infiles[0]
    evt_filter = filter_event_by_impacts(min_chan, singles=singles)

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_name = ''
    if       time_cal and     eng_cal:
        cal_name = '_calEngTime_'
    elif     time_cal and not eng_cal:
        cal_name = '_calTime_'
    elif not time_cal and     eng_cal:
        cal_name = '_calEng_'
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal)

    ebins   = np.arange(*map(float, conf.get('output', 'ebinning', fallback='0,300,1.5').split(',')))
    tbins   = np.arange(*map(float, conf.get('output', 'tbinning', fallback='9,25,0.2') .split(',')))
    out_dir = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    reader    = read_petsys_filebyfile(chan_map.ch_type, sm_filter=evt_filter, singles=singles)
    cal_sel   = cal_and_sel(cal_func, select_module(chan_map.get_minimodule))
    min_stats = conf.getint('filter', 'min_stats')
    for fn in infiles:
        print(f'Reading file {fn}')
        plotter = ChannelEHistograms(tbins, ebins, ebins)
        for evt in map(cal_sel, reader(fn)):
            sm_mm = tuple(map(lambda i: (chan_map.get_supermodule(i[0][0]),
                                         chan_map.get_minimodule (i[0][0])),
                              filter(lambda j: j, evt)                     ))
            plotter.add_all_energies(evt, sm_mm)

        output_time_plots  (plotter, cal_name, out_dir, fn, min_stats)
        output_energy_plots(plotter, cal_name, out_dir, fn, map_file, sm_nums)
