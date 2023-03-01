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
# from pet_code.src.io      import read_ymlmapping
from pet_code.src.plots   import ChannelEHistograms
from pet_code.src.util    import ChannelType
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import select_module
from pet_code.src.util    import shift_to_centres


# Candidate to be the ChannelCal used in general
# class ChannelCal:
#     def __init__(self, time_ch, eng_ch, tbins, ebins, cal) -> None:
#         self.time_id     = time_ch
#         self.eng_id      = eng_ch
#         self.tbin_edges  = tbins
#         self.ebin_edges  = ebins
#         self.calib       = cal
#         self.tdist       = {}
#         self.tindx_edist = {}

#     def add_evt(self, evt):
#         t_count = 0
#         for sm in self.calib(evt):
#             if len(sm) == 0: continue
#             sel_mod = select_module(sm, self.eng_id)
#             max_tid = [-99, -99]
#             for t_chan in filter(lambda x: x[0] in self.time_id, sel_mod):
#                 bin_indx = np.searchsorted(self.tbin_edges, t_chan[3]) - 1
#                 if bin_indx < 0:
#                     continue
#                 t_count += 1
#                 try:
#                     self.tdist[t_chan[0]][bin_indx] += 1
#                 except KeyError:
#                     nbins = len(self.tbin_edges) - 1
#                     self.tdist[t_chan[0]] = np.zeros(nbins, int)
#                     if bin_indx < nbins:
#                         self.tdist[t_chan[0]][bin_indx] += 1
#                 except IndexError:
#                     # Outside chosen range
#                     pass
#                 if t_chan[3] > max_tid[1]:
#                     max_tid = [t_chan[0], t_chan[3]]
#             if max_tid[0] != -99:
#                 esum = sum(map(lambda x: x[3], filter(lambda y: y[0] in self.eng_id, sel_mod)))
#                 bin_indx = np.searchsorted(self.ebin_edges, esum) - 1
#                 if bin_indx < 0:
#                     continue
#                 try:
#                     self.tindx_edist[max_tid[0]][bin_indx] += 1
#                 except KeyError:
#                     nbins = len(self.ebin_edges) - 1
#                     self.tindx_edist[max_tid[0]] = np.zeros(nbins, int)
#                     if bin_indx < nbins:
#                         self.tindx_edist[max_tid[0]][bin_indx] += 1
#                 except IndexError:
#                     # Outside chosen range
#                     pass
#         return t_count
    

def cal_and_sel(cal_func, sel_func):
    def _cal_and_sel(evt):
        return tuple(map(sel_func, cal_func(evt)))
    return _cal_and_sel


def output_slab_plots(histos, cal_name, out_dir):
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
            *_, fit_pars, _ = fit_gaussian(dist, histos.edges[htype], min_peak=100)
            mu_vals .append(fit_pars[1])
            sig_vals.append(fit_pars[2])
        except RuntimeError:
            print(f'Fit failed for channel {id}')
            plt.errorbar(histos.edges[htype], dist, yerr=np.sqrt(dist))
            plt.show()
            plt.clf()
    
    ## Fit distributions
    bins = min(mu_vals) - 2, max(mu_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(mu_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(mu_vals)} +- {np.std(mu_vals, ddof=1)}')
    plt.xlabel('Time channel peak positions (keV)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_timeEngMu.png')))
    plt.clf()
    bins = min(sig_vals) - 2, max(sig_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(sig_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(sig_vals)} +- {np.std(sig_vals, ddof=1)}')
    plt.xlabel('Time channel peak sigmas (keV)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_timeEngSig.png')))
    plt.clf()

    bcent, gvals, pars, _ = fit_gaussian(slab_sum, histos.edges[htype])
    plt.errorbar(bcent, slab_sum, yerr=np.sqrt(slab_sum), label='Binned values')
    plt.plot(bcent, gvals, label=f'Fit: mu = {pars[1]}, sigma = {pars[2]}')
    plt.xlabel('Time channel energy (keV)')
    plt.ylabel('AU')
    plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_timeAllDist.png')))
    plt.clf()
    ##


def output_energy_plots(histos, cal_name, out_dir, setup, no_super):
    """
    Make the plots for the energy channels.
    """
    fig_cols = 2        if 'brain' in setup else 4
    fig_rows = 8        if 'brain' in setup else 4
    psize    = (20, 15) if 'brain' in setup else (15, 15)
    all_eng  = np.zeros(histos.nbin_sum, int)

    mu_vals  = {}
    sig_vals = {}
    fig_ax   = [plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=psize)
                for _ in range(no_super)]
    for id, dist in histos.sum_dist.items():
        sm, mm = id

        all_eng += dist
        bcent, gvals, pars, _ = fit_gaussian(dist, histos.edges['ESUM'])
        fig_ax[sm][1].flatten()[mm].errorbar(bcent                ,
                                             dist                 ,
                                             yerr  = np.sqrt(dist),
                                             label = 'dataset'    )
        fig_ax[sm][1].flatten()[mm].plot(bcent, gvals, label=f'Fit: mu = {pars[1]}, sigma = {pars[2]}')
        fig_ax[sm][1].flatten()[mm].set_xlabel(f'mM {mm} energy sum (au)')
        fig_ax[sm][1].flatten()[mm].set_ylabel('AU')
        fig_ax[sm][1].flatten()[mm].legend()
        mu_vals [id] = pars[1]
        sig_vals[id] = pars[2]
    for i, (fig, _) in enumerate(fig_ax):
        out_name = os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_MMEngs_sm{i}.png'))
        fig.savefig(out_name)

    ## Fit distributions
    htype = 'ESUM'
    bins = min(mu_vals) - 2, max(mu_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(mu_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(mu_vals)} +- {np.std(mu_vals, ddof=1)}')
    plt.xlabel('Minimodule energy channel peak positions (au)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_mmEngMu.png')))
    plt.clf()
    bins = min(sig_vals) - 2, max(sig_vals) + 2, np.diff(histos.edges[htype][:2])[0]
    plt.hist(sig_vals, bins=np.arange(*bins),
             label=f'mean = {np.mean(sig_vals)} +- {np.std(sig_vals, ddof=1)}')
    plt.xlabel('Minimodule energy channel peak sigmas (au)')
    plt.ylabel('AU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_mmEngSig.png')))
    plt.clf()

    bcent, gvals, pars, _ = fit_gaussian(all_eng, histos.edges[htype])
    plt.errorbar(bcent, all_eng, yerr=np.sqrt(all_eng), label='Binned values')
    plt.plot(bcent, gvals, label=f'Fit: mu = {pars[1]}, sigma = {pars[2]}')
    plt.xlabel('All MM sum energy (keV)')
    plt.ylabel('AU')
    plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_engAllDist.png')))
    plt.clf()
    ##


if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    infiles  = args['INPUT']

    map_file = conf.get('mapping', 'map_file')
    chan_map = ChannelMap(map_file)
    num_sm   = chan_map.mapping.supermodule.unique().shape[0]
    # time_ch, eng_ch, mm_map, *_ = read_ymlmapping(map_file)

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

    ebins = np.arange(*map(float, conf.get('output', 'ebinning', fallback='0,300,1.5').split(',')))
    tbins = np.arange(*map(float, conf.get('output', 'tbinning', fallback='9,25,0.2') .split(',')))
    sm_no = (1, 3)
    out_dir = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    reader  = read_petsys_filebyfile(chan_map.ch_type, sm_filter=evt_filter, singles=singles)
    cal_sel = cal_and_sel(cal_func, select_module(chan_map.get_minimodule))
    for fn in infiles:
        print(f'Reading file {fn}')
        # plotter  = ChannelCal(time_ch, eng_ch, tbins, ebins, cal_func)
        plotter = ChannelEHistograms(tbins, ebins, ebins)
        for evt in map(cal_sel, reader(fn)):
            sm_mm   = tuple(map(lambda i: (chan_map.get_supermodule(i[0][0]), chan_map.get_minimodule(i[0][0])), evt))
            plotter.add_all_energies(evt, sm_mm)
        # num_time = list(map(plotter.add_evt, reader(fn)))
        # print(f'Time channels per selected module in file {fn}: {Counter(num_time)}')
        output_slab_plots  (plotter, cal_name, out_dir)
        output_energy_plots(plotter, cal_name, out_dir, map_file, num_sm)
        # print('Checking time channel energy distributions...')
        # mu_vals  = []
        # sig_vals = []
        # for id, dist in plotter.tdist.items():
        #     try:
        #         *_, fit_pars, _ = fit_gaussian(dist, plotter.tbin_edges, min_peak=100)
        #         mu_vals .append(fit_pars[1])
        #         sig_vals.append(fit_pars[2])
        #     except RuntimeError:
        #         print(f'Fit failed for channel {id}')
        #         plt.errorbar(plotter.tbin_edges[:-1], dist, yerr=np.sqrt(dist))
        #         plt.show()
        #         plt.clf()
        
        # ## Fit distributions
        # plt.hist( mu_vals, bins=np.arange(min( mu_vals) - 2, max( mu_vals) + 2, np.diff(plotter.tbin_edges[:2])[0]),
        #          label=f'mean = {np.mean(mu_vals)} +- {np.std(mu_vals, ddof=1)}')
        # plt.xlabel('Time channel peak positions (keV)')
        # plt.ylabel('AU')
        # plt.legend()
        # plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_timeEngMu.png')))
        # plt.clf()
        # plt.hist(sig_vals, bins=np.arange(min(sig_vals) - 2, max(sig_vals) + 2, np.diff(plotter.tbin_edges[:2])[0]),
        #          label=f'mean = {np.mean(sig_vals)} +- {np.std(sig_vals, ddof=1)}')
        # plt.xlabel('Time channel peak sigmas (keV)')
        # plt.ylabel('AU')
        # plt.legend()
        # plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_timeEngSig.png')))
        # plt.clf()
        # ##
        # mm_figs = [plt.subplots(nrows=4, ncols=4, figsize=(15, 15)),
        #            plt.subplots(nrows=4, ncols=4, figsize=(15, 15))]
        # mm_sums = {}
        # for id, dist in plotter.tindx_edist.items():
        #     sm_indx = 0 if id < 256 else 1#Temp?
        #     mm      = mm_map[id]
        #     r       = (mm - 1) // 4
        #     c       = (mm - 1) %  4
        #     mm_figs[sm_indx][1][r, c].errorbar(shift_to_centres(plotter.ebin_edges),
        #                                        dist, yerr=np.sqrt(dist),
        #                                        label=f'tmax id {id}')
        #     try:
        #         mm_sums[(sm_indx, mm)] += dist
        #     except KeyError:
        #         mm_sums[(sm_indx, mm)]  = dist
        # for i, (fig, axes) in enumerate(mm_figs):
        #     for j, ax in enumerate(axes.flatten()):
        #         ax.set_xlabel(f'MM{j+1} Energy sum (au)')
        #         ax.set_ylabel('Frequency per bin (au)')
        #         ax.legend()
        #     out_name = os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_MMEngs_sm{sm_no[i]}.png'))
        #     fig.savefig(out_name)
        # plt.clf()
        # mu_vals  = []
        # sig_vals = []
        # for key, dist in mm_sums.items():
        #     try:
        #         *_, fit_pars, _ = fit_gaussian(dist, plotter.ebin_edges, min_peak=100)
        #         mu_vals .append(fit_pars[1])
        #         sig_vals.append(fit_pars[2])
        #     except RuntimeError:
        #         print(f'Energy sum fit failed for sm {key[0]}, mm {key[1]}')
        # ## Fit distributions
        # plt.hist( mu_vals, bins=np.arange(min( mu_vals) - 2, max( mu_vals) + 2, np.diff(plotter.ebin_edges[:2])[0]),
        #          label=f'mean = {np.mean(mu_vals)} +- {np.std(mu_vals, ddof=1)}')
        # plt.xlabel('Energy sum peak positions (au)')
        # plt.ylabel('AU')
        # plt.legend()
        # plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_EngMu.png')))
        # plt.clf()
        # plt.hist(sig_vals, bins=np.arange(min(sig_vals) - 2, max(sig_vals) + 2, np.diff(plotter.ebin_edges[:2])[0]),
        #          label=f'mean = {np.mean(sig_vals)} +- {np.std(sig_vals, ddof=1)}')
        # plt.xlabel('Energy sum peak sigmas (au)')
        # plt.ylabel('AU')
        # plt.legend()
        # plt.savefig(os.path.join(out_dir, fn.split('/')[-1].replace('.ldat', f'{cal_name}_EngSig.png')))
        # plt.clf()
        # ##
            


