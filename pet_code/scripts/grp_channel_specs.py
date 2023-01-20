#!/usr/bin/env python3

"""Generate energy spectra for all channels that appear in a petsys group mode file

Usage: channel_specs.py [--out OUTFILE] (--conf CFILE) INPUT ...

Arguments:
    INPUT  Input file name or list of names

Required:
    --conf=CFILE  Name of the module mapping yml file

Options:
    --out=OUTFILE  Name base for output image files.
"""

import os
import configparser

from docopt      import docopt
from collections import Counter

import matplotlib.pyplot as plt
import numpy             as np

from scipy.signal import find_peaks

from pet_code.src.fits  import fit_gaussian
from pet_code.src.io    import ChannelMap
from pet_code.src.io    import read_petsys_filebyfile
from pet_code.src.io    import read_ymlmapping
from pet_code.src.plots import ChannelEHistograms
from pet_code.src.util  import ChannelType
from pet_code.src.util  import filter_event_by_impacts
from pet_code.src.util  import select_mod_wrapper
from pet_code.src.util  import select_module
from pet_code.src.util  import shift_to_centres


def slab_plots(out_file, plot_source, plot_wosource, bin_edges, min_stats):
    with open(out_file + 'timeSlabPeaks.txt', 'w') as par_out:
        par_out.write('ID\tMU\tMU_ERR\tSIG\tSIG_ERR\n')
        for id, engs in plot_source.slab_energies.items():
            s_vals, bin_e = np.histogram(engs, bins=bin_edges)
            try:
                ns_vals, _ = np.histogram(plot_wosource.slab_energies[id], bins=bin_edges)
            except KeyError:
                plt.plot(bin_e[:-1], s_vals , label='Source')
                plt.legend()
                plt.xlabel(f'Energy time channel {id}')
                plt.ylabel('au')
                plt.savefig(out_file + f'NoSourceZero_ch{id}.png')
                plt.clf()
                continue
            bin_errs  = np.sqrt(s_vals + ns_vals)
            diff_data = s_vals - ns_vals
            try:
                bcent, g_vals, fit_pars, cov = fit_gaussian(diff_data, bin_e, yerr=bin_errs, min_peak=min_stats)
                ## hack
                if fit_pars[1] <= bin_e[3]:
                    raise RuntimeError
            except RuntimeError:
                print(f'Failed fit for channel {id}')
                plt.errorbar(bin_e[:-1], diff_data, yerr=bin_errs)
                plt.show()
                min_x = input('Do you want to try a refit? [n]/min_pos')
                if min_x:
                    bin_wid   = np.diff(bin_e[:2])[0]
                    indx      = int(float(min_x) / bin_wid - bin_e[0])
                    diff_data = s_vals[indx:] - ns_vals[indx:]
                    bin_errs = np.sqrt(s_vals[indx:] + ns_vals[indx:])
                    try:
                        bcent, g_vals, fit_pars, cov = fit_gaussian(diff_data, bin_e[indx:], yerr=bin_errs, min_peak=int(min_peak * 0.7))
                    except RuntimeError:
                        fail_plot(bin_e, s_vals, ns_vals, np.sqrt(s_vals + ns_vals))
                        continue
                else:
                    fail_plot(bin_e, s_vals, ns_vals, bin_errs)
                    continue
            mu_err  = np.sqrt(cov[1, 1])
            sig_err = np.sqrt(cov[2, 2])
            par_out.write(f'{id}\t{round(fit_pars[1], 3)}\t{round(mu_err, 3)}\t{round(fit_pars[2], 3)}\t{round(sig_err, 3)}\n')
            plt.errorbar(bcent, diff_data, yerr=bin_errs, label='distribution')
            plt.plot(bcent, g_vals, label=f'Gaussian fit $\mu = {round(fit_pars[1], 2)}, \sigma = {round(fit_pars[2], 2)}$')
            plt.legend()
            plt.xlabel(f'Energy time channel {id}')
            plt.ylabel('source spec - no source spec (au)')
            plt.savefig(out_file + f'BackRest_ch{id}.png')
            plt.clf()


def sum_plots(out_file, plot_source, esum_bins):
    mm_figs = [plt.subplots(nrows=4, ncols=4, figsize=(15, 15)),
               plt.subplots(nrows=4, ncols=4, figsize=(15, 15))]
    for tid, engs in plot_source.eng_sum.items():
        sm_indx = 0 if tid < 256 else 1#Temp?
        mm      = mm_map[tid]
        r       = (mm - 1) // 4
        c       = (mm - 1) %  4
        mm_figs[sm_indx][1][r, c].hist(engs, bins=esum_bins, histtype='step', label=f'tmax id {tid}')
    for i, (fig, axes) in enumerate(mm_figs):
        for j, ax in enumerate(axes.flatten()):
            ax.set_xlabel(f'MM{j+1} Energy sum (au)')
            ax.set_ylabel('Frequency per bin (au)')
            ax.legend()
        sm_no = 1 if i == 0 else 3
        fig.savefig(out_file + f'MMEngs_sm{sm_no}.png')
    plt.clf()


def energy_plots(out_file, plot_source, plot_wosource, bin_edges):
    with open(out_file + 'eChannelPeaks.txt', 'w') as par_out:
        par_out.write('ID\tMU\tMU_ERR\n')
        for id, engs in plot_source.eng_max.items():
            vals, edges, _ = plt.hist(engs, bins=ebins, histtype='step', label='Source')
            try:
                valsNS, *_ = plt.hist(plot_wosource.eng_max[id], bins=ebins, histtype='step', label='No Source')
            except KeyError:
                valsNS = np.zeros_like(vals)

            bin_cent  = shift_to_centres(edges)
            hdiff     = vals - valsNS
            hdiff_err = np.sqrt(vals + valsNS)
            if peaks.shape[0] > 1:
                peak   = np.argmax(hdiff[peaks])
                p_indx = peaks[peak]
            elif peaks.shape[0] == 0:
                p_indx = np.argmax(hdiff)
            else:
                p_indx = peaks[0]
            ## Try to protect against noise floor
            nbin_fit = 5
            if p_indx <= 3 * nbin_fit:
                print(f'Peak near min for channel {id}')
                plt.errorbar(bin_cent, hdiff, yerr=hdiff_err)
                plt.show()
                plt.clf()
                min_x = input('Do you want to adjust? [n]/peak_pos ')
                if min_x:
                    p_indx = np.searchsorted(edges, float(min_x), side='right') - 1
            plt.errorbar(bin_cent, hdiff, yerr=hdiff_err, label='Difference')
            peaks, _  = find_peaks(hdiff, height=100, distance=5)
            plt.plot(bin_cent[peaks], hdiff[peaks], 'rv', markersize=15, label="Peak finder")
            mask          = (bin_cent > bin_cent[p_indx] - 5 * bin_wid) & (bin_cent < bin_cent[p_indx] + 5 * bin_wid)
            av_diff, wsum = np.average(bin_cent[mask], weights=hdiff[mask], returned=True)
            av_err        = average_error(bin_cent[mask], hdiff[mask], hdiff_err[mask], wsum)
            plt.axvspan(av_diff - av_err, av_diff + av_err, facecolor='#00FF00' , alpha = 0.3, label='Average diff')
            plt.xlabel(f'Energy E channel {id}')
            plt.ylabel('Frequency per bin (au)')
            plt.legend()

            par_out.write(f'{id}\t{round(av_diff, 3)}\t{round(av_err, 3)}\n')
            plt.savefig(out_file + f'Emax_ch{id}.png')
            plt.clf()


def channel_plots(config, infiles):
    """
    Channel calibration function.
    """
    map_file = config.get('mapping', 'map_file')
    chan_map = ChannelMap(map_file)

    min_chan  = tuple(map(int, config.get('filter', 'min_channels').split(',')))
    filt      = filter_event_by_impacts(*min_chan, singles=True)

    esum_bins = np.arange(*tuple(map(float, config.get('output', 'esum_binning', fallback='0,300,1.5').split(','))))
    tbins     = np.arange(*tuple(map(float, config.get('output',     'tbinning', fallback='5,30,0.2') .split(','))))
    ebins     = np.arange(*tuple(map(float, config.get('output',     'ebinning', fallback='7,40,0.4') .split(','))))

    plotS     = ChannelEHistograms(tbins, ebins, esum_bins)
    plotNS    = ChannelEHistograms(tbins, ebins, esum_bins)
    reader    = read_petsys_filebyfile(chan_map.ch_type, sm_filter=filt, singles=True)
    for fn in infiles:
        print(f'Reading {fn}')
        if 'wo' in fn:
            add_evt_wp = select_mod_wrapper(plotNS.add_emax_evt, chan_map.get_minimodule)
            num_mmsN = tuple(map(add_evt_wp, reader(fn)))
        else:
            add_evt_wp = select_mod_wrapper(plotS .add_emax_evt, chan_map.get_minimodule)
            num_mmsS = tuple(map(add_evt_wp, reader(fn)))
    
    print("First pass complete, mm multiplicities with    Source: ", Counter(num_mmsS))
    print("First pass complete, mm multiplicities without Source: ", Counter(num_mmsN))
    return plotS, plotNS


class ChannelCal:
    def __init__(self, time_ch, eng_ch, mm_map) -> None:
        self.slab_energies = {}
        self.eng_sum       = {}# Mod eng channel sum with max slab key
        self.eng_max       = {}# Eng channel spectra for max eng channel.
        self.time_id       = time_ch
        self.eng_id        = eng_ch
        self.sel_mod       = select_module(mm_map)

    def add_evt(self, supermods):
        mms     = {ch[1] for ch in supermods[0]}
        sel_mod = self.sel_mod(supermods[0])
        try:
            t_ch = next(filter(lambda ch: ch[0] in self.time_id, sel_mod))
        except StopIteration:
            return len(mms)
        try:
            self.slab_energies[t_ch[0]].append(t_ch[3])
        except KeyError:
            self.slab_energies[t_ch[0]] = [t_ch[3]]
        e_chans = list(map(lambda x: x[3] if x[0] in self.eng_id else 0, sel_mod))
        try:
            self.eng_sum[t_ch[0]].append(sum(e_chans))
        except KeyError:
            self.eng_sum[t_ch[0]] = [sum(e_chans)]
        try:
            max_eng = sel_mod[np.argmax(e_chans)]
            self.eng_max[max_eng[0]].append(max_eng[3])
        except KeyError:
            self.eng_max[max_eng[0]] = [max_eng[3]]
        return len(mms)


def average_error(x, y, yerr, ysum):
    bin_err = x / 2# Half bin width as error
    x_cont  = np.sum((y * bin_err)**2) / ysum**2
    y_cont  = np.sum(((x * ysum - y * x) * yerr)**2) / ysum**4
    return np.sqrt(x_cont + y_cont)


def fail_plot(bin_e, s_vals, ns_vals, bin_errs):
    plt.plot(bin_e[:-1], s_vals , label='Source')
    plt.plot(bin_e[:-1], ns_vals, label='No Source')
    plt.errorbar(bin_e[:-1], s_vals - ns_vals, yerr=bin_errs, label='difference')
    plt.legend()
    plt.xlabel(f'Energy time channel {id}')
    plt.ylabel('au')
    plt.savefig(out_file + f'FailFit_ch{id}.png')
    plt.clf()


if __name__ == '__main__':
    args     = docopt(__doc__)
    conf     = configparser.ConfigParser()
    conf.read(args['--conf'])
    out_file = args['--out' ]
    infiles  = args['INPUT' ]

    if out_file is None:
        out_file = conf.get('output', 'out_file', fallback='slabSpec')
    out_dir  = os.path.join(*out_file.split('/')[:-1])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    plotS, plotNS = channel_plots(conf, infiles)

    esum_bins = np.arange(*tuple(map(float, conf.get('output', 'esum_binning', fallback='0,300,1.5').split(','))))
    tbins     = np.arange(*tuple(map(float, conf.get('output',     'tbinning', fallback='5,30,0.2') .split(','))))
    ebins     = np.arange(*tuple(map(float, conf.get('output',     'ebinning', fallback='7,40,0.4') .split(','))))
    min_peak  = conf.getint('filter', 'min_stats', fallback=300)

    # Plotting and fitting.
    slab_plots  (out_file, plotS, plotNS, tbins, min_peak)
    sum_plots   (out_file, plotS, esum_bins)
    energy_plots(out_file, plotS, plotNS, ebins)
