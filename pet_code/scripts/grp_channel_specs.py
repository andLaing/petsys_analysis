#!/usr/bin/env python3

"""Generate energy spectra for all channels that appear in a petsys group mode file

Usage: channel_specs.py [--mm] [--out OUTFILE] (--map MFILE) INPUT ...

Arguments:
    INPUT  Input file name or list of names

Required:
    --map=MFILE  Name of the module mapping yml file

Options:
    --mm   Output mini-module level energy channel sum spectra?
    --out=OUTFILE  Name base for output image files [default: slabSpec]
"""

from docopt      import docopt
from collections import Counter

import matplotlib.pyplot as plt
import numpy             as np

from pet_code.src.fits import fit_gaussian
from pet_code.src.io   import read_petsys_filebyfile
from pet_code.src.io   import read_ymlmapping
from pet_code.src.util import filter_impact, filter_multihit
from pet_code.src.util import select_module
from pet_code.src.util import shift_to_centres


class ChannelCal:
    def __init__(self, time_ch, eng_ch) -> None:
        self.slab_energies = {}
        self.eng_sum       = {}# Mod eng channel sum with max slab key
        self.eng_max       = {}# Eng channel spectra for max eng channel.
        self.time_id       = time_ch
        self.eng_id        = eng_ch

    def add_evt(self, supermods):
        mms     = {ch[1] for ch in supermods[0]}
        sel_mod = select_module(supermods[0], self.eng_id)
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
        # try:
        #     max_eng = sel_mod[np.argmax(e_chans)]
        #     self.eng_max[max_eng[0]].append(max_eng[3])
        # except KeyError:
        #     self.eng_max[max_eng[0]] = [max_eng[3]]
        for imp in filter(lambda x: x[0] in self.eng_id, sel_mod):
            try:
                self.eng_max[imp[0]].append(imp[3])
            except KeyError:
                self.eng_max[imp[0]] = [imp[3]]
        return len(mms)


def filter_minch(min_ch, eng_ch):
    filt = filter_impact(min_ch, eng_ch)
    def valid_event(sm, _):
        return filt(sm)
    return valid_event


def filter_oneMM():
    def valid_event(sm, _):
        return filter_multihit(sm)
    return valid_event


def average_error(x, y, yerr, ysum):
    bin_err = x / 2# Half bin width as error
    x_cont  = np.sum((y * bin_err)**2) / ysum**2
    y_cont  = np.sum(((x * ysum - y * x) * yerr)**2) / ysum**4
    return np.sqrt(x_cont + y_cont)


if __name__ == '__main__':
    args     = docopt(__doc__)
    mm_spec  = args['--mm']
    map_file = args['--map']
    out_file = args['--out']
    infiles  = args['INPUT']

    time_ch, eng_ch, mm_map, _, _ = read_ymlmapping(map_file)
    filt   = filter_minch(4, eng_ch)
    plotS  = ChannelCal(time_ch, eng_ch)
    plotNS = ChannelCal(time_ch, eng_ch)
    # Maybe a bit dangerous memory wise, review
    for fn in infiles:
        reader = read_petsys_filebyfile(fn, mm_map, sm_filter=filt, singles=True)
        print(f'Reading {fn}')
        if 'wo' in fn:
            num_mmsN = tuple(map(plotNS.add_evt, reader()))
        else:
            num_mmsS = tuple(map(plotS .add_evt, reader()))
    print("First pass complete, mm multiplicities with    Source: ", Counter(num_mmsS))
    print("First pass complete, mm multiplicities without Source: ", Counter(num_mmsN))
    spec_bins = np.arange(5, 30, 0.2)
    with open(out_file + 'timefitVals.txt', 'w') as par_out:
        par_out.write('ID\t MU\t MU_ERR\t SIG\t SIG_ERR\n')
        for id, engs in plotS.slab_energies.items():
            s_vals , bin_e = np.histogram(engs, bins=spec_bins)
            try:
                ns_vals, _ = np.histogram(plotNS.slab_energies[id], bins=spec_bins)
            except KeyError:
                plt.plot(bin_e[:-1], s_vals , label='Source')
                plt.legend()
                plt.xlabel(f'Energy time channel {id}')
                plt.ylabel('au')
                plt.savefig(out_file + f'NoSourceZero_ch{id}.png')
                plt.clf()
                continue
            bin_errs       = np.sqrt(s_vals + ns_vals)
            try:
                bcent, g_vals, fit_pars, cov = fit_gaussian(s_vals - ns_vals, bin_e, yerr=bin_errs, min_peak=300)
            except RuntimeError:
                print(f'Failed fit for channel {id}')
                plt.plot(bin_e[:-1], s_vals , label='Source')
                plt.plot(bin_e[:-1], ns_vals, label='No Source')
                plt.errorbar(bin_e[:-1], s_vals - ns_vals, yerr=bin_errs, label='difference')
                plt.legend()
                plt.xlabel(f'Energy time channel {id}')
                plt.ylabel('au')
                plt.savefig(out_file + f'FailFit_ch{id}.png')
                plt.clf()
                continue
            par_out.write(f'{id}\t {fit_pars[1]}\t {cov[1, 1]}\t {fit_pars[2]}\t {cov[2, 2]}\n')
            plt.errorbar(bcent, s_vals - ns_vals, yerr=bin_errs, label='distribution')
            plt.plot(bcent, g_vals, label=f'Gaussian fit $\mu = {round(fit_pars[1], 2)}, \sigma = {round(fit_pars[2], 2)}$')
            plt.legend()
            plt.xlabel(f'Energy time channel {id}')
            plt.ylabel('source spec - no source spec (au)')
            plt.savefig(out_file + f'BackRest_ch{id}.png')
            plt.clf()
    sm_add = (0, 512)
    esum_bins = np.arange(0, 300, 1.5)
    for sm_chmin in sm_add:
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
        for tid, engs in plotS.eng_sum.items():
            if tid < sm_chmin or tid >= sm_chmin + 256: continue
            mm = mm_map[tid]
            axes[(mm - 1) // 4, (mm - 1) % 4].hist(engs, bins=esum_bins, histtype='step', label=f'tch max id {tid}')
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlabel(f'MM{i+1} Energy sum (au)')
            ax.set_ylabel('Frequency per bin (au)')
            ax.legend()
            fig.savefig(out_file + f'MMEngs_sm{(sm_chmin//256) + 1}.png')
    plt.clf()
    spec_bins = np.arange(7, 30, 0.2)
    with open(out_file + 'engAvDiff.txt', 'w') as par_out:
        par_out.write('ID\t MU\t MU_ERR\n')
        for id, engs in plotS.eng_max.items():
            vals  , edges, _ = plt.hist(engs              , bins=spec_bins, histtype='step', label='Source')
            valsNS, *_       = plt.hist(plotNS.eng_max[id], bins=spec_bins, histtype='step', label='No Source')
            bin_cent  = shift_to_centres(edges)
            hdiff     = vals - valsNS
            hdiff_err = np.sqrt(vals + valsNS)
            plt.errorbar(bin_cent, hdiff, yerr=hdiff_err, label='Difference')
            max_bin   = np.argmax(hdiff)
            mask      = (hdiff > 0.3 * hdiff.max()) & (bin_cent > bin_cent[max_bin] - 3) & (bin_cent < bin_cent[max_bin] + 3)
            av_diff, wsum = np.average(bin_cent[mask], weights=hdiff[mask], returned=True)
            av_err        = average_error(bin_cent[mask], hdiff[mask], hdiff_err[mask], wsum)
            plt.axvspan(av_diff - av_err, av_diff + av_err, facecolor='#00FF00' , alpha = 0.3, label='Average diff')
            par_out.write(f'{id}\t {round(av_diff, 3)}\t {round(av_err, 3)}\n')
            plt.xlabel(f'Energy E channel {id}')
            plt.ylabel('Frequency per bin (au)')
            plt.legend()
            plt.savefig(out_file + f'Emax_ch{id}.png')
            plt.clf()
