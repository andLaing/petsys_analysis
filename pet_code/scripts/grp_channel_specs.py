#!/usr/bin/env python3

"""Reads a group of files with and without (names should contain 'woSource')
sources present, generates energy histograms for each channel when it is the
maximum of its type in an event (group), takes the difference of the spectra
with and without source and fits the most prominant peak. The peak values
are then saved to disc as a txt file that can be used to relatively equalize
between channels.


Usage: grp_channel_specs.py [--out OUTFILE] [--petsys PSFILE] (--conf CFILE) INPUT ...

Arguments:
    INPUT  Input file name or list of names

Required:
    --conf=CFILE  Name of the module mapping yml file

Options:
    --out=OUTFILE  Name base for output image files.
    --petsys=PSFILE  Name of the PETsys format output, if not provided not output.
"""

import os
import configparser

from docopt import docopt
from typing import Callable

import matplotlib.pyplot as plt
import numpy             as np

from scipy.signal import find_peaks

from pet_code.src.filters  import filter_event_by_impacts
from pet_code.src.fits     import curve_fit_fn
from pet_code.src.fits     import fit_gaussian
from pet_code.src.fits     import gaussian
from pet_code.src.fits     import lorentzian
from pet_code.src.io       import ChannelMap
from pet_code.src.io       import read_petsys_filebyfile
from pet_code.src.plots    import ChannelEHistograms
from pet_code.src.util     import pd
from pet_code.src.util     import ChannelType
from pet_code.src.util     import get_electronics_nums
from pet_code.src.util     import shift_to_centres


def slab_plots(out_file     : str               ,
               plot_source  : ChannelEHistograms,
               plot_wosource: ChannelEHistograms,
               min_stats    : int               ,
               pk_finder    : str       = 'max' ,
               monitor_ids  : dict[int] = {}
               ) -> int:
    bin_edges  = plot_source.edges[ChannelType.TIME]
    check_fits = pd.DataFrame(shift_to_centres(bin_edges), columns=['bin_centres'])
    with open(out_file + 'timeSlabPeaks.txt', 'w') as par_out:
        par_out.write('ID\tMU\tMU_ERR\tSIG\tSIG_ERR\n')
        for id, s_vals in plot_source.tdist.items():
            min_indx = 0
            try:
                ns_vals = plot_wosource.tdist[id]
            except KeyError:
                ns_vals = np.zeros_like(s_vals)
            bin_errs  = np.sqrt(s_vals + ns_vals)
            diff_data = s_vals - ns_vals
            try:
                (bcent   , g_vals,
                 fit_pars, cov   , chi_ndf) = fit_gaussian(diff_data            ,
                                                           bin_edges            ,
                                                           yerr      = bin_errs ,
                                                           min_peak  = min_stats,
                                                           pk_finder = pk_finder)
                if (fit_pars[1] <= bin_edges[ 5] or
                    fit_pars[1] >= bin_edges[-5] or
                    chi_ndf > 5):
                    raise RuntimeError
            except RuntimeError:
                check_fits[f'ch{id}'    ] = diff_data
                check_fits[f'ch{id}_err'] = bin_errs
                continue
            mu_err  = np.sqrt(cov[1, 1])
            sig_err = np.sqrt(cov[2, 2])
            par_out.write(f'{id}\t{round(fit_pars[1], 3)}\t{round(mu_err, 3)}\t{round(fit_pars[2], 3)}\t{round(sig_err, 3)}\n')
            if id in monitor_ids:
                plt.errorbar(bcent, diff_data[min_indx:], yerr=bin_errs[min_indx:], label='distribution')
                plt.plot(bcent, g_vals, label=f'Gaussian fit mu = {round(fit_pars[1], 2)}, sigma = {round(fit_pars[2], 2)}')
                plt.legend()
                plt.xlabel(f'Energy time channel {id}')
                plt.ylabel('source spec - no source spec (au)')
                plt.savefig(out_file + f'BackRest_ch{id}.png')
                plt.clf()
    print(f'{check_fits.shape[1] - 1} time channels with suspect distributions.')
    if check_fits.shape[0] > 1:
        check_fits.to_feather(out_file + 'suspectTimeFits.feather')
    return check_fits.shape[1] - 1


def refit_slab(out_file : str               ,
               id       : int               ,
               bin_edges: np.ndarray        ,
               source   : ChannelEHistograms,
               nosource : ChannelEHistograms,
               diff_data: np.ndarray        ,
               yerr     : np.ndarray
               ) -> tuple:
    print(f'Failed or unsafe fit for channel {id}')
    plt.errorbar(bin_edges[:-1], diff_data, yerr=yerr)
    plt.show()
    min_x = input('Do you want to try a refit? [n]/start x of plot ')
    if min_x:
        bin_wid   = np.diff(bin_edges[:2])[0]
        indx      = int(float(min_x) / bin_wid - bin_edges[0])
        diff_data = source[indx:] - nosource[indx:]
        bin_errs  = np.sqrt(source[indx:] + nosource[indx:])
        try:
            bcent, g_vals, fit_pars, cov, _ = fit_gaussian(diff_data, bin_edges[indx:], yerr=bin_errs, min_peak=int(min_peak * 0.7))
        except RuntimeError:
            fail_plot(out_file, id, bin_edges, source, nosource, np.sqrt(source + nosource))
            return None
    else:
        fail_plot(out_file, id, bin_edges, source, nosource, yerr)
        return None
    return bcent, indx, g_vals, fit_pars, cov


def refit_channel(bin_centres: np.ndarray,
                  data       : np.ndarray,
                  errors     : np.ndarray,
                  fit_func   : Callable  ,
                  nbin_fit   : int
                  ) -> tuple[float, float, float, float]:
    plt.errorbar(bin_centres, data, yerr=errors)
    plt.title('Decide approx position for mean and close plot.')
    plt.show()
    bin_wid    = np.diff(bin_centres)[0]
    edges      = bin_centres - bin_wid / 2
    mean_seed  = input('Please indicate a seed position for the centroid ')
    p_indx     = np.searchsorted(edges, mean_seed, side='right') - 1
    mask       = ((bin_centres > bin_centres[p_indx] - nbin_fit * bin_wid) &
                  (bin_centres < bin_centres[p_indx] + nbin_fit * bin_wid))
    p0         = [data[p_indx], mean_seed, bin_wid * nbin_fit / np.sqrt(12)]
    pars, pcov = curve_fit_fn(fit_func, bin_centres[mask], data[mask], errors[mask], p0)
    return pars[1], np.sqrt(pcov[1, 1]), pars[2], np.sqrt(pcov[2, 2])


def energy_plots(out_file     : str               ,
                 plot_source  : ChannelEHistograms,
                 plot_wosource: ChannelEHistograms,
                 min_peak     : int               ,
                 monitor_ids  : dict[int]={}
                 ) -> int:
    bin_edges  = plot_source.edges[ChannelType.ENERGY]
    bin_cent   = shift_to_centres(bin_edges)
    bin_wid    = np.diff(bin_edges[:2])[0]
    check_fits = pd.DataFrame(bin_cent, columns=['bin_centres'])
    nbin_fit   = 5
    with open(out_file + 'eChannelPeaks.txt', 'w') as par_out:
        par_out.write('ID\tMU\tMU_ERR\n')
        for id, vals in plot_source.edist.items():
            try:
                valsNS = plot_wosource.edist[id]
            except KeyError:
                valsNS = np.zeros_like(vals)

            hdiff     = vals - valsNS
            hdiff_err = np.sqrt(vals + valsNS)
            peaks, _  = find_peaks(hdiff, height=min_peak, distance=5)
            if peaks.shape[0] > 1:
                peak   = np.argmax(hdiff[peaks])
                p_indx = peaks[peak]
            elif peaks.shape[0] == 0:
                p_indx = np.argmax(hdiff)
            else:
                p_indx = peaks[0]
            ## Try to protect against noise floor
            if p_indx <= 2 * nbin_fit:
                print(f'Peak near min for channel {id}')
                check_fits[f'ch{id}'    ] = hdiff
                check_fits[f'ch{id}_err'] = hdiff_err
                continue
            mask = (bin_cent > bin_cent[p_indx] - nbin_fit * bin_wid) & (bin_cent < bin_cent[p_indx] + nbin_fit * bin_wid)
            try:
                p0            = [hdiff[p_indx], bin_cent[p_indx], 3]
                fit_pars, cov = curve_fit_fn(lorentzian, bin_cent[mask], hdiff[mask], np.sqrt(hdiff[mask]), p0)
                av_diff = fit_pars[1]
                av_err  = np.sqrt(cov[1, 1])
                if av_err > 1.0:
                    av_diff, av_err = weighted_average(plt, bin_cent, hdiff, hdiff_err, mask)
            except RuntimeError:
                av_diff, av_err = weighted_average(plt, bin_cent, hdiff, hdiff_err, mask)

            par_out.write(f'{id}\t{round(av_diff, 3)}\t{round(av_err, 3)}\n')
            if id in monitor_ids:
                plt.errorbar(bin_cent, hdiff, yerr=hdiff_err, label='Difference')
                plt.plot(bin_cent[peaks], hdiff[peaks], 'rv', markersize=15, label="Peak finder")
                plt.plot(bin_cent, lorentzian(bin_cent, *fit_pars), label='Gaussian fit')
                plt.xlabel(f'Energy E channel {id}')
                plt.ylabel('Frequency per bin (au)')
                plt.legend()
                plt.savefig(out_file + f'Emax_ch{id}.png')
                plt.clf()
    print(f'{check_fits.shape[1] - 1} energy channels with suspect distributions.')
    if check_fits.shape[1] > 1:
        check_fits.to_feather(out_file + 'suspectEnergyFits.feather')
    return check_fits.shape[1] - 1


def weighted_average(axis     : plt.Axes  ,
                     bin_cent : np.ndarray,
                     hdiff    : np.ndarray,
                     hdiff_err: np.ndarray,
                     mask     : np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    av_diff, wsum = np.average(bin_cent[mask], weights=hdiff[mask], returned=True)
    av_err        = average_error(bin_cent[mask], hdiff[mask], hdiff_err[mask], wsum)
    axis.axvspan(av_diff - av_err, av_diff + av_err, facecolor='#00FF00' , alpha = 0.3, label='Average diff')
    return av_diff, av_err


def channel_plots(config : configparser.ConfigParser,
                  infiles: list[str]
                  ) -> tuple:
    """
    Channel calibration function.
    """
    map_file = config.get('mapping', 'map_file')
    chan_map = ChannelMap(map_file)

    min_chan  = config.getint('filter', 'min_channels')
    filt      = filter_event_by_impacts(min_chan, singles=True)

    esum_bins = np.arange(*map(float, config.get('output', 'esum_binning', fallback='0,300,1.5').split(',')))
    tbins     = np.arange(*map(float, config.get('output',     'tbinning', fallback='5,30,0.2') .split(',')))
    ebins     = np.arange(*map(float, config.get('output',     'ebinning', fallback='7,40,0.4') .split(',')))

    plotS     = ChannelEHistograms(tbins, ebins, esum_bins)
    splots    = plotS .add_emax_evt
    plotNS    = ChannelEHistograms(tbins, ebins, esum_bins)
    nsplots   = plotNS.add_emax_evt

    reader    = read_petsys_filebyfile(chan_map.ch_type, sm_filter=filt, singles=True)
    for fn in infiles:
        print(f'Reading {fn}')
        if 'woSource' in fn:
            _ = tuple(map(nsplots, reader(fn)))
        else:
            _ = tuple(map(splots , reader(fn)))

    return plotS, plotNS


def average_error(x   : np.ndarray,
                  y   : np.ndarray,
                  yerr: np.ndarray,
                  ysum: np.ndarray
                  ) -> np.ndarray:
    bin_err = x / 2# Half bin width as error
    x_cont  = np.sum((y * bin_err)**2) / ysum**2
    y_cont  = np.sum(((x * ysum - y * x) * yerr)**2) / ysum**4
    return np.sqrt(x_cont + y_cont)


def fail_plot(out_file: str       ,
              id      : int       ,
              bin_e   : np.ndarray,
              s_vals  : np.ndarray,
              ns_vals : np.ndarray,
              bin_errs: np.ndarray
              ) -> None:
    plt.plot(bin_e[:-1], s_vals , label='Source')
    plt.plot(bin_e[:-1], ns_vals, label='No Source')
    plt.errorbar(bin_e[:-1], s_vals - ns_vals, yerr=bin_errs, label='difference')
    plt.legend()
    plt.xlabel(f'Energy time channel {id}')
    plt.ylabel('au')
    plt.savefig(out_file + f'FailFit_ch{id}.png')
    plt.clf()


def review_distributions(ntime, neng, out_base):
    """
    Review and refit flagged distributions.
    """
    if ntime:
        print('Reviewing time channel distributions...')
        dist_df     = pd.read_feather(out_base + 'suspectTimeFits.feather')
        bin_centres = dist_df.bin_centres.values
        with open(out_base + 'timeSlabPeaks.txt', 'a') as tout:
            for col in dist_df.columns[dist_df.columns.str.contains('_err')]:
                print(f'time channel {col[2:-4]}')
                try:
                    mu, mu_err, sig, sig_err = refit_channel(bin_centres, dist_df[col[:-4]], dist_df[col], gaussian, 8)
                    tout.write(f'{col[2:-4]}\t{round(mu, 3)}\t{round(mu_err, 3)}\t{round(sig, 3)}\t{round(sig_err, 3)}\n')
                except RuntimeError:
                    print(f'Fit fail time channel {col[2:-4]}.')
                    continue
    if neng:
        print('Reviewing energy channel distributions...')
        dist_df     = pd.read_feather(out_base + 'suspectEnergyFits.feather')
        bin_centres = dist_df.bin_centres.values
        with open(out_base + 'eChannelPeaks.txt', 'a') as eout:
            for col in dist_df.columns[dist_df.columns.str.contains('_err')]:
                print(f'energy channel {col[2:-4]}')
                try:
                    mu, mu_err, *_ = refit_channel(bin_centres, dist_df[col[:-4]], dist_df[col], lorentzian, 5)
                    eout.write(f'{col[2:-4]}\t{round(mu, 3)}\t{round(mu_err, 3)}\n')
                except RuntimeError:
                    print(f'Fit fail energy channel {col[2:-4]}.')
                    continue


def petsys_file(map_file: str  ,
                tchans  : str  ,
                echans  : str  ,
                eref    : float,
                out_name: str
                ) -> None:
    """
    create a file with the format expected by PETsys
    for energy correction.
    No linearisation at the moment.
    """
    all_ids               = pd.read_feather(map_file).id.sort_values()
    cal_fact              = pd.concat((pd.read_csv(tchans, sep='\t').set_index('ID').MU.map(lambda x: 511.0 / x),
                                       pd.read_csv(echans, sep='\t').set_index('ID').MU.map(lambda x:  eref / x))).sort_index()
    cal_fact.index.name   = 'id'
    out_cols              = ['#portID', 'slaveID', 'chipID', 'channelID',
                             'tacID', 'p0', 'p1', 'p2', 'p3']
    ch_gain               = pd.DataFrame([[1.0, 1.0, 1.0]]       ,
                                         columns = out_cols[5:-1],
                                         index   = all_ids       ).reset_index()
    ch_gain[out_cols[:4]] = np.row_stack(ch_gain.id.map(np.vectorize(get_electronics_nums)))
    ch_gain = pd.merge(ch_gain                        ,
                       cal_fact.round(3).reset_index(),
                       on  = 'id'                     ,
                       how = 'left'                   ).rename(columns={'MU': out_cols[-1]}).fillna(1.0)

    ch_gain               = ch_gain.loc[ch_gain.index.repeat(4)].reset_index(drop=True)
    ch_gain[out_cols[4 ]] = np.tile(np.arange(4), ch_gain.shape[0] // 4)
    ch_gain[out_cols    ].to_csv(out_name, sep='\t', index=False)


if __name__ == '__main__':
    args     = docopt(__doc__)
    conf     = configparser.ConfigParser()
    conf.read(args['--conf'])
    out_file = args['--out' ]
    infiles  = args['INPUT' ]

    if out_file is None:
        out_dir  = conf.get('output',  'out_dir', fallback='generic_cal')
        out_file = conf.get('output', 'out_file', fallback='slabSpec'   )
        out_file = os.path.join(out_dir, out_file)
    else:
        try:
            out_dir  = os.path.join(*out_file.split(os.sep)[:-1])
        except TypeError:
            out_dir = '.'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    plotS, plotNS = channel_plots(conf, infiles)

    min_peak  = conf.getint('filter', 'min_stats'  , fallback=300  )
    pk_finder = conf.get   ('filter', 'peak_finder', fallback='max')
    try:
        monitor_id = set(map(int, conf.get('output', 'mon_ids').split(',')))
    except (configparser.NoSectionError, configparser.NoOptionError):
        monitor_id = {}

    # Plotting and fitting.
    nslab_bad = slab_plots  (out_file, plotS, plotNS, min_peak,
                             pk_finder=pk_finder, monitor_ids=monitor_id)
    neng_bad  = energy_plots(out_file, plotS, plotNS, min_peak, monitor_ids=monitor_id)

    ## Review suspect distributions?
    if nslab_bad > 0 or neng_bad > 0:
        review = input(f'There are {nslab_bad} suspect time channels and {neng_bad} energy channels, review now? y/[n]')
        if review:
            review_distributions(nslab_bad, neng_bad, out_file)
