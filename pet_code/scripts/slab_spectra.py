#!/usr/bin/env python3

"""Calculate and save time channel level energy spectra

Usage: slab_spectra.py (--conf CONFFILE) INFILE

Arguments:
    INFILE  File name to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt    import docopt
from itertools import chain

import matplotlib.pyplot as plt
import numpy  as np

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.filters import filter_impacts_one_minimod
from pet_code.src.filters import filter_impacts_module_list
from pet_code.src.fits    import fit_gaussian
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.plots   import ChannelEHistograms
from pet_code.src.util    import ChannelType
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import shift_to_centres


def accept_channel(channel_set):
    def _accept(imp):
        return imp[1] is ChannelType.TIME and imp[0] in channel_set
    return _accept


if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    map_file = conf.get('mapping', 'map_file')
    infile   = args['INFILE']

    chan_map  = ChannelMap(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    singles   = 'coinc' not in infile
    # Should improve with an enum or something
    if 'Impacts'    in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_event_by_impacts(min_chan, singles=singles)
        valid_ch   = accept_channel(set(chan_map.mapping.index))
    elif 'OneMod'   in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        singles    = 'coinc' not in infile
        evt_select = filter_impacts_one_minimod(min_chan, chan_map.get_minimodule, singles=singles)
        valid_ch   = accept_channel(set(chan_map.mapping.index))
    elif 'Specific' in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        valid_sms  = tuple(map(int, conf.get('filter', 'sm_nums').split(',')))
        valid_mms  = tuple(map(int, conf.get('filter', 'mm_nums').split(',')))
        chan_set   = set(np.concatenate([chan_map.get_minimodule_channels(*v)
                                         for v in np.vstack(np.stack(np.meshgrid(valid_sms, valid_mms)).T)]))
        valid_ch   = accept_channel(chan_set)
        evt_select = filter_impacts_module_list(chan_map.get_minimodule_channels,
                                                valid_sms, valid_mms, min_chan, singles=singles)
    else:
        print('No valid filter found, fallback to 4 minimum energy channels')
        evt_select = filter_event_by_impacts(4, singles=singles)
        valid_ch   = accept_channel(set(chan_map.mapping.index))

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal)

    pet_reader = read_petsys_filebyfile(chan_map.ch_type, evt_select, singles=singles)
    out_dir    = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    min_peak   = conf.getint('output', 'min_peak_fit', fallback=100)
    bin_edges  = np.arange(*map(float, conf.get('output', 'binning', fallback='9,25,0.2').split(',')))
    out_base   = os.path.join(out_dir, conf.get('output', 'out_file', fallback=infile.split(os.sep)[-1]))
    slab_plots = ChannelEHistograms(bin_edges, np.arange(1), np.arange(1))
    for evt in map(cal_func, pet_reader(infile)):
        for imp in filter(valid_ch, chain(*evt)):
            slab_plots.fill_time_channel(imp)

    bwid = np.diff(bin_edges)[0]
    for id, hist in slab_plots.tdist.items():
        plt.errorbar(shift_to_centres(bin_edges), hist, yerr=np.sqrt(hist), label='energy histogram')
        try:
            bcent, gvals, pars, _, _ = fit_gaussian(hist, bin_edges, min_peak=min_peak)
            plt.plot(bcent, gvals, label=f'Gaussian fit: mu = {round(pars[1], 2)}, sigma = {round(pars[2], 2)}')
        except RuntimeError:
            max_bin = np.argmax(hist)
            minE, maxE = bin_edges[max_bin - 8], bin_edges[max_bin + 8]
            plt.axvspan(minE, maxE, facecolor='#00FF00' , alpha = 0.3, label='Max +- 8 bins')
        plt.xlabel(f'Energy spectrum channel {id}')
        plt.ylabel(f'Entries per {bwid} counts')
        plt.legend()
        plt.savefig(out_base.replace('.ldat', f'_slab{id}Spec.png'))
        plt.clf()
