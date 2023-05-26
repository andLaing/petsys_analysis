#!/usr/bin/env python3

"""
Reads and selects events from PETsys binary,
applies all corrections to energies, and performs
fits to resulting distribution.

usage: res_plot.py (--conf CONFFILE) INPUT ...

Arguments:
    INPUT  File(s) to be analysed.

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt import docopt
from typing import Callable

from pet_code.src.filters import filter_max_coin_event
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.util    import np
from pet_code.src.util    import ChannelType
from pet_code.src.util    import convert_to_kev

from pet_code.scripts.make_listmode import equal_and_select


def esum_plot(bin_edges: np.ndarray, kev_conv: Callable):
    def _bin_value(evt):
        for sm in evt:
            eng    = kev_conv(sm[0][0]) * sum(imp[3] for imp in filter(lambda x: x[1] is ChannelType.ENERGY, sm))
            if eng < bin_edges[-1] and (bn_idx := np.searchsorted(bin_edges, eng, side='right') - 1) >= 0:
                _bin_value.spec[bn_idx] += 1
    _bin_value.spec = np.zeros(bin_edges.shape[0] - 1, np.uint)
    return _bin_value



if __name__ == '__main__':
    args = docopt(__doc__)
    conf = configparser.ConfigParser()
    conf.read(args['--conf'])

    map_file = conf.get('mapping', 'map_file')
    infiles  = args['INPUT']

    chan_map = ChannelMap(map_file)

    min_chan  = conf.getint('filter', 'min_channels')
    max_super = conf.getint('filter', 'max_sm'      )
    evt_filt  = filter_max_coin_event(chan_map.get_supermodule, max_super, min_chan)
    evt_read  = read_petsys_filebyfile(chan_map.ch_type, evt_filt)

    tcal     = conf.get     ('calibration',   'time_channels' )
    ecal     = conf.get     ('calibration', 'energy_channels' )
    eref     = conf.getfloat('calibration', 'energy_reference', fallback=None)
    sel_func = equal_and_select(chan_map, tcal, ecal)
    mm_eng   = convert_to_kev(conf.get('calibration', 'kev_convert'), chan_map.get_modules)

    esum_bins = np.arange(*map(float, conf.get('output', 'esum_binning', fallback='0,300,1.5').split(',')))
    plotter   = esum_plot(esum_bins, mm_eng)

    out_dir  = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print('Starting event loop')
    for fn in infiles:
        for evt in map(sel_func, evt_read(fn)):
            plotter(evt)

    # cutre a txt por ahora:
    print('Starting text output')
    with open(os.path.join(out_dir, 'esum_dist.txt'), 'w') as dout:
        dout.write('BinLow\tBinVal\n')
        for bn, vl in zip(esum_bins[:-1], plotter.spec):
            dout.write(f'{round(bn, 2)}\t{round(vl, 3)}\n')
