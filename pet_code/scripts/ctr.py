#!/usr/bin/env python3

"""Monitor time resolution with or without energy calibration

Usage: ctr.py (--map MAPFILE) [--tcal TCAL] [--ecal ECAL] [--eref EREF] [--elim ELIM] [--sk SKEW] INPUT

Arguments:
    INPUT  File to be processed.

Required:
    --map=MAPFILE  Configuration file for run.

Options:
    --tcal=TCAL  File for time channel equalization [default: '']
    --ecal=ECAL  File for the energy channel equalization [default: '']
    --elim=ELIM  Lower and upper limits for slab energy [default: 420,600]
    --sk=SKEW    Name of the file containing skew values per channel.
    --eref=EREF  Reference value for energy equalization, defaults to mean value of peaks [default: 0]
"""

import os
from docopt import docopt

import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.fits    import fit_gaussian
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.plots   import ctr
from pet_code.src.plots   import slab_energy_spectra
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import centroid_calculation
from pet_code.src.util    import get_absolute_id
from pet_code.src.util    import select_energy_range
from pet_code.src.util    import slab_energy_centroids


def read_skewfile(file_name):
    """
    Make utility?
    """
    elec_cols       = ['#portID', 'slaveID', 'chipID', 'channelID']
    skew_vals       = pd.read_csv(skew_file, sep='\t')
    skew_vals['id'] = skew_vals.apply(lambda r: get_absolute_id(*r[elec_cols]),
                                      axis=1).astype('int')
    return skew_vals.set_index('id')['tOffset (ps)']


if __name__ == '__main__':
    args       = docopt(__doc__)
    file_name  =       args['INPUT' ]
    map_file   =       args['--map' ]
    tcal       =       args['--tcal']
    ecal       =       args['--ecal']
    skew_file  =       args['--sk'  ]
    eng_limits =       args['--elim']
    eref_val   = float(args['--eref'])

    chan_map = ChannelMap(map_file)

    evt_filter    = filter_event_by_impacts(4)
    reader        = read_petsys_filebyfile(chan_map.ch_type, sm_filter=evt_filter)
    cal_func      = calibrate_energies(chan_map.get_chantype_ids, tcal, ecal)
    filtered_evts = list(map(cal_func, reader(file_name)))

    skew       = read_skewfile(skew_file)
    sel_energy = select_energy_range(*map(float, eng_limits.split(',')))
    ctr_raw    = ctr(sel_energy)
    ctr_skew   = ctr(sel_energy, skew)
    dts_raw    = [dt for evt in filtered_evts if (dt := ctr_raw (evt))]
    dts_skew   = [dt for evt in filtered_evts if (dt := ctr_skew(evt))]

    hist_bins = np.linspace(-10000, 10000, 800, endpoint=False)
    plt.hist(dts_raw, bins=hist_bins, histtype='step', label='Raw times')
    bin_vals, bin_edges, _ = plt.hist(dts_skew, bins=hist_bins, histtype='step', label='skew corrected')
    bcent, gvals, pars, _  = fit_gaussian(bin_vals, bin_edges)
    plt.plot(bcent, gvals, label=f'fit centroid = {round(pars[1], 3)}, sigma = {round(pars[2], 3)}')
    plt.legend()
    plt.xlabel('timestamp difference (ps)')
    plt.savefig(file_name.split(os.sep)[-1].replace('.ldat', '_skewDT.png'))
    plt.show()


    # c_calc     = centroid_calculation(centroid_map)
    # evt_engs   = slab_energy_centroids(filtered_evts, c_calc, time_ch)
    # peak_funcs = [slab_energy_spectra(evtE, None, 100, np.arange(420, 600, 2)) for evtE in evt_engs]

    # ctr_func   = ctr(time_ch, peak_funcs)
    # dts        = list(filter(lambda x: x, map(ctr_func, filtered_evts)))

    # hist_bins  = np.linspace(-5000, 5000, 400, endpoint=False)
    # plt.hist(dts, bins=hist_bins)
    # plt.show()
