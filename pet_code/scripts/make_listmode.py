#!/usr/bin/env python3

"""
Main data processing dataflow.
Takes PETsys binary output,
selects coincidences according to number of channels
and total energy, reconstructs impact positions and
writes to the listmode binary.

Usage: make_listmode.py (--conf CONFFILE) INPUT ...

Arguments:
    INPUT  File(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt import docopt
from typing import Callable, Tuple

import numpy  as np
import pandas as pd

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.filters import filter_event_by_impacts_noneg
from pet_code.src.filters import filter_max_coin_event
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.util    import ChannelType
from pet_code.src.util    import energy_weighted_average
from pet_code.src.util    import get_supermodule_eng
from pet_code.src.util    import select_energy_range
from pet_code.src.util    import select_max_energy
from pet_code.src.util    import select_module


def local_pixel(bins_x: np.ndarray, bins_y: np.ndarray) -> Callable:
    """
    Convert position in pixel number in x and y according to
    bins_x and bins_y.
    """
    def get_pixels(pos_x: float, pos_y: float) -> Tuple[int, int]:
        return (np.searchsorted(bins_x, pos_x, side='right') - 1,
                np.searchsorted(bins_y, pos_y, side='right') - 1)
    return get_pixels


if __name__ == '__main__':
    args = docopt(__doc__)
    conf = configparser.ConfigParser()
    conf.read(args['--conf'])

    map_file = conf.get('mapping', 'map_file')
    infiles  = args['INPUT']

    chan_map  = ChannelMap(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    if 'Impacts'    in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_event_by_impacts(min_chan)
    elif 'NoNeg'    in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_event_by_impacts_noneg(min_chan)
    elif 'MaxCoinc' in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        max_super  = conf.getint('filter', 'max_sm'      )
        evt_select = filter_max_coin_event(chan_map.get_supermodule,
                                           max_super, min_chan)
    else:
        print(f'filter type {filt_type} not recognised.')
        print( 'falling back to no negative, 2 sm, 4 echannels min per SM')
        ## Falback to full filter
        evt_select = filter_max_coin_event(chan_map.get_supermodule)

    evt_reader = read_petsys_filebyfile(chan_map.ch_type, evt_select)
    elimits    = map(float, conf.get('filter', 'elimits').split(','))
    eselect    = select_energy_range(*elimits)
    sel_mod    = select_module(chan_map.get_minimodule)
    # Hardwiring which local coord is e maybe not ideal
    ecog       = energy_weighted_average(chan_map.get_plot_position, 1, 2)

    xlimits    = map(float, conf.get('output', 'pixels_x').split(','))
    ylimits    = map(float, conf.get('output', 'pixels_y').split(','))
    pixel_vals = local_pixel(np.linspace(*xlimits), np.linspace(*ylimits))
    p_cols     = ['p', 's0', 's1']
    pair_name  = conf.get('mapping', 'pairs')
    p_lookup   = pd.read_csv(pair_name     ,
                             sep   = '\t'  ,
                             names = p_cols).set_index(p_cols[1:])

    ## Assume channel equalisation and skew correction done in PETsys,
    ## introduce option once working.
    for fn in infiles:
        # Open output file.
        for evt in evt_reader(fn):
            # Select the minimodules with most energy.
            # Do we want to filter those with too many?
            mm_info = tuple(map(sel_mod, evt))

            # Get the summed energy deposit for each impact.
            mm_energies = tuple(map(lambda imp: get_supermodule_eng(imp)[1], mm_info))
            if all(eselect(mm_eng) for mm_eng in mm_energies):
                # Time channels, just use max for now.
                max_chans  = tuple(map(select_max_energy, mm_info, [ChannelType.TIME] * 2))
                sm_nums    = (chan_map.get_supermodule  (max_chans[0][0])   ,
                              chan_map.get_supermodule  (max_chans[1][0])   )
                local_tpos = (chan_map.get_plot_position(max_chans[0][0])[0],
                              chan_map.get_plot_position(max_chans[1][0])[0])

                ## Here we'd like to use the trained network to get the position
                ## cog for now.
                pixels     = tuple(map(pixel_vals, local_tpos, map(ecog, evt)))
                try:
                    pair               = p_lookup.loc[sm_nums      ].p
                    e1, e2             = mm_energies
                    (x1, y1), (x2, y2) = pixels
                    dt                 = max_chans[0][2] - max_chans[1][2]
                except KeyError:
                    pair               = p_lookup.loc[sm_nums[::-1]].p
                    e1, e2             = mm_energies[::-1]
                    (x1, y1), (x2, y2) = pixels     [::-1]
                    dt                 = max_chans[1][2] - max_chans[0][2]


