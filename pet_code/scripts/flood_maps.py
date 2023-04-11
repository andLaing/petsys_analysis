#!/usr/bin/env python3

"""Calculate and save mini-module level energy spectra and floodmaps

Usage: flood_maps.py (--conf CONFFILE) INFILES ...

Arguments:
    INFILES  File name(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt import docopt
from typing import Callable, Iterator

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.filters import filter_event_by_impacts_noneg
from pet_code.src.filters import filter_max_coin_event
from pet_code.src.filters import filter_impacts_one_minimod
from pet_code.src.fits    import fit_gaussian
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.plots   import plt
from pet_code.src.plots   import mm_energy_spectra
from pet_code.src.plots   import plot_settings
from pet_code.src.util    import np
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import centroid_calculation
from pet_code.src.util    import get_supermodule_eng
from pet_code.src.util    import mm_energy_centroids
from pet_code.src.util    import select_module
from pet_code.src.util    import shift_to_centres


def _get_bin(bin_edges, val):
    if val >= bin_edges[-1]:
        return
    indx = np.searchsorted(bin_edges, val, side='right') - 1
    if indx >= 0:
        return indx


def xyE_binning(chan_map: ChannelMap,
                xbins   : np.ndarray,
                ybins   : np.ndarray,
                ebins   : np.ndarray,
                cal_func: Callable  ,
                sel_func: Callable  ,
                c_calc  : Callable
                ) -> tuple[Callable, Callable]:
    h_shape  = (xbins.size - 1, ybins.size - 1, ebins.size - 1)
    sm_specs = {sm: {mm: np.zeros(h_shape, np.uint) for mm in range(16)}
                for sm in chan_map.mapping.supermodule.unique()     }
    def fill_bin(evt: tuple[list, list]) -> None:
        cal_evt = cal_func(evt)
        for sm_info in filter(lambda x: x, map(sel_func, cal_evt)):
            sm      = chan_map.get_supermodule(sm_info[0][0])
            mm      = chan_map.get_minimodule (sm_info[0][0])
            x, y, _ = c_calc(sm_info)
            _, eng  = get_supermodule_eng(sm_info)
            xb      = _get_bin(xbins,   x)
            yb      = _get_bin(ybins,   y)
            eb      = _get_bin(ebins, eng)
            if all((xb, yb, eb)):
                sm_specs[sm][mm][xb, yb, eb] += 1

    def make_plots(plotter: Callable) -> None:
        # for sm, mms in sm_specs.items():
        #     plotter(sm, mms)
        for sm in list(sm_specs.keys()):
            plotter(sm, sm_specs[sm])
            del sm_specs[sm]
    return fill_bin, make_plots


import time
if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    start    = time.time()
    map_file = conf.get('mapping', 'map_file')
    infiles  = args['INFILES']

    chan_map  = ChannelMap(map_file)
    filt_type = conf.get('filter', 'type', fallback='Impacts')
    # Should improve with an enum or something
    if 'Impacts'  in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_event_by_impacts(min_chan)
    elif 'OneMod' in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_impacts_one_minimod(min_chan, chan_map.get_minimodule)
    elif 'NoNeg'  in filt_type:
        min_chan   = conf.getint('filter', 'min_channels')
        evt_select = filter_event_by_impacts_noneg(min_chan)
    elif 'nSM'    in filt_type:
        ## Impacts, no negatives, only n SM.
        min_chan   = conf.getint('filter', 'min_channels')
        max_sms    = conf.getint('filter', 'max_sm')
        evt_select = filter_max_coin_event(chan_map.get_supermodule, max_sms, min_chan)
    else:
        print('No valid filter found, fallback to 4 minimum energy channels')
        evt_select = filter_event_by_impacts(4)

    time_cal = conf.get('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get('calibration', 'energy_channels', fallback='')
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal)

    c_calc   = centroid_calculation(chan_map.plotp)
    max_sel  = select_module(chan_map.get_minimodule) if conf.getboolean('filter', 'sel_max_mm') else lambda x: x
    out_dir  = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nsigma   = conf.getint('output', 'nsigma', fallback=2)
    sm_setup = 'ebrain' if 'brain' in map_file else 'tbpet'
    mm_ecent = mm_energy_centroids(c_calc, chan_map.get_supermodule,
                                   chan_map.get_minimodule, mod_sel=max_sel)
    try:
        out_form = conf.get('output', 'out_file')
    except configparser.NoOptionError:
        out_form = None

    pet_reader = read_petsys_filebyfile(chan_map.ch_type, evt_select)
    end_sec    = time.time()
    print(f'Time elapsed in setups: {end_sec - start} s')
    start_sec  = end_sec
    for fn in infiles:
        print(f'Reading file {fn}')
        xbins = np.linspace(0, 104, 500)
        ybins = np.linspace(0, 104, 500)
        ebins = np.linspace(0, 300, 200)
        binner, plotter = xyE_binning(chan_map, xbins, ybins, ebins, cal_func, max_sel, c_calc)
        _ = tuple(map(binner, pet_reader(fn)))
        # filtered_events = list(map(cal_func, pet_reader(fn)))
        end_sec         = time.time()
        print(f'Time enlapsed reading: {end_sec - start_sec} s')
        #print("length check: ", len(filtered_events))

        start_sec = end_sec
        # mod_dicts = mm_ecent(filtered_events)

        cal     = 'cal' if time_cal else 'noCal'
        set_end = f'_{cal}_filt{filt_type}.ldat'
        if out_form is None:
            fbase    = fn.split(os.sep)[-1]
            out_base = os.path.join(out_dir, fbase.replace('.ldat', set_end))
        else:
            fbase    = out_form + fn.split(os.sep)[-1].replace('.ldat', set_end)
            out_base = os.path.join(out_dir, fbase)
        # for sm, specs in yielder():
        #     plot_floodmap(sm, specs, xbins, ybins, ebins, min_peak=100, nsigma=nsigma, plot_output=out_base)
        plotter(mm_energy_spectra(sm_setup, out_base, 100, nsigma=nsigma))
        # plotter  = mm_energy_spectra(sm_setup, out_base, 100, nsigma=nsigma)
        # photo_peak = {i: plotter(i, vals) for i, vals in mod_dicts.items()}
        # photo_peak = dict(map(lambda i: i[0]: plotter(*i), mod_dicts.items()))
        end_p = time.time()
        print("Time enlapsed plotting: {} s".format(int(end_p - start_sec)))
