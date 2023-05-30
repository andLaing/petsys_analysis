#!/usr/bin/env python3

"""Calculate and save mini-module level energy spectra and floodmaps
for each super module in the given setup.
Config file should contain at least he name of the map
file as [mapping] map_file = name.

Usage: flood_maps.py (--conf CONFFILE) INFILES ...

Arguments:
    INFILES  File name(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt    import docopt
from functools import partial
from itertools import islice
from typing    import Callable

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.filters import filter_event_by_impacts_noneg
from pet_code.src.filters import filter_max_coin_event
from pet_code.src.filters import filter_impacts_one_minimod
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.plots   import sm_floodmaps
from pet_code.src.util    import np
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import centroid_calculation
from pet_code.src.util    import get_supermodule_eng
from pet_code.src.util    import select_module


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

    time_cal = conf.get     ('calibration',   'time_channels' , fallback='')
    eng_cal  = conf.get     ('calibration', 'energy_channels' , fallback='')
    eref     = conf.getfloat('calibration', 'energy_reference', fallback=None)
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal, eref=eref)

    c_calc   = centroid_calculation(chan_map.plotp)
    max_sel  = select_module(chan_map.get_minimodule) if conf.getboolean('filter', 'sel_max_mm') else lambda x: x
    out_dir  = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nsigma   = conf.getint('output', 'nsigma', fallback=2)
    sm_setup = 'ebrain' if 'brain' in map_file else 'tbpet'
    try:
        out_form = conf.get('output', 'out_file')
    except configparser.NoOptionError:
        out_form = None

    pet_reader = read_petsys_filebyfile(chan_map.ch_type, evt_select)
    end_sec    = time.time()

    max_evt   = conf.getint('output', 'max_events', fallback=None)
    xbins     = np.linspace(*map(int, conf.get('output', 'xbinning', fallback='0,104,500').split(',')))
    ybins     = np.linspace(*map(int, conf.get('output', 'ybinning', fallback='0,104,500').split(',')))
    ebins     = np.linspace(*map(int, conf.get('output', 'ebinning', fallback='0,300,200').split(',')))
    min_stats = conf.getint    ('output', 'min_stats', fallback=100)
    log_plot  = conf.getboolean('output',  'log_plot', fallback=False)
    cmap      = conf.get       ('output',  'colormap', fallback='Reds')
    fl_plot   = partial(sm_floodmaps        ,
                        setup    = sm_setup ,
                        min_peak = min_stats,
                        nsigma   = nsigma   ,
                        xbins    = xbins    ,
                        ybins    = ybins    ,
                        ebins    = ebins    ,
                        log      = log_plot ,
                        cmap     = cmap     )
    print(f'Time elapsed in setups: {end_sec - start} s')
    start_sec  = end_sec
    for fn in infiles:
        print(f'Reading file {fn}')
        binner, plotter = xyE_binning(chan_map, xbins, ybins, ebins, cal_func, max_sel, c_calc)
        _               = tuple(map(binner, islice(pet_reader(fn), max_evt)))
        end_sec         = time.time()
        print(f'Time enlapsed reading: {end_sec - start_sec} s')

        start_sec = end_sec

        cal     = 'cal' if time_cal else 'noCal'
        set_end = f'_{cal}_filt{filt_type}.ldat'
        if out_form is None:
            fbase    = fn.split(os.sep)[-1]
            out_base = os.path.join(out_dir, fbase.replace('.ldat', set_end))
        else:
            fbase    = out_form + fn.split(os.sep)[-1].replace('.ldat', set_end)
            out_base = os.path.join(out_dir, fbase)

        plotter(fl_plot(out_base=out_base))
        end_p = time.time()
        print("Time enlapsed plotting: {} s".format(int(end_p - start_sec)))
