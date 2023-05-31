#!/usr/bin/env python3

"""Calculate and save mini-module level energy spectra and floodmaps
for each super module in the given setup using NN.
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
from pet_code.src.slab_nn import SlabNN
from pet_code.src.slab_nn import SlabSystem
from pet_code.src.slab_nn import neural_net_pcalc
from pet_code.src.plots   import sm_floodmaps
from pet_code.src.util    import ChannelType
from pet_code.src.util    import np
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import get_supermodule_eng
from pet_code.src.util    import select_max_energy
from pet_code.src.util    import select_module


def _get_bin(bin_edges, val):
    if val >= bin_edges[-1]:
        return
    indx = np.searchsorted(bin_edges, val, side='right') - 1
    if indx >= 0:
        return indx


def bunch_predictions(system    : str      ,
                      bunch_size: int      ,
                      y_file    : str      ,
                      doi_file  : str      ,
                      mm_indx   : Callable ,
                      local_pos : Callable ,
                      loc_trans : bool=True
                      ) -> tuple[Callable, Callable]:
    slab_max         = select_max_energy(ChannelType.TIME)
    positions        = neural_net_pcalc(system, y_file, doi_file,
                                        local_pos, loc_trans=loc_trans)
    infer_type       = np.dtype([("slab_idx", np.int32), ("Esignals", np.float32, 8)])
    channel_energies = np.zeros(bunch_size, infer_type)
    def _bunch(sm_info: tuple[list, list], count: int) -> None:
        slab_id = slab_max(sm_info)[0]
        channel_energies[count]['slab_idx'] = slab_id
        for imp in filter(lambda x: x[1] is ChannelType.ENERGY, sm_info):
            channel_energies[count]['Esignals'][mm_indx(imp[0])] = imp[3]

    def _predict() -> tuple[np.ndarray, np.ndarray]:
        print('About to predict')
        xy_pos, doi = positions(channel_energies['slab_idx'], channel_energies)
        channel_energies.fill(0)
        return xy_pos, doi
    return _bunch, _predict

# def neural_net_pcalc(y_file: str, doi_file: str, mm_indx: Callable, local_pos: Callable) -> Callable:
#     """
#     """
#     NN = SlabNN(SlabSystem("IMAS-1ring"))
#     NN.build_combined(y_file, doi_file, print_model=False)
#     ## Dummies for event by event for now.
#     categories = np.zeros(1, np.int32)
#     slab_max = select_max_energy(ChannelType.TIME)
#     def _predict(sm_info: list[list]) -> tuple[float, float]:
#         energies = np.zeros((1, 8), np.float32)
#         for imp in filter(lambda x: x[1] is ChannelType.ENERGY, sm_info):
#             energies[0][mm_indx(imp[0])] = imp[3]
#         mm_y, doi = NN.predict([categories, energies,categories, energies])
#         # print(f'DOI is {doi}, types {type(mm_y)}')
#         max_slab = slab_max(sm_info)[0]
#         local_x, slab_y = local_pos(max_slab)
#         local_y         = mm_y[0] + slab_y
#         return local_x, local_y, doi[0]
#     return _predict


def xyE_binning(chan_map  : ChannelMap,
                xbins     : np.ndarray,
                ybins     : np.ndarray,
                ebins     : np.ndarray,
                cal_func  : Callable  ,
                sel_func  : Callable  ,
                bunch_func: Callable  ,
                bunch_size: int       ,
                pred_func : Callable
                ) -> tuple[Callable, Callable]:
    h_shape  = (xbins.size - 1, ybins.size - 1, ebins.size - 1)
    def fill_bin(evt: tuple[list, list]) -> None:
        cal_evt = cal_func(evt)
        for sm_info in filter(lambda x: x, map(sel_func, cal_evt)):
            sm     = chan_map.get_supermodule(sm_info[0][0])
            mm     = chan_map.get_minimodule (sm_info[0][0])
            _, eng = get_supermodule_eng(sm_info)
            fill_bin.sm_mm_e[fill_bin.count][0] = sm
            fill_bin.sm_mm_e[fill_bin.count][1] = mm
            fill_bin.sm_mm_e[fill_bin.count][2] = eng
            try:
                bunch_func(sm_info, fill_bin.count)
            except TypeError:
                continue
            fill_bin.count += 1
            if fill_bin.count >= bunch_size:
                xy_pos, _ = pred_func()
                for xy, mod_vals in zip(xy_pos, fill_bin.sm_mm_e):
                    xb = _get_bin(xbins,       xy[0])
                    yb = _get_bin(ybins,       xy[1])
                    eb = _get_bin(ebins, mod_vals[2])
                    if all((xb, yb, eb)):
                        fill_bin.sm_specs[int(mod_vals[0])][int(mod_vals[1])][xb, yb, eb] += 1
                fill_bin.count = 0
                fill_bin.sm_mm_e.fill(0)
    fill_bin.count = 0
    fill_bin.sm_specs = {sm: {mm: np.zeros(h_shape, np.uint) for mm in range(16)}
                         for sm in chan_map.mapping.supermodule.unique()     }
    fill_bin.sm_mm_e  = np.empty((bunch_size, 3), np.float32)
            # try:
            #     x, y, _ = c_calc(sm_info)
            # except TypeError:
            #     continue
            # xb      = _get_bin(xbins,   x)
            # yb      = _get_bin(ybins,   y)
            # eb      = _get_bin(ebins, eng)
            # if all((xb, yb, eb)):
            #     sm_specs[sm][mm][xb, yb, eb] += 1

    def make_plots(plotter: Callable) -> None:
        for sm in list(fill_bin.sm_specs.keys()):
            plotter(sm, fill_bin.sm_specs[sm])
            del fill_bin.sm_specs[sm]
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

    # c_calc   = centroid_calculation(chan_map.plotp)
    nn_yfile   = conf.get   ('network',      'y_file')
    nn_dfile   = conf.get   ('network',    'doi_file')
    batch_size = conf.getint('network',  'batch_size', fallback=1000)
    system_nm  = conf.get   ('network', 'system_name', fallback='IMAS-1ring')
    # c_calc   = neural_net_pcalc(nn_yfile, nn_dfile, chan_map.get_minimodule_index, chan_map.get_plot_position)
    b_func, pr_func = bunch_predictions(system_nm, batch_size, nn_yfile, nn_dfile,
                                        chan_map.get_minimodule_index, chan_map.get_plot_position)
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
        binner, plotter = xyE_binning(chan_map, xbins, ybins, ebins, cal_func, max_sel, b_func, batch_size, pr_func)
        _               = tuple(map(binner, islice(pet_reader(fn), max_evt)))
        if binner.count != 0:
            print('doing a last prediction')
            xy_pos, _ = pr_func()
            for i in range(binner.count):
            # for xy, mod_vals in zip(xy_pos, binner.sm_mm_e):
                xb = _get_bin(xbins,         xy_pos[i][0])
                yb = _get_bin(ybins,         xy_pos[i][1])
                eb = _get_bin(ebins, binner.sm_mm_e[i][2])
                if all((xb, yb, eb)):
                    binner.sm_specs[int(binner.sm_mm_e[i][0])][int(binner.sm_mm_e[i][1])][xb, yb, eb] += 1
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
