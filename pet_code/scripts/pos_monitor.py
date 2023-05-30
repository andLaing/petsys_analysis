#!/usr/bin/env python3

"""Make position plots under given (conf) calibration conditions and return plots

Usage: pos_monitor.py (--conf CONFFILE) [-j] INPUT ...

Arguments:
    INPUT  File(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.

Options:
    -j  Join data from multiple files instead of treating them as separate.
"""

import os
import configparser

from docopt import docopt
from typing import Callable

import matplotlib.pyplot as plt

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.util    import np
from pet_code.src.util    import pd
from pet_code.src.util    import ChannelType
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import energy_weighted_average
from pet_code.src.util    import get_supermodule_eng
from pet_code.src.util    import select_energy_range
from pet_code.src.util    import select_max_energy
from pet_code.src.util    import select_module
from pet_code.src.util    import shift_to_centres

from pet_code.scripts.cal_monitor import cal_and_sel


def cog_doi(sm_info: list[list]) -> float:
    esum = sum(x[3] for x in filter(lambda y: y[1] is ChannelType.ENERGY, sm_info))
    return esum / max(sm_info, key=lambda x: x[3])[3]


def position_histograms(ybins    : np.ndarray,
                        dbins    : np.ndarray,
                        yrec     : Callable  ,
                        drec     : Callable  ,
                        ch_per_mm: int       ,
                        emin_max : tuple     ,
                        chan_map : ChannelMap
                        ) -> Callable:
    max_slab = select_max_energy(ChannelType.TIME)
    erange   = select_energy_range(*emin_max)
    ebins    = np.arange(0, 300, 10)
    # Channel ordering
    icols    = ['supermodule', 'minimodule', 'local_y']
    isTime   = chan_map.mapping.type.map(lambda x: x is ChannelType.TIME)
    ord_chan = chan_map.mapping[isTime].sort_values(icols).groupby(icols[:-1]).head(ch_per_mm).index
    chan_idx = {id: idx % ch_per_mm for idx, id in enumerate(ord_chan)}
    sm_nums  = chan_map.mapping.supermodule.unique()
    mm_nums  = chan_map.mapping.minimodule .unique()
    def _plotter(evt: tuple[list, list]) -> None:
        for sm_info in evt:
            _plotter.all_count += 1
            _, eng = get_supermodule_eng(sm_info)
            # if eng < ebins[-1] and (bn_idx := np.searchsorted(ebins, eng, side='right') - 1) >= 0:
            #     _plotter.all_spec[bn_idx] += 1
            if erange(eng):
                _plotter.ecount += 1
                try:
                    Y = yrec(sm_info)
                except IndexError:
                    continue
                if (mx_tchn := max_slab(sm_info)):
                    _plotter.sl_count += 1
                    slab_id = mx_tchn[0]
                    Y      -= chan_map.mapping.at[slab_id, 'local_y']
                    if Y < ybins[-1] and (bn_idx := np.searchsorted(ybins, Y, side='right') - 1) >= 0:
                        slab_idx = chan_idx[slab_id]
                        sm_mm    = chan_map.get_modules(slab_id)
                        # _plotter.yspecs[sm_mm][bn_idx, slab_idx] += 1
                        plotter.yspecs[sm_mm][bn_idx] += 1
                        DOI = drec(sm_info)
                        if DOI < dbins[-1] and (bn_idx := np.searchsorted(dbins, DOI, side='right') - 1) >= 0:
                            # _plotter.dspecs[sm_mm][bn_idx, slab_idx] += 1
                            _plotter.dspecs[sm_mm][bn_idx] += 1
    # _plotter.yspecs = {(sm, mm): np.zero((ybins.shape[0] - 1, ch_per_mm), np.uint) for mm in mm_nums for sm in sm_nums}
    # _plotter.dspecs = {(sm, mm): np.zero((dbins.shape[0] - 1, ch_per_mm), np.uint) for mm in mm_nums for sm in sm_nums}
    _plotter.yspecs = {(sm, mm): np.zeros(ybins.shape[0] - 1, np.uint) for mm in mm_nums for sm in sm_nums}
    _plotter.dspecs = {(sm, mm): np.zeros(dbins.shape[0] - 1, np.uint) for mm in mm_nums for sm in sm_nums}
    _plotter.all_count = 0
    _plotter.ecount    = 0
    _plotter.sl_count  = 0
    # _plotter.all_spec  = np.zeros(ebins.shape[0] - 1, np.uint)
    return _plotter


if __name__ == '__main__':
    args   = docopt(__doc__)
    conf   = configparser.ConfigParser()
    conf.read(args['--conf'])

    infiles  = args['INPUT']

    map_file = conf.get('mapping', 'map_file')
    chan_map = ChannelMap(map_file)

    min_chan   = conf.getint('filter', 'min_channels')
    singles    = 'coinc' not in infiles[0]
    evt_filter = filter_event_by_impacts(min_chan, singles=singles)

    time_cal = conf.get     ('calibration',   'time_channels', fallback='')
    eng_cal  = conf.get     ('calibration', 'energy_channels', fallback='')
    eref     = conf.getfloat('calibration', 'energy_reference', fallback=None)
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_cal, eng_cal, eref=eref)

    emin_max = tuple(map(float, conf.get('output', 'elimits', fallback='42,78').split(',')))
    ybins    = np.arange(*map(float, conf.get('output',   'ybinning', fallback='-13,13,0.2').split(',')))
    dbins    = np.arange(*map(float, conf.get('output', 'doibinning', fallback='0,21,0.2').split(',')))
    out_dir  = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    reader  = read_petsys_filebyfile(chan_map.ch_type, sm_filter=evt_filter, singles=singles)
    cal_sel = cal_and_sel(cal_func, select_module(chan_map.get_minimodule))

    rec_type = conf.get('output', 'reco_type', fallback='cog')
    yrec = energy_weighted_average(chan_map.get_plot_position, 1, 2)
    drec = cog_doi
    plotter = position_histograms(ybins, dbins, yrec, drec, 8, emin_max, chan_map)
    for fn in infiles:
        print(f'Reading {fn}')
        for evt in map(cal_sel, reader(fn)):
            plotter(evt)

    ydf = pd.DataFrame(shift_to_centres(ybins).round(2), columns=['bin_centres'])
    for (sm, mm), hist in plotter.yspecs.items():
        ydf[f'{sm}_{mm}'] = hist#.sum(axis=1)
    out_file = os.path.join(out_dir, f'yspecs_{rec_type}.tsv')
    ydf.to_csv(out_file, sep='\t', index=False)
    ddf = pd.DataFrame(shift_to_centres(dbins).round(2), columns=['bin_centres'])
    for (sm, mm), hist in plotter.dspecs.items():
        ddf[f'{sm}_{mm}'] = hist#.sum(axis=1)
    out_file = os.path.join(out_dir, f'DOIspecs_{rec_type}.tsv')
    ddf.to_csv(out_file, sep='\t', index=False)

    print(f'All {plotter.all_count}, eng {plotter.ecount}, slabs {plotter.sl_count}')
    # plt.plot(shift_to_centres(np.arange(0, 300, 10)), plotter.all_spec)
    # plt.show()
