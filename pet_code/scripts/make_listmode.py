#!/usr/bin/env python3

"""
Main data processing dataflow.
Takes PETsys binary output,
selects coincidences according to number of channels
and total energy, reconstructs impact positions and
writes to the listmode binary.

Usage: python make_listmode.py (--conf CONFFILE) INPUT ...

Arguments:
    INPUT  File(s) to be analysed

Required:
    --conf=CONFFILE  Configuration file for run.
"""

import os
import configparser

from docopt    import docopt
from itertools import islice
from typing    import BinaryIO, Callable, Tuple

import numpy  as np
import pandas as pd

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.filters import filter_event_by_impacts_noneg
from pet_code.src.filters import filter_max_coin_event
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import CoincidenceV3, LMHeader
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.util    import ChannelType
from pet_code.src.util    import calibrate_energies
from pet_code.src.util    import convert_to_kev
from pet_code.src.util    import energy_weighted_average
from pet_code.src.util    import read_skewfile
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


def equal_and_select(chan_map : ChannelMap       ,
                     time_file: str              ,
                     eng_file : str              ,
                     eref     : float | None=None
                     ) -> Callable:
    cal_func = calibrate_energies(chan_map.get_chantype_ids, time_file, eng_file, eref=eref)
    sel_func = select_module(chan_map.get_minimodule)
    def _select(evt: tuple[list, list]) -> tuple[list, list]:
        return tuple(map(sel_func, cal_func(evt)))
    return _select


def _is_eng(imp: list) -> bool:
    return imp[1] is ChannelType.ENERGY


def supermod_energy(kev_map: Callable) -> float:
    def _sm_eng(imp: list) -> float:
        imp_it = filter(_is_eng, imp)
        return kev_map(imp[0][0]) * sum(hit[3] for hit in imp_it)
    return _sm_eng


def write_header(bin_out: BinaryIO                 ,
                 cf     : configparser.ConfigParser,
                 xpixels: np.ndarray               ,
                 ypixels: np.ndarray
                 ) -> None:
    """
    Write the LMHeader object to file.
    """
    sec = 'header'
    # Only save a minimum for now and assume set in config where necessary
    header = LMHeader(
        identifier         = cf.get     (sec,      'identifier', fallback='IMAS').encode('utf-8'),
        acqTime            = cf.getfloat(sec,         'acqTime', fallback=60    ),
        isotope            = cf.get     (sec,         'isotope', fallback='FDG' ).encode('utf-8'),
        detectorSizeX      = cf.getfloat(sec,   'detectorSizeX')                 ,
        detectorSizeY      = cf.getfloat(sec,   'detectorSizeY')                 ,
        startTime          = cf.getfloat(sec,       'startTime', fallback= 0    ),
        measurementTime    = cf.getfloat(sec, 'measurementTime', fallback=10    ),
        moduleNumber       = cf.getint  (sec,    'moduleNumber')                 ,
        ringNumber         = cf.getint  (sec,      'ringNumber')                 ,
        ringDistance       = cf.getfloat(sec,    'ringDistance')                 ,
        detectorPixelSizeX = np.diff(xpixels)[0]                                 ,
        detectorPixelSizeY = np.diff(ypixels)[0]                                 ,
        version            = (9, 9)                                              ,
        detectorPixelsX    = xpixels.size - 1                                    ,
        detectorPixelsY    = ypixels.size - 1
        )
    bin_out.write(header)


def cog_loop(chan_map  : ChannelMap,
             evt_reader: Callable  ,
             sel_func  : Callable  ,
             eselect   : Callable  ,
             pixel_vals: Callable  ,
             mm_eng    : Callable  ,
             skew      : dict      ,
             p_lookup  : dict
             ) -> Callable:
    """
    Event by event loop from PETsys to LM.
    """
    ecog     = energy_weighted_average(chan_map.get_plot_position, 1, 2)
    max_slab = select_max_energy(ChannelType.TIME)
    def _evt_loop(file_name: str, lm_out: BinaryIO) -> None:
        coinc        = CoincidenceV3()
        coinc.amount = 1.0
        for mm_info in map(sel_func, evt_reader(file_name)):
            # Get the summed energy deposit for each impact.
            mm_energies = tuple(map(mm_eng, mm_info))
            if (all(eselect(mmE) for mmE in mm_energies) and
                all(max_chans := tuple(map(max_slab, mm_info)))):

                sm_nums    = (chan_map.get_supermodule  (max_chans[0][0])   ,
                              chan_map.get_supermodule  (max_chans[1][0])   )
                local_tpos = (chan_map.get_plot_position(max_chans[0][0])[0],
                              chan_map.get_plot_position(max_chans[1][0])[0])

                ## Here we'd like to use the trained network to get the position
                ## cog for now.
                pixels    = tuple(map(pixel_vals, local_tpos, map(ecog, mm_info)))
                skew_corr = skew.get(max_chans[1][0], 0) - skew.get(max_chans[0][0], 0)
                try:
                    pair               = p_lookup[sm_nums]
                    e1, e2             = mm_energies
                    (x1, y1), (x2, y2) = pixels
                    dt                 = max_chans[0][2] - max_chans[1][2] + skew_corr
                except KeyError:
                    try:
                        pair               = p_lookup[sm_nums[::-1]]
                        e1, e2             = mm_energies[::-1]
                        (x1, y1), (x2, y2) = pixels     [::-1]
                        dt                 = max_chans[1][2] - max_chans[0][2] - skew_corr
                    except KeyError:
                        continue
                # Coincidence output
                coinc.pair       =     pair
                coinc.energy1    = round(e1)
                coinc.energy2    = round(e2)
                coinc.xPosition1 =       x1
                coinc.yPosition1 =       y1
                coinc.xPosition2 =       x2
                coinc.yPosition2 =       y2
                coinc.time       =       dt
                lm_out.write(coinc)
                #
    return _evt_loop


def nn_loop(chan_map  : ChannelMap,
            evt_reader: Callable  ,
            sel_func  : Callable  ,
            eselect   : Callable  ,
            pixel_vals: Callable  ,
            kev_conv  : Callable  ,
            skew      : dict      ,
            p_lookup  : dict      ,
            npred     : int
            ) -> None:
    """
    Loop over events optimising calls to neural network.
    npred : Number of predictions to send each time.
    """
    # Should this be configurable?
    nchan            = 8
    ## Will need two entries per coincidence, change structure? Changing it to contain both would simplify calls
    infer_type       = np.dtype([("slab_idx", np.int32), ("Esignals", np.float32, nchan)])
    channel_energies = np.zeros(npred     , infer_type   )
    coincidences     = np.zeros(npred // 2, CoincidenceV3)
    # There's probably a better way to do this,
    for c in coincidences:
        c['amount'] = 1.0
    # Channel ordering
    icols    = ['supermodule', 'minimodule', 'local_y']
    isEng    = chan_map.mapping.type.map(_is_eng)
    ord_chan = chan_map.mapping[isEng].sort_values(icols).groupby(icols[:-1]).head(nchan).index
    chan_idx = {id: idx % nchan for idx, id in enumerate(ord_chan)}
    max_slab = select_max_energy(ChannelType.TIME)
    # Define predictor?
    def _evt_loop(file_name: str, lm_out: BinaryIO):
        read_evt = evt_reader(file_name)
        i = 1
        while i != 0:
            i = 0
            for evt in map(sel_func, islice(read_evt, npred // 2)):
                chids = []
                for j, sm in enumerate(evt):
                    chids.append(sm[0][0])
                    for imp in filter(_is_eng, sm):
                        channel_energies[2 * i + j]['Esignals'][chan_idx[imp[0]]] = imp[3]

                mm_energies = [kev_conv(chid) * mm['Esignals'].sum()
                               for chid, mm in zip(chids, channel_energies[2 * i:2 * i + 1])]
                if (all(eselect(mmE) for mmE in mm_energies) and
                    all(max_chans := tuple(map(max_slab, evt)))):

                    sm_nums   = (chan_map.get_supermodule(max_chans[0][0]),
                                 chan_map.get_supermodule(max_chans[1][0]))
                    skew_corr = skew.get(max_chans[1][0], 0) - skew.get(max_chans[0][0], 0)
                    try:
                        coincidences[i][   'pair'] = p_lookup[sm_nums]
                        coincidences[i]['energy1'] = mm_energies[0]
                        coincidences[i]['energy2'] = mm_energies[1]
                        coincidences[i][   'time'] = max_chans[0][2] - max_chans[1][2] + skew_corr
                    except KeyError:
                        try:
                            coincidences[i][   'pair'] = p_lookup[sm_nums]
                            coincidences[i]['energy1'] = mm_energies[1]
                            coincidences[i]['energy2'] = mm_energies[0]
                            coincidences[i][   'time'] = max_chans[1][2] - max_chans[0][2] - skew_corr
                            channel_energies[[2 * i, 2 * i + 1]] = channel_energies[[2 * i + 1, 2 * i]]
                        except KeyError:
                            continue
                    channel_energies[2 * i    ]['slab_idx'] = max_chans[0][0]
                    channel_energies[2 * i + 1]['slab_idx'] = max_chans[1][0]
                    i += 1
            ## predict here. Fake returned object for now.
            predicted_xy = np.empty(npred, np.dtype(('X', np.float32), ('Y', np.float32)))
            for i in range(0, npred // 2):#This way forced by structure.
                pixels = tuple(map(lambda xy: pixel_vals(*xy), predicted_xy[2 * i:2 * i+2]))
                coincidences[i]['xPosition1'] = pixels[0][0]
                coincidences[i]['yPosition1'] = pixels[0][1]
                coincidences[i]['xPosition2'] = pixels[1][0]
                coincidences[i]['yPosition2'] = pixels[1][1]
                lm_out.write(coincidences[i])
        pass
    return _evt_loop


if __name__ == '__main__':
    args = docopt(__doc__)
    conf = configparser.ConfigParser()
    conf.read(args['--conf'])

    map_file = conf.get('mapping', 'map_file')
    infiles  = args['INPUT']

    chan_map  = ChannelMap(map_file)
    filt_type = conf.get('filter', 'type', fallback='maxCoinc')
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
    # Hardwiring which local coord is e maybe not ideal
    ecog       = energy_weighted_average(chan_map.get_plot_position, 1, 2)

    xlimits    = tuple(map(float, conf.get('output', 'pixels_x').split(',')))
    ylimits    = tuple(map(float, conf.get('output', 'pixels_y').split(',')))
    xpixels    = np.linspace(*xlimits[:2], int(xlimits[2]))
    ypixels    = np.linspace(*ylimits[:2], int(ylimits[2]))
    pixel_vals = local_pixel(xpixels, ypixels)
    p_cols     = ['p', 's0', 's1']
    pair_name  = conf.get('mapping', 'pairs')
    p_lookup   = pd.read_csv(pair_name     ,
                             sep   = '\t'  ,
                             names = p_cols).set_index(p_cols[1:]).to_dict()['p']

    ## Will probably have all this in PETsys but used for now.
    ## Can be left as an option if need be.
    tcal     = conf.get     ('calibration',   'time_channels' )
    ecal     = conf.get     ('calibration', 'energy_channels' )
    eref     = conf.getfloat('calibration', 'energy_reference', fallback=None)
    sel_func = equal_and_select(chan_map, tcal, ecal)
    skew     = {i: 0 for i in chan_map.mapping.index}#pd.Series(0, index=chan_map.mapping.index)#read_skewfile(conf.get('calibration', 'skew')).to_dict()
    ## This last convertor has to be here.
    mm_eng   = supermod_energy(convert_to_kev(conf.get('calibration', 'kev_convert'), chan_map.get_modules))
    ##
    out_dir  = conf.get('output', 'out_dir')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    evt_loop = cog_loop(chan_map, evt_reader, sel_func, eselect, pixel_vals, mm_eng, skew, p_lookup)
    for fn in infiles:
        # Open output file.
        out_file = os.path.join(out_dir, fn.split(os.sep)[-1].replace('.ldat', '_LM.bin'))
        with open(out_file, 'wb') as lm_out:
            ## Write header.
            write_header(lm_out, conf, xpixels, ypixels)
            ##
            evt_loop(fn, lm_out)
