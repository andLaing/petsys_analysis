import numpy  as np
import pandas as pd

from enum   import auto, Enum
from typing import List, Callable, Union

from scipy.constants import c as c_vac


class ChannelType(Enum):
    TIME   = auto()
    ENERGY = auto()


def get_no_eng_channels(mod_data):
    """
    Return the number of channels for energy
    measurement in the module data list.
    """
    return sum(x[1] is ChannelType.ENERGY for x in mod_data)


def get_supermodule_eng(mod_data):
    """
    Return the number of channels for energy
    measurement in the module data and the
    total energy deposited.
    """
    eng_ch = list(filter(lambda x: x[1] is ChannelType.ENERGY, mod_data))
    return len(eng_ch), sum(hit[3] for hit in eng_ch)


def centroid_calculation(plot_pos, offset_x=0.00001, offset_y=0.00001):
    """
    Calculates the centroid of a set of module
    data according to a centroid map.
    """
    powers  = [1, 2]
    offsets = [offset_x, offset_y]
    plot_ax = ['local_x', 'local_y']
    def centroid(data):
        """
        Calculate the average position of the time
        and energy channels and return them plus
        the total energy channel deposit.
        """
        sums    = [0.0, 0.0]
        weights = [0.0, 0.0]
        for imp in data:
            en_t           = imp[1].value - 1
            pos            = plot_pos[imp[0]][plot_ax[en_t]]
            weight         = (imp[3] + offsets[en_t])**powers[en_t]
            sums   [en_t] += weight * pos
            weights[en_t] += weight
        return (sums[0] / weights[0] if weights[0] else 0.0,
                sums[1] / weights[1] if weights[1] else 0.0, weights[1])
    return centroid


def slab_indx(pos):
    """
    Calculate the index of
    a slab in a mini-module using
    its position.
    """
    slab_num = round((pos - 1.6) / 3.2) # Safe to have these hardwired?
    indx     = slab_num - slab_num // 8 * 8
    return indx if indx < 8 else indx - 8


# def slab_x(rc_num, sm_num=1):
#     """
#     Get the x position of the channel
#     given rc_num.
#     super module 2 gives different result.
#     TODO: generalise!
#     """
#     if sm_num == 2:
#         return round(100.8 - 3.2 * rc_num, 2)
#     return round(1.6 + 3.2 * rc_num, 2)
def slab_x(row):
    """
    Get the x position of the channel
    given the mm row.
    TODO: generalise, only valid for
    current iteration of cal setup.
    """
    # return round(90.65 - 25.6 * row, 2)
    return round(25.9 * (0.5 + row % 4), 2)


# def slab_y(row):
#     """
#     Get the y position of the channel
#     given row number.
#     TODO: generalise, only valid for
#     current iteration of cal setup.
#     """
#     return round(103.6 - 25.9 * (0.5 + row), 2)
def slab_y(rc_num, sm_num=1):
    """
    Get the y position of the channel
    given the slab in row number (rc_num).
    sm_num == 2 is inverted in cal setup.
    TODO: generalise, only valid for
    current iteration of cal setup.
    """
    # if sm_num == 2:
    #     return round(-100.8 + 3.2 * rc_num, 2)
    # return round(-1.6 - 3.2 * rc_num, 2)
    # Is this correction correct?
    # Correct for extra spacing between MMs
    mm_wrap = round(0.3 * (rc_num // 8), 2)
    if sm_num == 2:
        return round(-1.75 - mm_wrap - 3.2 * rc_num, 2)
    return round(-103.6 + 1.75 + mm_wrap + 3.2 * rc_num, 2)


def slab_z(sm_num):
    """
    Get the z position of super module.
    TODO Generalise, hardwired for calibration setup.
    """
    return 123.7971 if sm_num == 2 else 0


def select_energy_range(minE, maxE):
    """
    Return a function that can select
    energies in an open range (minE, maxE).
    """
    def select_eng(eng_val):
        return (eng_val > minE) & (eng_val < maxE)
    return select_eng


def select_mod_wrapper(fn, mm_map):
    sel_mod = select_module(mm_map)
    def wrapped(evt):
        max_mms = tuple(map(sel_mod, evt))
        return fn(max_mms)
    return wrapped


# def select_module(mm_map):
#     """
#     Select the mini module with
#     highest energy in a SM.
#     """
#     # Is there a vectorize decorator?
#     to_mm   = np.vectorize(mm_map)
#     is_eng  = np.vectorize(lambda x: x is ChannelType.ENERGY)
#     def select(sm_info):
#         if not sm_info:
#             return sm_info

#         sm_arr  = np.asarray(sm_info, dtype='object')
#         mms     = to_mm(sm_arr[:, 0])
#         mms_uni = np.unique(mms)
#         if mms_uni.shape[0] == 1:
#             return sm_info

#         e_chan = is_eng(sm_arr[:, 1])
#         sums   = np.fromiter((sm_arr[(mms == mm) & e_chan, 3].sum() for mm in mms_uni), float)
#         max_mm = mms_uni[np.argmax(sums)]
#         return sm_arr[mms == max_mm, :].tolist()
#     return select


def select_module(mm_map):
    """
    Select the mini module with highest energy in sm.
    Version to avoid converting to numpy.
    """
    def val_if_eng(impact):
        return impact[3] if impact[1] is ChannelType.ENERGY else 0
    
    def select(sm_info):
        if not sm_info:
            return sm_info
        
        mm_dict = {}
        for imp in sm_info:
            mm = mm_map(imp[0])
            try:
                mm_dict[mm][0] += val_if_eng(imp)
                mm_dict[mm][1].append(imp)
            except KeyError:
                mm_dict[mm] = [val_if_eng(imp), [imp]]
                
        return max(mm_dict.values(), key=lambda x: x[0])[1]
    return select


def get_electronics_nums(channel_id):
    """
    Calculates the electronics numbers:
    portID, slaveID, chipID, channelID
    """
    portID    =   channel_id                   // 131072
    slaveID   =  (channel_id % 131072)         //   4096
    chipID    = ((channel_id % 131072) % 4096) //     64
    channelID =   channel_id                   %      64
    return portID, slaveID, chipID, channelID


def get_absolute_id(portID, slaveID, chipID, channelID):
    """
    Calculates absolute channel id from
    electronics numbers.
    """
    return 131072 * portID + 4096 * slaveID + 64 * chipID + channelID


def select_max_energy(superm, chan_type=None):
    """
    Select the channel with highest deposit.
    superm    : List
               List of impacts with [id, mm, time, eng]
    chan_type : Optional ChannelType
                The channel type to be compared.
    """
    try:
        if chan_type is None:
            return max(superm, key=lambda x: x[3])
        return max(filter(lambda x: x[1] is chan_type, superm), key=lambda y: y[3])
    except ValueError:
        return None


def shift_to_centres(bin_low_edge):
    """
    Get the bin centres from a list/array
    of lower edges (standard bin labels).
    """
    return bin_low_edge[:-1] + np.diff(bin_low_edge) * 0.5


def time_of_flight(source_pos: np.ndarray):
    """
    Function to calculate time of flight
    for a gamma emitted from source_pos.
    source_pos: np.ndarray
                source position (x, y, z) in mm.
    """
    c_mm_per_ps = c_vac * 1000 / 1e12
    def flight_time(slab_pos: Union[List, np.ndarray]):
        """
        Get the time of flight from source to
        the given slab position in mm.
        return flight time in ps
        """
        distance = np.linalg.norm(np.asarray(slab_pos) - source_pos)        
        return distance / c_mm_per_ps
    return flight_time


def mm_energy_centroids(c_calc, sm_map, mm_map, mod_sel=lambda sm: sm):
    """
    Calculate centroid and energy for
    mini modules per event assuming
    one mini module per SM per event.
    """
    def _mm_ecentroids(events):
        mod_dicts = {}
        for evt in events:
            sel_evt = tuple(filter(lambda x: x, map(mod_sel, evt)))
            for ev_sel in sel_evt:
                sm = sm_map(ev_sel[0][0])
                mm = mm_map(ev_sel[0][0])
                x, y, _ = c_calc(ev_sel)
                _, eng  = get_supermodule_eng(ev_sel)
            # for ((x, y, _), (_, eng)) in zip(map(c_calc, sel_evt),
            #                                  map(get_supermodule_eng, sel_evt)):
            #     sm = sm_map(sel_evt[i][0][0])
            #     mm = mm_map(sel_evt[i][0][0])
                try:
                    mod_dicts[sm][mm]['x'].append(x)
                    mod_dicts[sm][mm]['y'].append(y)
                    mod_dicts[sm][mm]['energy'].append(eng)
                except KeyError:
                    try:
                        mod_dicts[sm][mm] = {'x': [x], 'y': [y], 'energy': [eng]}
                    except KeyError:
                        mod_dicts[sm] = {mm: {'x': [x], 'y': [y], 'energy': [eng]}}
        return mod_dicts
    return _mm_ecentroids


def all_mm_energy_centroids(events, c_calc, eng_ch):
    """
    HAck for now to work with multiple mm per sm.
    """
    mod_dicts = [{}, {}]
    for evt in events:
        arr_evt = list(map(np.asarray, evt))
        for i, sm in enumerate(arr_evt):
            for mm in np.unique(sm[:, 1]):
                mm_sel    = sm[sm[:, 1] == mm, :]
                x, y  , _ = c_calc(mm_sel)
                _, eng    = get_supermodule_eng(mm_sel, eng_ch)
                try:
                    mod_dicts[i][mm]['x'].append(x)
                    mod_dicts[i][mm]['y'].append(y)
                    mod_dicts[i][mm]['energy'].append(eng)
                except KeyError:
                    mod_dicts[i][mm] = {'x': [x], 'y': [y], 'energy': [eng]}
    return mod_dicts
            


def slab_energy_centroids(events, c_calc):
    """
    Calculate centroids for mini module
    assuming one mini module per SM and
    save energy for the time channels.
    """
    slab_dicts = {}
    for evt in events:
        for (x, y, _), sm in zip(map(c_calc, evt), evt):
            for imp in filter(lambda x: x[1] is ChannelType.TIME, sm):
                try:
                    slab_dicts[imp[0]]['x'].append(x)
                    slab_dicts[imp[0]]['y'].append(y)
                    slab_dicts[imp[0]]['energy'].append(imp[3])
                except KeyError:
                    slab_dicts[imp[0]] = {'x': [x], 'y': [y], 'energy': [imp[3]]}
    return slab_dicts


def calibrate_energies(type_ids, time_cal, eng_cal, sep='\t'):
    """
    Equalize the energy for the channels
    given the peak positions in a file (for now)
    for time channels and energy channels.
    OBSOLETE??
    """
    if not time_cal and not eng_cal:
        # No calibration.
        return lambda x: x

    if time_cal:
        # Need to fix file format to remove space
        # time calibrated relative to 511 keV peak
        tcal    = pd.read_csv(time_cal, sep=sep).set_index('ID')['MU'].apply(lambda x: 511 / x)
    else:
        tcal    = pd.Series(1, index=type_ids(ChannelType.TIME))
    if eng_cal:
        # Need to fix file format to remove space
        # Energy channels calibrated relative to mean. Maybe unstable between calibrations, review.
        ecal    = pd.read_csv(eng_cal, sep=sep).set_index('ID')['MU']
        mu_mean = np.mean(ecal)
        ecal    = ecal.apply(lambda x: mu_mean / x)
    else:
        ecal    = pd.Series(1, index=type_ids(ChannelType.ENERGY))

    cal = tcal.append(ecal).to_dict()
    def apply_calibration(event):
        for sm in event:
            for imp in sm:
                imp[3] *= cal.get(imp[0], 0.0)#Effectively mask channels without calibration
        return event
    return apply_calibration


def bar_source_dt(bar_xy: np.ndarray, bar_r: float, slab_pos: Callable) -> Callable:
    """
    Define an axially infinite bar of radius bar_r centred on
    bar_xy so that geometric dt can be predicted.

    bar_xy : np.ndarray
             XY postion of bar transverse centre
    bar_r  : float
             Radius of bar
    slab_pos : Callable
               Function returning pos
    """
    c_mm_per_ps = c_vac * 1000 / 1e12
    c_corr = np.square(bar_xy).sum() - bar_r**2
    def geom_dt(ref_id: int, coinc_id: int) -> float:
        ref_pos   = slab_pos(  ref_id)
        coinc_pos = slab_pos(coinc_id)
        dir_norm  = np.linalg.norm(coinc_pos - ref_pos)
        dir       = (coinc_pos - ref_pos) / dir_norm

        a = np.square(dir[:2]).sum()
        b = 2 * (dir[0] * (ref_pos[0] - bar_xy[0]) + dir[1] * (ref_pos[1] - bar_xy[1]))
        c = ref_pos[0] * (ref_pos[0] - 2 * bar_xy[0]) + ref_pos[1] * (ref_pos[1] - 2 * bar_xy[1]) + c_corr

        determ = b**2 - 4 * a * c
        if determ < 0:
            return

        t1    = 0.5 * (-b + np.sqrt(determ)) / a
        t2    = 0.5 * (-b - np.sqrt(determ)) / a
        med_t = (t1 + t2) / 2
        dt    = (2 * med_t  - dir_norm) / c_mm_per_ps
        return dt
    return geom_dt

