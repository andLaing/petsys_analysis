import numpy as np

from itertools import repeat

from scipy.constants import c as c_vac

def get_no_eng_channels(mod_data, energy_chid):
    """
    Return the number of channels for energy
    measurement in the module data list.
    """
    return sum(x[0] in energy_chid for x in mod_data)


def get_supermodule_eng(mod_data, energy_chid):
    """
    Return the number of channels for energy
    measurement in the module data and the
    total energy deposited.
    """
    eng_ch = list(filter(lambda x: x[0] in energy_chid, mod_data))
    return len(eng_ch), sum(hit[3] for hit in eng_ch)


def centroid_calculation(centroid_map, offset_x=0.00001, offset_y=0.00001):
    """
    Calculates the centroid of a set of module
    data according to a centroid map.
    """
    powers  = [1, 2]
    offsets = [offset_x, offset_y]
    def centroid(data):
        """
        Calculate the average position of the time
        and energy channels and return them plus
        the total energy channel deposit.
        """
        sums    = [0.0, 0.0]
        weights = [0.0, 0.0]
        for imp in data:
            en_t, pos      = centroid_map[imp[0]]
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
    return round(90.65 - 25.6 * row, 2)


# def slab_y(row):
#     """
#     Get the y position of the channel
#     given row number.
#     """
#     return round(89.6 - 25.6 * row, 2)
def slab_y(rc_num, sm_num=1):
    """
    Get the y position of the channel
    given the slab in row number (rc_num).
    sm_num == 2 is inverted in cal setup.
    TODO: generalise, only valid for
    current iteration of cal setup.
    """
    if sm_num == 2:
        return round(-100.8 + 3.2 * rc_num, 2)
    return round(-1.6 - 3.2 * rc_num, 2)


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


## Event and impact filters...
def filter_impact(min_ch, energy_chid):
    """
    Make a filter to check impacts recorded
    in sufficient channels.
    """
    def valid_impact(mod_data):
        neng = get_no_eng_channels(mod_data, energy_chid)
        return min_ch < neng < len(mod_data)
    return valid_impact


def filter_multihit(sm):
    """
    Select super modules with one
    and only one mini modules hit.
    """
    n_mm = len(set(x[1] for x in sm))
    return n_mm == 1


def filter_event_by_impacts(eng_map, min_sm1, min_sm2):
    """
    Event filter based on the minimum energy channel
    hist for each of the two super modules in coincidence.
    """
    m1_filter = filter_impact(min_sm1, eng_map)
    m2_filter = filter_impact(min_sm2, eng_map)
    def valid_event(sm1, sm2):
        return m1_filter(sm1) and m2_filter(sm2)
    return valid_event


def filter_one_minimod(sm1, sm2):
    """
    Select events with only one
    minimodule hit in each super module.
    """
    return filter_multihit(sm1) and filter_multihit(sm2)


def filter_impacts_one_minimod(eng_map, min_sm1, min_sm2):
    """
    Event filter based on the minimum energy channel
    hist for each of the two super modules in coincidence
    with additional filter requiring only one mini module
    per super module.
    """
    m1_filter = filter_impact(min_sm1, eng_map)
    m2_filter = filter_impact(min_sm2, eng_map)
    def valid_event(sm1, sm2):
        return filter_one_minimod(sm1, sm2) and m1_filter(sm1) and m2_filter(sm2)
    return valid_event


def filter_specific_mm(sm_num, mm_num):
    """
    Select events in a specific mini module.
    TODO review for full body.
    """
    def valid_event(sm1, sm2):
        if sm_num == 0:
            return mm_num in set(x[1] for x in sm1)
        return mm_num in set(x[1] for x in sm2)
    return valid_event


def filter_impacts_specific_mod(sm_num, mm_num, eng_map, min_sm1, min_sm2):
    """
    Combines requirements of impacts, specific mm and
    that only one module hit in both sm.
    """
    sel_mm = filter_specific_mm(sm_num, mm_num)
    m1_filter = filter_impact(min_sm1, eng_map)
    m2_filter = filter_impact(min_sm2, eng_map)
    def valid_event(sm1, sm2):
        return sel_mm(sm1, sm2) and filter_one_minimod(sm1, sm2)\
                and m1_filter(sm1) and m2_filter(sm2)
    return valid_event
## End filters (examples)


def select_module(sm_info, eng_ch):
    """
    Select the mini module with
    highest energy in a SM.
    """
    sm  = np.asarray(sm_info)
    mms = np.unique(sm[:, 1])
    if mms.shape[0] == 1:
        return sm_info
    e_chan = np.fromiter(map(lambda x: x[0] in eng_ch, sm), bool)
    sums = np.fromiter((sm[(sm[:, 1] == mm) & e_chan, 3].sum() for mm in mms), float)
    max_mm = mms[np.argmax(sums)]
    #return sm[sm[:, 1] == max_mm, :].tolist()
    return list(filter(lambda x: x[1] == max_mm, sm_info))


def shift_to_centres(bin_low_edge):
    """
    Get the bin centres from a list/array
    of lower edges (standard bin labels).
    """
    return bin_low_edge[:-1] + np.diff(bin_low_edge) * 0.5


def time_of_flight(source_pos):
    """
    Function to calculate time of flight
    for a gamma emitted from source_pos.
    source_pos: np.ndarray
                source position (x, y, z) in mm.
    """
    c_mm_per_ps = c_vac * 1000 / 1e12
    def flight_time(slab_pos):
        """
        Get the time of flight from source to
        the given slab position in mm.
        return flight time in ps
        """
        distance = np.linalg.norm(np.array(slab_pos) - source_pos)        
        return distance / c_mm_per_ps
    return flight_time


def mm_energy_centroids(events, c_calc, eng_ch, mod_sel=lambda sm: sm):
    """
    Calculate centroid and energy for
    mini modules per event assuming
    one mini module per SM per event.
    """
    mod_dicts = [{}, {}]
    for evt in events:
        sel_evt = tuple(map(mod_sel, evt))
        for i, ((x, y, _), (_, eng)) in enumerate(zip(map(c_calc, sel_evt), map(get_supermodule_eng, sel_evt, repeat(eng_ch)))):
            mm = evt[i][0][1]
            try:
                mod_dicts[i][mm]['x'].append(x)
                mod_dicts[i][mm]['y'].append(y)
                mod_dicts[i][mm]['energy'].append(eng)
            except KeyError:
                mod_dicts[i][mm] = {'x': [x], 'y': [y], 'energy': [eng]}
    return mod_dicts


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
            


def slab_energy_centroids(events, c_calc, time_ch):
    """
    Calculate centroids for mini module
    assuming one mini module per SM and
    save energy for the time channels.
    """
    slab_dicts = [{}, {}]
    for evt in events:
        for i, ((x, y, _), sm) in enumerate(zip(map(c_calc, evt), evt)):
            for imp in filter(lambda x: x[0] in time_ch, sm):
                try:
                    slab_dicts[i][imp[0]]['x'].append(x)
                    slab_dicts[i][imp[0]]['y'].append(y)
                    slab_dicts[i][imp[0]]['energy'].append(imp[3])
                except KeyError:
                    slab_dicts[i][imp[0]] = {'x': [x], 'y': [y], 'energy': [imp[3]]}
    return slab_dicts
