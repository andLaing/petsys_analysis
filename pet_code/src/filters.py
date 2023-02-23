from . util import np
from . util import get_no_eng_channels


def filter_impact(min_ch):
    """
    Make a filter to check impacts recorded
    in sufficient channels.
    """
    print("First commit")
    def valid_impact(mod_data):
        neng = get_no_eng_channels(mod_data)
        return min_ch < neng < len(mod_data)
    return valid_impact


def filter_multihit(sm, mm_map):
    """
    Select super modules with one
    and only one mini modules hit.
    """
    n_mm = len(set(mm_map(x[0]) for x in sm))
    return n_mm == 1


def filter_event_by_impacts(min_sm1, min_sm2, singles=False):
    """
    Event filter based on the minimum energy channel
    hist for each of the two super modules in coincidence.
    """
    m1_filter = filter_impact(min_sm1)
    m2_filter = lambda x: True
    if not singles:
        m2_filter = filter_impact(min_sm2)
    def valid_event(sm1, sm2):
        return m1_filter(sm1) and m2_filter(sm2)
    return valid_event


def filter_one_minimod(sm1, sm2, mm_map, singles=False):
    """
    Select events with only one
    minimodule hit in each super module.
    """
    if singles:
        return filter_multihit(sm1, mm_map)
    return filter_multihit(sm1, mm_map) and filter_multihit(sm2, mm_map)


def filter_impacts_one_minimod(min_sm1, min_sm2, mm_map, singles=False):
    """
    Event filter based on the minimum energy channel
    hist for each of the two super modules in coincidence
    with additional filter requiring only one mini module
    per super module.
    """
    m1_filter = filter_impact(min_sm1)
    m2_filter = lambda x: True
    if not singles:
        m2_filter = filter_impact(min_sm2)
    def valid_event(sm1, sm2):
        one_mm = filter_one_minimod(sm1, sm2, mm_map, singles)
        return one_mm and m1_filter(sm1) and m2_filter(sm2)
    return valid_event


def filter_specific_mm(sm_num, mm_num, mm_map):
    """
    Select events in a specific mini module.
    TODO review for full body.
    """
    def valid_event(sm1, sm2):
        if sm_num == 0:
            return mm_num in set(mm_map(x[0]) for x in sm1)
        return mm_num in set(mm_map(x[0]) for x in sm2)
    return valid_event


def filter_impacts_specific_mod(sm_num, mm_num, mm_map, min_sm1, min_sm2):
    """
    Combines requirements of impacts, specific mm and
    that only one module hit in both sm.
    """
    sel_mm    = filter_specific_mm(sm_num, mm_num, mm_map)
    m1_filter = filter_impact(min_sm1)
    m2_filter = filter_impact(min_sm2)
    def valid_event(sm1, sm2):
        return sel_mm(sm1, sm2) and filter_one_minimod(sm1, sm2)\
                and m1_filter(sm1) and m2_filter(sm2)
    return valid_event


def filter_impacts_mmgroup(mm_sm1, mm_sm2, mm_map, min_sm1, min_sm2):
    """
    Only use events with mini-modules within
    specific groups.
    """
    sm1_filter = filter_impact(min_sm1)
    sm2_filter = filter_impact(min_sm2)
    def valid_event(sm1, sm2):
        sm1_grp = (mm_map(id) in mm_sm1 for id, *_ in sm1)
        sm2_grp = (mm_map(id) in mm_sm2 for id, *_ in sm2)
        return all(sm1_grp) and all(sm2_grp) and sm1_filter(sm1) and sm2_filter
    return valid_event


def filter_negatives(sm):
    """
    Return true if all energies
    are positive.
    """
    if not sm:
        return True
    return all(np.asarray(sm)[:, 3] > 0)


def filter_event_by_impacts_noneg(min_sm1, min_sm2, singles=False):
    """
    Event filter based on the minimum number of energy channels
    for each of the two super modules in coincidence filtering any
    events with negative signals.
    """
    m1_filter = filter_impact(min_sm1)
    m2_filter = lambda x: True
    if not singles:
        m2_filter = filter_impact(min_sm2)
    def valid_event(sm1, sm2):
        return m1_filter(sm1) and m2_filter(sm2) and\
            filter_negatives(sm1) and filter_negatives(sm2)
    return valid_event
