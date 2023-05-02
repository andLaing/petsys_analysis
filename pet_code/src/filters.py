from itertools import chain
from typing    import Callable

from . util import np
from . util import get_no_eng_channels


def filter_impact(min_ch: int) -> Callable:
    """
    Make a filter to check impacts recorded
    in sufficient channels.
    """
    def valid_impact(mod_data: list[list]) -> bool:
        neng = get_no_eng_channels(mod_data)
        return min_ch < neng < len(mod_data)
    return valid_impact


def filter_multihit(sm: list[list], mm_map: Callable) -> bool:
    """
    Select super modules with one
    and only one mini modules hit.
    """
    n_mm = len(set(mm_map(x[0]) for x in sm))
    return n_mm == 1


def filter_event_by_impacts(min_sm: int, singles: bool=False) -> Callable:
    """
    Event filter based on the minimum energy channel
    hist for each of the two super modules in coincidence.
    """
    nch1_filter = filter_impact(min_sm)
    nch2_filter = nch1_filter
    if singles:
        nch2_filter = lambda x: True
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        return nch1_filter(sm1) and nch2_filter(sm2)
    return valid_event


def filter_one_minimod(sm1    : list[list],
                       sm2    : list[list],
                       mm_map : Callable  ,
                       singles: bool=False
                       ) -> bool:
    """
    Select events with only one
    minimodule hit in each super module.
    """
    if singles:
        return filter_multihit(sm1, mm_map)
    return filter_multihit(sm1, mm_map) and filter_multihit(sm2, mm_map)


def filter_impacts_one_minimod(min_sm : int       ,
                               mm_map : Callable  ,
                               singles: bool=False
                               ) -> Callable:
    """
    Event filter based on the minimum energy channel
    hist for each of the two super modules in coincidence
    with additional filter requiring only one mini module
    per super module.
    """
    nch1_filter = filter_impact(min_sm)
    nch2_filter = nch1_filter
    if singles:
        nch2_filter = lambda x: True
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        one_mm = filter_one_minimod(sm1, sm2, mm_map, singles)
        return one_mm and nch1_filter(sm1) and nch2_filter(sm2)
    return valid_event


def filter_specific_mm(sm_num: int, mm_num: int, mm_map: Callable) -> Callable:
    """
    Select events in a specific mini module.
    TODO review for full body.
    """
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        if sm_num == 0:
            return mm_num in set(mm_map(x[0]) for x in sm1)
        return mm_num in set(mm_map(x[0]) for x in sm2)
    return valid_event


def filter_channel_list(valid_channels: np.ndarray) -> Callable:
    """
    Filter event based on one of the impacts
    having channels in the valid list.
    """
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        return all(imp[0] in valid_channels for imp in sm1) or\
               all(imp[0] in valid_channels for imp in sm2)
    return valid_event


def filter_module_list(smMm_to_id: Callable         ,
                       valid_sm  : list | np.ndarray,
                       valid_mm  : list | np.ndarray
                       ) -> Callable:
    """
    Filter event based on list of
    valid supermodules and minimodules.
    """
    valid_ids = np.concatenate([smMm_to_id(*v) for v in np.vstack(np.stack(np.meshgrid(valid_sm, valid_mm)).T)])
    return filter_channel_list(valid_ids)


def filter_impacts_channel_list(valid_channels: list | np.ndarray,
                                min_ch        : int              ,
                                mm_map        : Callable
                                ) -> Callable:
    """
    Filter events by valid channels
    and that there's only one minimodule
    with information.
    Only for COINC mode!
    """
    sel_chans = filter_channel_list(valid_channels)
    ch_filter = filter_impact(min_ch)
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        return sel_chans(sm1, sm2) and filter_one_minimod(sm1, sm2, mm_map)\
                and ch_filter(sm1) and ch_filter(sm2)
    return valid_event


def filter_impacts_module_list(smMm_to_id: Callable,
                               sms       : list | np.ndarray,
                               mms       : list | np.ndarray,
                               min_sm    : int              ,
                               singles   : bool=False
                               ) -> Callable:
    """
    Combines requirements of impacts, specific mm and
    that only one module hit in both sm.
    """
    sel_mm     = filter_module_list(smMm_to_id, sms, mms)
    ch1_filter = filter_impact(min_sm)
    ch2_filter = ch1_filter
    if singles:
        ch2_filter = lambda x: True
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        return sel_mm(sm1, sm2) and ch1_filter(sm1) and ch2_filter(sm2)
    return valid_event


def filter_negatives(sm: list[list]) -> bool:
    """
    Return true if all energies
    are positive.
    """
    if not sm:
        return True
    # return all(np.asarray(sm)[:, 3] > 0)
    return all(imp[3] > 0 for imp in sm)


def filter_event_by_impacts_noneg(min_sm: int, singles: bool=False) -> Callable:
    """
    Event filter based on the minimum number of energy channels
    for each of the two super modules in coincidence filtering any
    events with negative signals.
    """
    ch1_filter = filter_impact(min_sm)
    ch2_filter = ch1_filter
    if singles:
        ch2_filter = lambda x: True
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        return ch1_filter(sm1) and ch2_filter(sm2) and\
            filter_negatives(sm1) and filter_negatives(sm2)
    return valid_event


def filter_max_sm(max_sm: int, sm_map: Callable) -> Callable:
    """
    Filter event with more than
    max_sm supermodules present.
    Only valid for coinc mode.
    """
    # vec_sm_map = np.vectorize(sm_map)
    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        sm_set = set()
        for imp in chain(sm1, sm2):
            sm_set.add(sm_map(imp[0]))
            if not (v_evt := len(sm_set) <= max_sm):
                break
        return v_evt
        # all_ids = np.vstack((sm1, sm2))[:, 0].astype('int')
        # return np.unique(vec_sm_map(all_ids)).shape[0] <= max_sm
    return valid_event


def filter_max_coin_event(sm_map : Callable    ,
                          max_sm : int  = 2    ,
                          min_ch : int  = 4    ,
                          singles: bool = False
                          ) -> Callable:
    """
    Filter event to select two SM
    and no negative event.
    """

    ch1_filter = filter_impact(min_ch)
    ch2_filter = ch1_filter
    if singles:
        ch2_filter = lambda x: True
    max_sm_filter = filter_max_sm(max_sm, sm_map)

    def valid_event(sm1: list[list], sm2: list[list]) -> bool:
        return filter_negatives(sm1) and filter_negatives(sm2) and\
            max_sm_filter(sm1, sm2) and ch1_filter(sm1) and ch2_filter(sm2)
    return valid_event
