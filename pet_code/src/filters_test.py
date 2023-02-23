import os

from pytest    import mark

from . io      import read_ymlmapping
from . io      import ChannelMap

from . filters import filter_event_by_impacts
from . filters import filter_impact
from . filters import filter_impacts_one_minimod
from . filters import filter_multihit
from . filters import filter_one_minimod
from . filters import filter_channel_list
from . filters import filter_module_list
from . filters import filter_impacts_specific_mod
from . filters import filter_event_by_impacts_noneg
from . filters import filter_max_sm

from . util_test import enum_dummy


def test_filter_impact(DUMMY_SM):
    impact_filter = filter_impact(4)
    ## The dummy event is all energy so not expected to pass. Improve test?
    assert not impact_filter(enum_dummy(DUMMY_SM))


def test_filter_multihit(TEST_DATA_DIR, DUMMY_SM):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, _, mm_map, *_ = read_ymlmapping(test_yml)
    assert filter_multihit(DUMMY_SM, lambda id: mm_map[id])


def test_filter_event_by_impacts(DUMMY_EVT):
    evt_select      = filter_event_by_impacts(4)
    dummy_with_enum = tuple(map(enum_dummy, DUMMY_EVT))
    assert evt_select(*dummy_with_enum)


def test_filter_one_minimod(TEST_DATA_DIR, DUMMY_EVT):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, _, mm_map, *_ = read_ymlmapping(test_yml)
    dummy_with_enum = tuple(map(enum_dummy, DUMMY_EVT))
    assert not filter_one_minimod(*dummy_with_enum, lambda id: mm_map[id])


def test_filter_impacts_one_minimod(TEST_DATA_DIR, DUMMY_EVT):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, _, mm_map, *_ = read_ymlmapping(test_yml)

    evt_select = filter_impacts_one_minimod(4, lambda id: mm_map[id])
    dummy_with_enum = tuple(map(enum_dummy, DUMMY_EVT))
    assert not evt_select(*dummy_with_enum)


def test_filter_channel_list(DUMMY_EVT):
    test_valid_channels   = {64, 112, 697, 699}
    test_invalid_channels = { 1,   2,   3}
    evt_select_valid      = filter_channel_list(test_valid_channels)
    evt_select_invalid    = filter_channel_list(test_invalid_channels)

    assert     evt_select_valid  (*DUMMY_EVT)
    assert not evt_select_invalid(*DUMMY_EVT)


@mark.filterwarnings("ignore:Imported map")
def test_filter_module_list(TEST_DATA_DIR, DUMMY_EVT):
    map_file = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    chan_map = ChannelMap(map_file)

    smMm_valid         = (0), (5, 6)
    smMm_invalid       = (2), (3, 4)
    mod_ids            = chan_map.get_minimodule_channels
    evt_select_valid   = filter_module_list(mod_ids, *smMm_valid  )
    evt_select_invalid = filter_module_list(mod_ids, *smMm_invalid)

    assert     evt_select_valid  (*DUMMY_EVT)
    assert not evt_select_invalid(*DUMMY_EVT)


@mark.filterwarnings("ignore:Imported map")
def test_filter_impacts_specific_mod(TEST_DATA_DIR, DUMMY_EVT):
    map_file = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    chan_map = ChannelMap(map_file)

    evt_select = filter_impacts_specific_mod(0, 11, chan_map.get_minimodule, 4)

    assert not evt_select(*DUMMY_EVT)


def test_filter_event_by_impacts_noneg(DUMMY_EVT):
    evt_select = filter_event_by_impacts_noneg(4)
    dummy_with_enum = tuple(map(enum_dummy, DUMMY_EVT))

    assert evt_select(*dummy_with_enum)


@mark.filterwarnings("ignore:Imported map")
def test_filter_max_sm(TEST_DATA_DIR, DUMMY_EVT):
    map_file = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    chan_map = ChannelMap(map_file)

    evt_select_valid   = filter_max_sm(2, chan_map.get_supermodule)
    evt_select_invalid = filter_max_sm(1, chan_map.get_supermodule)

    assert     evt_select_valid  (*DUMMY_EVT)
    assert not evt_select_invalid(*DUMMY_EVT)
