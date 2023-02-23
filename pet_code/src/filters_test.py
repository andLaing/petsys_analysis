import os

from . io      import read_ymlmapping

from . filters import filter_event_by_impacts
from . filters import filter_impact
from . filters import filter_impacts_one_minimod
from . filters import filter_multihit
from . filters import filter_one_minimod

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
    evt_select      = filter_event_by_impacts(5, 4)
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

    evt_select = filter_impacts_one_minimod(5, 4, lambda id: mm_map[id])
    dummy_with_enum = tuple(map(enum_dummy, DUMMY_EVT))
    assert not evt_select(*dummy_with_enum)