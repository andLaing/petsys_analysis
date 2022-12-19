import os

from numpy.testing import assert_almost_equal

from . io   import read_ymlmapping
from . util import c_vac
from . util import np
from . util import centroid_calculation
from . util import filter_event_by_impacts
from . util import filter_impact
from . util import filter_impacts_one_minimod
from . util import filter_multihit
from . util import filter_one_minimod
from . util import get_no_eng_channels
from . util import get_absolute_id
from . util import get_electronics_nums
from . util import get_supermodule_eng
from . util import time_of_flight


def test_get_no_eng_channels(TEST_DATA_DIR, DUMMY_SM):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    n_eng = get_no_eng_channels(DUMMY_SM, energy_chid)
    assert n_eng == len(DUMMY_SM)


def test_get_supermodule_eng(TEST_DATA_DIR, DUMMY_SM):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    expected_eng   = sum(x[3] for x in DUMMY_SM)
    n_eng, tot_eng = get_supermodule_eng(DUMMY_SM, energy_chid)
    assert n_eng == len(DUMMY_SM)
    assert_almost_equal(tot_eng, expected_eng, decimal=5)


def test_get_absolute_id():
    elec_nums = [(0, 0, 0, 1), (0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 5)]
    exp_ids   = [          1 ,          129,         4161,       135237]

    results = [get_absolute_id(*nums) for nums in elec_nums]
    assert all([x == y for x, y in zip(results, exp_ids)])


def test_get_electronics_nums():
    ids      = [          1 ,          129,         4161,       135237]
    exp_nums = [(0, 0, 0, 1), (0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 5)]

    results = [get_electronics_nums(id) for id in ids]
    assert all([x == y for x, y in zip(results, exp_nums)])


def test_centroid_calculation(TEST_DATA_DIR, DUMMY_SM):
    test_yml            = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    *_, centroid_map, _ = read_ymlmapping(test_yml)

    centroid = centroid_calculation(centroid_map)
    time_mean, eng_mean, tot_eng = centroid(DUMMY_SM)

    ## DUMMY not ideal as 100% energy channels, to be improved.
    expected_eng = sum((x[3] + 0.00001)**2 for x in DUMMY_SM)
    assert time_mean == 0.0
    assert_almost_equal(eng_mean, 40.00842, decimal=5)
    assert_almost_equal(tot_eng, expected_eng, decimal=5)


def test_filter_impact(TEST_DATA_DIR, DUMMY_SM):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    impact_filter = filter_impact(4, energy_chid)
    ## The dummy event is all energy so not expected to pass. Improve test?
    assert not impact_filter(DUMMY_SM)


def test_filter_multihit(DUMMY_SM):
    assert filter_multihit(DUMMY_SM)


def test_filter_event_by_impacts(TEST_DATA_DIR, DUMMY_EVT):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    evt_select = filter_event_by_impacts(energy_chid, 5, 4)
    assert evt_select(*DUMMY_EVT)


def test_filter_one_minimod(DUMMY_EVT):
    assert not filter_one_minimod(*DUMMY_EVT)


def test_filter_impacts_one_minimod(TEST_DATA_DIR, DUMMY_EVT):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    evt_select = filter_impacts_one_minimod(energy_chid, 5, 4)
    assert not evt_select(*DUMMY_EVT)


def test_time_of_flight(TEST_DATA_DIR):
    test_yml         = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    *_, slab_mapping = read_ymlmapping(test_yml)

    source_pos   = [38.4, 38.4, 22.5986]
    flight_time  = time_of_flight(source_pos)
    max_distance = 0.1888 # metres corner to corner in cal setup
    max_time     = max_distance / c_vac

    assert all(np.fromiter(map(flight_time, slab_mapping.values()), float) < max_time * 1e12)