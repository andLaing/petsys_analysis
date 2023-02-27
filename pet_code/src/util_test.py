import os

from pytest        import mark, fixture
from numpy.testing import assert_almost_equal

from . io   import read_ymlmapping
from . io   import ChannelMap
from . util import c_vac
from . util import np
from . util import ChannelType
from . util import centroid_calculation
from . util import get_no_eng_channels
from . util import get_absolute_id
from . util import get_electronics_nums
from . util import get_supermodule_eng
from . util import time_of_flight


def enum_dummy(SM):
    return list(map(lambda imp: [imp[0], ChannelType[imp[1]], imp[2], imp[3]], SM))


def test_get_no_eng_channels(DUMMY_SM):
    n_eng = get_no_eng_channels(enum_dummy(DUMMY_SM))
    assert n_eng == len(DUMMY_SM)


def test_get_supermodule_eng(DUMMY_SM):
    expected_eng   = sum(x[3] for x in DUMMY_SM)
    n_eng, tot_eng = get_supermodule_eng(enum_dummy(DUMMY_SM))
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


@mark.filterwarnings("ignore:Imported map")
def test_centroid_calculation(TEST_DATA_DIR, DUMMY_SM):
    test_mapping = os.path.join(TEST_DATA_DIR, "twoSM_IMAS_map.feather")
    channel_map  = ChannelMap(test_mapping)

    centroid                     = centroid_calculation(channel_map.plotp)
    time_mean, eng_mean, tot_eng = centroid(enum_dummy(DUMMY_SM))

    ## DUMMY not ideal as 100% energy channels, to be improved.
    expected_eng = sum((x[3] + 0.00001)**2 for x in DUMMY_SM)
    assert time_mean == 0.0
    assert_almost_equal(eng_mean,     43.51608, decimal=5)
    assert_almost_equal(tot_eng , expected_eng, decimal=5)


def test_time_of_flight(TEST_DATA_DIR):
    test_yml         = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    *_, slab_mapping = read_ymlmapping(test_yml)

    source_pos   = [38.4, 38.4, 22.5986]
    flight_time  = time_of_flight(source_pos)
    max_distance = 0.1888 # metres corner to corner in cal setup
    max_time     = max_distance / c_vac

    assert all(np.fromiter(map(flight_time, slab_mapping.values()), float) < max_time * 1e12)