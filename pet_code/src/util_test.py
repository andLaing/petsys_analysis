import os
from unicodedata import decimal

from numpy.testing import assert_almost_equal

from . io   import read_ymlmapping
from . util import centroid_calculation
from . util import filter_impact
from . util import get_no_eng_channels
from . util import get_supermodule_eng


def test_get_no_eng_channels(TEST_DATA_DIR, DUMMY_EVT):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    n_eng = get_no_eng_channels(DUMMY_EVT, energy_chid)
    assert n_eng == len(DUMMY_EVT)


def test_get_supermodule_eng(TEST_DATA_DIR, DUMMY_EVT):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    expected_eng   = sum(x[3] for x in DUMMY_EVT)
    n_eng, tot_eng = get_supermodule_eng(DUMMY_EVT, energy_chid)
    assert n_eng == len(DUMMY_EVT)
    assert_almost_equal(tot_eng, expected_eng, decimal=5)


def test_filter_impact(TEST_DATA_DIR, DUMMY_EVT):
    test_yml           = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, energy_chid, *_ = read_ymlmapping(test_yml)

    impact_filter = filter_impact(4, energy_chid)
    ## The dummy event is all energy so not expected to pass. Improve test?
    assert not impact_filter(DUMMY_EVT)


def test_centroid_calculation(TEST_DATA_DIR, DUMMY_EVT):
    test_yml         = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    *_, centroid_map = read_ymlmapping(test_yml)

    centroid = centroid_calculation(centroid_map)
    time_mean, eng_mean, tot_eng = centroid(DUMMY_EVT)

    ## DUMMY not ideal as 100% energy channels, to be improved.
    expected_eng = sum((x[3] + 0.00001)**2 for x in DUMMY_EVT)
    assert time_mean == 0.0
    assert_almost_equal(eng_mean, 40.00842, decimal=5)
    assert_almost_equal(tot_eng, expected_eng, decimal=5)