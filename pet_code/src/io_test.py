import os

from pytest import raises

from . io import read_ymlmapping
from . io import write_event_trace


def test_read_ymlmapping(TEST_DATA_DIR):
    test_yml = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")

    all_time, all_eng, mm_map, cent_map = read_ymlmapping(test_yml)

    FEM_num_ch = 256
    # Expect two super modules
    assert len(all_time) == 2 * FEM_num_ch
    assert len(all_eng)  == 2 * FEM_num_ch
    assert set(all_time).issubset(mm_map.keys())
    assert set(all_eng) .issubset(mm_map.keys())
    ## Need more tests


def test_nonyml_raises(TMP_TXT):
    with raises(RuntimeError):
        catch = read_ymlmapping(TMP_TXT)


def test_write_event_trace(TEST_DATA_DIR, TMP_OUT, DUMMY_EVT):
    test_yml         = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    *_, centroid_map = read_ymlmapping(test_yml)

    tmp_out = os.path.join(TMP_OUT, 'first_test.txt')
    with open(tmp_out, 'w') as out_buf:
        writer = write_event_trace(out_buf, centroid_map)
        writer(DUMMY_EVT)

    with open(tmp_out) as txt_test:
        all_values = txt_test.read().split('\t')
        assert len(all_values    ) == 17
        assert int(all_values[-1]) == 10
        assert all(float(val) >= 0 for val in all_values[:-1])
