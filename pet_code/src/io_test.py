import os

import pytest

from . io import np
from . io import read_petsys
from . io import read_petsys_filebyfile
from . io import read_ymlmapping
from . io import write_event_trace


@pytest.fixture(scope = 'module')
def module_mapping():
    """
    Mapping of modules for test.
    To avoid using read_ymlmapping in other io tests.
    """
    time_channels = [ 55,  54,  57,  56,  59,  58,  61,  60,  37,  35,  33,
                      36,  34,  31,  32,  29,  79,  91,  81,  82,  83,  88,
                      86,  84,  80,  85,  87,  77,  89,  74,  90,  75,  13,
                      12,  10,  14,   6,   4,   2,   0,   8,  16,  15,  18,
                      17,  19,  21,  23, 102, 103, 104, 105, 106, 107, 108,
                     109,  68,  69,  66,  67,  64,  65, 112, 114, 178, 176,
                     129, 128, 131, 130, 133, 132, 173, 172, 171, 170, 169,
                     168, 167, 166, 215, 213, 211, 209, 210, 207, 208, 200,
                     192, 194, 196, 198, 206, 202, 204, 205, 139, 154, 138,
                     153, 141, 151, 149, 144, 148, 150, 152, 147, 146, 145,
                     155, 143, 221, 224, 223, 226, 225, 228, 227, 229, 252,
                     253, 250, 251, 248, 249, 246, 247]
    eng_channels  = [ 49,  46,  48,  41,  50,  51,  53,  52,  63,  62,   1,
                       3,   5,   7,   9,  11, 185, 184, 183, 182, 181, 180,
                     179, 177, 164, 135, 162, 134, 137, 158, 136, 156,  47,
                      44,  45,  42,  40,  43,  38,  39,  30,  27,  28,  26,
                      24,  25,  22,  20, 186, 190, 191, 189, 188, 187, 175,
                     174, 165, 163, 161, 140, 160, 159, 142, 157,  93,  78,
                      95,  96,  76,  97,  99, 101, 110, 111, 123, 124, 125,
                     127, 126, 122, 212, 214, 217, 216, 218, 220, 219, 222,
                     231, 230, 235, 232, 234, 237, 236, 239,  92,  72,  94,
                      73,  70,  98,  71, 100, 113, 115, 116, 117, 118, 119,
                     120, 121, 203, 201, 199, 197, 195, 193, 254, 255, 244,
                     245, 243, 242, 233, 240, 238, 241]
    mapping = {}
    FEM_num_ch = 256
    slab_num   =   1
    mM_energyMapping = {1:1,  2:5,  3:9 ,  4:13,  5:2,  6:6,  7:10,  8:14,
                        9:3, 10:7, 11:11, 12:15, 13:4, 14:8, 15:12, 16:16}
    for i in range(4):
        mm_num = 1
        for t_ch, e_ch in zip(time_channels, eng_channels):
            abs_tch = t_ch + i * FEM_num_ch
            abs_ech = e_ch + i * FEM_num_ch
            mM_num_en = mM_energyMapping[mm_num]
            mapping[abs_tch] = mm_num
            mapping[abs_ech] = mM_num_en

            if slab_num%8 == 0:
                mm_num += 1
            slab_num += 1
    return mapping


def test_read_petsys(TEST_DATA_DIR, module_mapping):
    infile = os.path.join(TEST_DATA_DIR,
                          "petsys_test_TB_SMGathered_120s_4.0OV_10T1_15T2_5E_coinc.ldat")
    pet_reader = read_petsys(module_mapping)
    all_evts = [evt for evt in pet_reader([infile])]

    assert len(all_evts) == 20
    assert all(np.fromiter(map(len, all_evts), int) == 2)
    # Check no repetition of ids in events.
    assert all(len(set(np.array(sm)[:, 0])) == len(sm) for sm, _  in all_evts)
    assert all(len(set(np.array(sm)[:, 0])) == len(sm) for _ , sm in all_evts)
    ids1 = [x[0] for evt in all_evts for x in evt[0]]
    ids2 = [x[0] for evt in all_evts for x in evt[1]]
    assert set(ids1).issubset(module_mapping.keys())
    assert set(ids2).issubset(module_mapping.keys())
    # Check energy and timestamp?


def test_read_petsys_mod1(TEST_DATA_DIR, module_mapping):
    infile = os.path.join(TEST_DATA_DIR,
                          "petsys_test_TB_SMGathered_120s_4.0OV_10T1_15T2_5E_coinc.ldat")
    def filt_one_mod(sm1, _):
        return all(np.array(sm1)[:, 1] == 9)
    pet_reader = read_petsys(module_mapping, filt_one_mod)
    all_evts = [evt for evt in pet_reader([infile])]

    exp_evt1 = ([[696, 9, 1137456113288,  5.1196], [644, 9, 1137456104789,  4.9302],
                 [697, 9, 1137456111320,  4.9263], [695, 9, 1137456114648,  4.1518],
                 [694, 9, 1137456117132,  3.9170], [693, 9, 1137456116532,  3.0733],
                 [692, 9, 1137456118194,  2.8072], [691, 9, 1137456118303,  2.3376],
                 [689, 9, 1137456118804,  1.5792], [645, 9, 1137456105668,  0.5493],
                 [642, 9, 1137456126721,  0.3673]],
                [[  7, 5, 1137456110189, 22.7387], [  5, 5, 1137456110402, 19.2009],
                 [  9, 5, 1137456110310, 17.8486], [  3, 5, 1137456111027, 16.4262],
                 [ 63, 5, 1137456111874, 15.2988], [ 11, 5, 1137456111220, 14.6819],
                 [ 62, 5, 1137456111221, 13.9333], [  1, 5, 1137456111258, 12.3697],
                 [  4, 5, 1137456106047,  4.5095], [  6, 5, 1137456109639,  1.8361],
                 [ 14, 5, 1137456119187,  1.3980], [ 10, 5, 1137456128473,  0.7132]])
    assert len(all_evts) == 2
    np.testing.assert_allclose(all_evts[0][0], exp_evt1[0], rtol=1e-4)
    np.testing.assert_allclose(all_evts[0][1], exp_evt1[1], rtol=1e-4)

# Should test in singles mode too one have a file.

def test_read_petsys_filebyfile(TEST_DATA_DIR, module_mapping):
    infile = os.path.join(TEST_DATA_DIR,
                          "petsys_test_TB_SMGathered_120s_4.0OV_10T1_15T2_5E_coinc.ldat")
    pet_reader = read_petsys_filebyfile(infile, module_mapping)
    all_evts = [evt for evt in pet_reader()]

    assert len(all_evts) == 20
    assert all(np.fromiter(map(len, all_evts), int) == 2)
    # Check no repetition of ids in events.
    assert all(len(set(np.array(sm)[:, 0])) == len(sm) for sm, _  in all_evts)
    assert all(len(set(np.array(sm)[:, 0])) == len(sm) for _ , sm in all_evts)
    ids1 = [x[0] for evt in all_evts for x in evt[0]]
    ids2 = [x[0] for evt in all_evts for x in evt[1]]
    assert set(ids1).issubset(module_mapping.keys())
    assert set(ids2).issubset(module_mapping.keys())
    # Check energy and timestamp?


def test_read_ymlmapping(TEST_DATA_DIR):
    test_yml = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")

    all_time, all_eng, mm_map, cent_map, slab_positions = read_ymlmapping(test_yml)

    FEM_num_ch = 256
    # Expect two super modules
    assert len(all_time) == 2 * FEM_num_ch
    assert len(all_eng)  == 2 * FEM_num_ch
    assert all_time.issubset(mm_map.keys())
    assert all_eng .issubset(mm_map.keys())
    ## Need more tests


def test_nonyml_raises(TMP_TXT):
    with pytest.raises(RuntimeError):
        catch = read_ymlmapping(TMP_TXT)


def test_write_event_trace(TEST_DATA_DIR, TMP_OUT, DUMMY_SM):
    test_yml         = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    *_, centroid_map, _ = read_ymlmapping(test_yml)

    tmp_out = os.path.join(TMP_OUT, 'first_test.txt')
    with open(tmp_out, 'w') as out_buf:
        writer = write_event_trace(out_buf, centroid_map)
        writer(DUMMY_SM)

    with open(tmp_out) as txt_test:
        all_values = txt_test.read().split('\t')
        assert len(all_values    ) == 17
        assert int(all_values[-1]) == 10
        assert all(float(val) >= 0 for val in all_values[:-1])
