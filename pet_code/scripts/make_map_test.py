import os
import yaml

from pytest import fixture

from .. src.util import ChannelType

from . make_map import channel_sm_coordinate
from . make_map import np
from . make_map import pd
from . make_map import row_gen
from . make_map import single_ring
from . make_map import sm_gen


def test_channel_sm_coordinate():
    test_chans = [(0,  0, ChannelType.TIME  ),
                  (0,  0, ChannelType.ENERGY),
                  (3, 31, ChannelType.TIME  ),
                  (3, 31, ChannelType.ENERGY)]
    
    results = [channel_sm_coordinate(*args) for args in test_chans]

    np.testing.assert_allclose(results[0], (101.85,  90.65))
    np.testing.assert_allclose(results[1], ( 90.65, 101.85))
    np.testing.assert_allclose(results[2], (  1.75,  12.95))
    np.testing.assert_allclose(results[3], ( 12.95,   1.75))


@fixture(scope = 'module')
def channel_types(TEST_DATA_DIR):
    test_yaml = os.path.join(TEST_DATA_DIR, 'SM_mapping_corrected.yaml')

    with open(test_yaml) as map_file:
        ch_map = yaml.safe_load(map_file)

    return ch_map['time_channels'], ch_map['energy_channels']


@fixture(scope = 'module')
def sm_ringYX():
    """
    Y, X for each SM centre. Ordering invert from normal.
    """
    yx_dict = { 0: ( -53.5157,  406.4924),  1: (-156.9002,  378.7906),
                2: (-249.5922,  325.2749),  3: (-325.2749,  249.5922),
                4: (-378.7906,  156.9002),  5: (-406.4924,   53.5157),
                6: (-406.4924,  -53.5157),  7: (-378.7906, -156.9002),
                8: (-325.2749, -249.5922),  9: (-249.5922, -325.2749),
               10: (-156.9002, -378.7906), 11: ( -53.5157, -406.4924),
               12: (  53.5157, -406.4924), 13: ( 156.9002, -378.7906),
               14: ( 249.5922, -325.2749), 15: ( 325.2749, -249.5922),
               16: ( 378.7906, -156.9002), 17: ( 406.4924,  -53.5157),
               18: ( 406.4924,   53.5157), 19: ( 378.7906,  156.9002),
               20: ( 325.2749,  249.5922), 21: ( 249.5922,  325.2749),
               22: ( 156.9002,  378.7906), 23: (  53.5157,  406.4924)}
    return pd.DataFrame(yx_dict, index=['Y', 'X']).T


def test_sm_gen(channel_types):
    tchans, echans = channel_types
    mm_emap = {0:0, 1:4,  2:8 ,  3:12,  4:1,  5:5,  6:9 ,  7:13,
               8:2, 9:6, 10:10, 11:14, 12:3, 13:7, 14:11, 15:15}

    gen_sm = sm_gen(256, 8, tchans, echans, mm_emap)

    cols  = ['id', 'type', 'minimodule', 'local_x', 'local_y']
    sm_df = pd.DataFrame((ch for ch in gen_sm(0)), columns=cols)

    assert sm_df.shape == (256, 5)
    time_chans = sm_df.type.str.contains('TIME')
    eng_chans  = sm_df.type.str.contains('ENERGY')
    assert sm_df[time_chans].shape == (128, 5)
    assert sm_df[ eng_chans].shape == (128, 5)
    assert set(tchans).issubset(sm_df.id)
    assert set(echans).issubset(sm_df.id)
    for mm in range(0, 16):
        assert sm_df[sm_df.minimodule == mm].shape[0] == 16
    # Probably need a more complete test for this
    assert all(sm_df.local_x >=   1.75)
    assert all(sm_df.local_x <= 101.85)
    assert all(sm_df.local_y >=   1.75)
    assert all(sm_df.local_y <= 101.85)


def test_single_ring(channel_types, sm_ringYX):
    tchans, echans = channel_types

    nFEM    = 256
    ring_df = single_ring(nFEM, 8, tchans, echans)

    cols = ['id', 'type', 'supermodule', 'minimodule',
            'local_x', 'local_y', 'X', 'Y', 'Z']
    assert ring_df.shape == (nFEM * 24, len(cols))
    assert all(hasattr(ring_df, col) for col in cols)

    # Approx centre of SM with means
    xyz        = ['X', 'Y', 'Z']
    sm_centres = ring_df.groupby('supermodule')[xyz].apply(np.mean)
    np.testing.assert_allclose(sm_centres.Z,           0, atol=1e-7)
    np.testing.assert_allclose(sm_centres.X, sm_ringYX.X)
    np.testing.assert_allclose(sm_centres.Y, sm_ringYX.Y)

    # Check z of time channels in row always the same.
    mask =    (ring_df.supermodule == 3)\
            & (ring_df.minimodule.isin([0, 1, 2, 3]))\
            & (ring_df.type == 'TIME')
    np.testing.assert_allclose(ring_df.Z[mask][1:], ring_df.Z[mask].iloc[0])


def test_row_gen(TEST_DATA_DIR, channel_types):
    test_map       = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    tchans, echans = channel_types

    cols = ['id', 'type', 'supermodule', 'minimodule', 'X', 'Y', 'Z', 'PLOTP', 'recID']
    twoSM_gen = (row for row in row_gen(256, 8, tchans, echans))
    twoSM_map = pd.DataFrame(twoSM_gen, columns=cols)

    exp_map = pd.read_feather(test_map)
    pd.testing.assert_frame_equal(twoSM_map, exp_map)
