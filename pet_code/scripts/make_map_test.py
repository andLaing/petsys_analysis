import os
import yaml

from pytest import fixture

from .. src.util import ChannelType

from . make_map import channel_sm_coordinate
from . make_map import np
from . make_map import pd
from . make_map import R
from . make_map import n_rings
from . make_map import row_gen
from . make_map import single_ring
from . make_map import sm_gen

from . make_map import mm_edge, mm_spacing, slab_width

from numpy.testing import assert_allclose


def test_channel_sm_coordinate():
    test_chans = [(0,  0, ChannelType.TIME  ),
                  (0,  0, ChannelType.ENERGY),
                  (3, 31, ChannelType.TIME  ),
                  (3, 31, ChannelType.ENERGY)]
    
    results = [channel_sm_coordinate(*args) for args in test_chans]

    exp_first = (round(mm_edge * 4 - (mm_spacing + slab_width) / 2, 3),
                 round(3.5 * mm_edge, 3))
    exp_last  = (round((mm_spacing + slab_width) / 2, 3), round(mm_edge / 2, 3))
    assert_allclose(results[0], exp_first      )
    assert_allclose(results[1], exp_first[::-1])
    assert_allclose(results[2], exp_last       )
    assert_allclose(results[3], exp_last [::-1])


@fixture(scope = 'module')
def channel_types(TEST_DATA_DIR):
    test_yaml = os.path.join(TEST_DATA_DIR, 'SM_mapping_1ring.yaml')

    with open(test_yaml) as map_file:
        ch_map = yaml.safe_load(map_file)

    return ch_map


def test_sm_gen(channel_types):
    tchans  = channel_types[  'time_channels']
    echans  = channel_types['energy_channels']
    feb_map = {0: [0, 0]}
    mm_emap = {0:0, 1:4,  2:8 ,  3:12,  4:1,  5:5,  6:9 ,  7:13,
               8:2, 9:6, 10:10, 11:14, 12:3, 13:7, 14:11, 15:15}

    gen_sm = sm_gen(256, 8, tchans, echans, mm_emap, feb_map)

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
    min_pos = (mm_spacing + slab_width) / 2
    max_pos = 4 * mm_edge
    assert all(sm_df.local_x >= min_pos)
    assert all(sm_df.local_x <= max_pos)
    assert all(sm_df.local_y >= min_pos)
    assert all(sm_df.local_y <= max_pos)


def test_single_ring(channel_types):
    tchans  = channel_types[  'time_channels']
    echans  = channel_types['energy_channels']
    ring_r  = channel_types[        'ring_r' ]
    ring_yx = channel_types[        'ring_yx']
    feb_map = channel_types[     'sm_feb_map']

    nFEM    = 256
    ring_df = single_ring(nFEM, 8, tchans, echans, ring_r, ring_yx, feb_map)

    cols = ['id', 'type', 'supermodule', 'minimodule',
            'local_x', 'local_y', 'X', 'Y', 'Z']
    assert ring_df.shape == (nFEM * 24, len(cols))
    assert all(hasattr(ring_df, col) for col in cols)

    # Approx centre of SM with means
    xyz        = ['X', 'Y', 'Z']
    yx_df      = pd.DataFrame(ring_yx, index=['Y', 'X']).T
    sm_centres = ring_df.groupby('supermodule')[xyz].mean()
    assert_allclose(sm_centres.Z,       0, atol=1e-4)
    assert_allclose(sm_centres.X, yx_df.X, rtol=1e-5)
    assert_allclose(sm_centres.Y, yx_df.Y, rtol=1e-5)

    # Check z of time channels in row always the same.
    mask =    (ring_df.supermodule == 3)\
            & (ring_df.minimodule.isin([0, 1, 2, 3]))\
            & (ring_df.type == 'TIME')
    assert_allclose(ring_df.Z[mask][1:], ring_df.Z[mask].iloc[0])


def test_single_ring_corners(TEST_DATA_DIR, channel_types):
    corners = os.path.join(TEST_DATA_DIR, 'IMAS1R_CORNERS-Table.csv')
    tchans  = channel_types[  'time_channels']
    echans  = channel_types['energy_channels']
    ring_r  = channel_types[        'ring_r' ]
    ring_yx = channel_types[        'ring_yx']
    feb_map = channel_types[     'sm_feb_map']
    yx_df   = pd.DataFrame(ring_yx, index=['Y', 'X']).T
    sm_angs = yx_df.apply(lambda x: np.arctan2(x.Y, x.X), axis=1)

    exp_xyz     = ['GlobalX', 'GlobalY', 'GlobalZ']
    exp_corners = pd.read_csv(corners, sep=';', decimal=',')
    # We're going to compare the local max and min corners
    exp_corners = (pd.concat((exp_corners.groupby('Module').head(1),
                              exp_corners.groupby('Module').tail(1)))
                     .sort_values('Module').reset_index(drop=True))

    nFEM    = 256
    ring_df = single_ring(nFEM, 8, tchans, echans, ring_r, ring_yx, feb_map)

    # Compare using the local_x min and max (TIME channels)
    corn_vec   = np.array([0, (mm_spacing + slab_width) / 2, -mm_edge / 2])
    cols       = ['X', 'Y', 'Z']
    time_smgrp = ring_df[ring_df.type.str.contains('TIME')].groupby('supermodule')
    min_max    = (pd.concat((time_smgrp.head(1), time_smgrp.tail(1)))
                    .sort_values(['supermodule', 'minimodule'])
                    .reset_index(drop=True))
    for sm, sm_info in min_max.groupby('supermodule'):
        to_corner = R.from_euler('z', sm_angs.at[sm]).apply(corn_vec)
        sm_corner = exp_corners[exp_corners.Module == sm]
        indx_min  = sm_corner.LocalX.idxmin()
        indx_max  = sm_corner.LocalX.idxmax()
        print('SM: ', sm, ', ', indx_min, ', ', indx_max)
        lmax_corner = sm_info.iloc[0][cols].astype('float').values + to_corner
        lmin_corner = sm_info.iloc[1][cols].astype('float').values - to_corner
        assert_allclose(lmin_corner, sm_corner.loc[indx_min][exp_xyz], rtol=5e-4)
        assert_allclose(lmax_corner, sm_corner.loc[indx_max][exp_xyz], rtol=5e-4)


def test_row_gen(TEST_DATA_DIR, channel_types):
    test_map       = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    tchans  = channel_types[  'time_channels']
    echans  = channel_types['energy_channels']
    # tchans, echans = channel_types

    cols = ['id', 'type', 'supermodule', 'minimodule', 'local_x', 'local_y', 'X', 'Y', 'Z']
    twoSM_gen = (row for row in row_gen(256, 8, tchans, echans))
    twoSM_map = pd.DataFrame(twoSM_gen, columns=cols)

    exp_map = pd.read_feather(test_map)
    pd.testing.assert_frame_equal(twoSM_map, exp_map)


def test_n_rings(channel_types):
    tchans  = channel_types[  'time_channels']
    echans  = channel_types['energy_channels']
    ring_r  = channel_types[        'ring_r' ]
    ring_yx = channel_types[        'ring_yx']
    feb_map = {sm: [sm // 6, 5 - sm % 6] for sm in range(24 * 5)}
    z_pos   = [-305.5312 + i * 152.7656 for i in range(5)]

    nFEM = 256
    n_sm =  24
    five_rings = n_rings(z_pos, nFEM, 8, tchans, echans, ring_r, ring_yx, feb_map)

    cols = ['id', 'type', 'supermodule', 'minimodule',
            'local_x', 'local_y', 'X', 'Y', 'Z']
    assert five_rings.shape == (nFEM * n_sm * len(z_pos), len(cols))
    assert all(hasattr(five_rings, col) for col in cols)
    assert five_rings.supermodule.unique().shape == (       n_sm * len(z_pos),)
    assert five_rings.id         .unique().shape == (nFEM * n_sm * len(z_pos),)

    # Check some randomly chosen SMs
    exp_cent = {  0: {'X':  406.4924, 'Y':  -53.5157, 'Z': -305.5312},
                 27: {'X':  249.5922, 'Y': -325.2749, 'Z': -152.7656},
                 68: {'X':  249.5922, 'Y':  325.2749, 'Z':    0.0   },
                 78: {'X':  -53.5157, 'Y': -406.4924, 'Z':  152.7656},
                112: {'X': -156.9002, 'Y':  378.7906, 'Z':  305.5312}}
    sm_mask  = five_rings.supermodule.isin(exp_cent.keys())
    sm_cols  = ['supermodule', 'X', 'Y', 'Z']
    mean_xyz = five_rings.loc[sm_mask, sm_cols].groupby('supermodule').mean()
    pd.testing.assert_frame_equal(mean_xyz         , pd.DataFrame(exp_cent).T    ,
                                  check_names=False, check_exact=False, atol=1e-4)
