import os
import configparser
import yaml

from pytest import approx, fixture, mark
from types  import FunctionType

from .. src.io   import ChannelMap

from . skew_calc import np
from . skew_calc import pd
from . skew_calc import calculate_skews
from . skew_calc import geom_loc_point
from . skew_calc import geom_loc_bar
from . skew_calc import peak_position
from . skew_calc import process_raw_data


@fixture(scope = 'module')
def deltat_df():
    nsamples   = 10000
    dist_mean  =   100
    dist_sigma =    10
    dt_dict    = {'ref_ch'  : np.zeros(nsamples, dtype=int),
                  'coinc_ch': np.random.randint(512, 700, nsamples),
                  'corr_dt' : np.random.normal(dist_mean, dist_sigma, nsamples)}
    return dist_mean, dist_sigma, pd.DataFrame(dt_dict)


@mark.filterwarnings("ignore:Covariance")
@mark.filterwarnings("ignore:divide by zero")
def test_peak_position(deltat_df):
    hist_bins   = np.linspace(-10000, 10000, 400, endpoint=False)
    all_ids     = np.arange(768)
    bias_finder = peak_position(hist_bins, 100, pd.Series(np.zeros_like(all_ids), index=all_ids))

    exp_mu, exp_sig, dt_df = deltat_df
    biases = dt_df.groupby('ref_ch', group_keys=False).apply(bias_finder)

    assert biases.shape      == (1,)
    assert biases.index.name == 'ref_ch'
    assert biases.index[0]   == 0
    ## Very weak requirement. Might have to consider hypothesis
    np.testing.assert_allclose(biases, exp_mu, atol=exp_sig)


@mark.filterwarnings("ignore:Imported map")
def test_process_raw_data(TEST_DATA_DIR, TMP_OUT):
    test_file   = os.path.join(TEST_DATA_DIR                              ,
                               '20221121_SourceSM1pos8_1000evt_coinc.ldat')
    test_conf   = os.path.join(TMP_OUT      , 'skew.conf')
    mapfile     = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    source_file = os.path.join(TEST_DATA_DIR, 'twoSM_skewSource_pos.yaml')

    with open(test_conf, 'w') as conf:
        conf.write(f'[mapping]\nmap_file = {mapfile}\nsetup = pointSource\n')
        conf.write('SM_NO_CORR = -1\n')
        conf.write(f'source_pos = {source_file}\n')
        conf.write( '[filter]\nmin_channels = 4\nelimits = 5,15\n')
        conf.write(f'[output]\nout_dir = {TMP_OUT}')
    
    conf = configparser.ConfigParser()
    conf.read(test_conf)
    ch_map = ChannelMap(mapfile)

    dt_files = process_raw_data([test_file], conf, ch_map)
    assert len(dt_files) == 1
    assert os.path.isfile(dt_files[0])

    saved_dt = pd.read_feather(dt_files[0])
    exp_cols = ['ref_ch', 'coinc_ch', 'corr_dt']
    assert all(hasattr(saved_dt, col) for col in exp_cols)
    assert saved_dt.shape == (183, len(exp_cols))

    map_df  = ch_map.mapping
    ref_chs = map_df[(map_df.supermodule == 0) & (map_df.minimodule == 14)].index
    coi_chs = map_df[ map_df.supermodule == 2 ].index
    assert all(saved_dt.ref_ch  .isin(ref_chs))
    assert all(saved_dt.coinc_ch.isin(coi_chs))
    assert saved_dt.corr_dt.mean() == approx(750.84)


@mark.filterwarnings("ignore:Covariance")
@mark.filterwarnings("ignore:divide by zero")
def test_calculate_skews_nobias(TEST_DATA_DIR, TMP_OUT):
    test_file = os.path.join(TEST_DATA_DIR                                 ,
                             '20221121_SourceSM1pos8_1000evt_coinc.feather')
    test_conf = os.path.join(TMP_OUT      , 'skew.conf')

    with open(test_conf, 'w') as conf:
        conf.write('[filter]\nrelax_fact = 0.7\nmin_stats=10\n')
        conf.write('hist_bins = -5000,5000,800\n')

    start_bias = pd.Series(0, index=np.arange(768))
    conf       = configparser.ConfigParser()
    conf.read(test_conf)
    skews      = calculate_skews([test_file], conf, start_bias)
    ref_chs    = pd.read_feather(test_file).ref_ch.unique()
    assert skews[skews != 0].shape[0] == ref_chs.shape[0]
    assert all(skews[skews != 0].index.isin(ref_chs))
    # Too much hardwiring as normal!
    exp_bias = np.array([326.666667, 364.000000, 482.222222, 681.333333,
                         622.758621, 513.333333,  370.588235, 580.000000])
    np.testing.assert_allclose(skews[skews != 0], exp_bias)


@mark.filterwarnings("ignore:Imported map")
def test_geom_loc_point(TEST_DATA_DIR):
    test_file   = os.path.join(TEST_DATA_DIR                              ,
                               '20221121_SourceSM1pos8_1000evt_coinc.ldat')
    map_file    = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    source_file = os.path.join(TEST_DATA_DIR, 'twoSM_skewSource_pos.yaml')

    ch_map = ChannelMap(map_file)
    with open(source_file) as sfile:
        source_yml = yaml.safe_load(sfile)
    chan_list, find_indx, geom_dt = geom_loc_point(test_file, ch_map, source_yml)

    exp_chans = (245, 243, 242, 233, 240, 238, 241, 239,
                 253, 250, 251, 248, 249, 246, 247, 244)
    assert len(chan_list) == 16
    assert np.isin(chan_list, exp_chans).all()
    assert isinstance(find_indx, FunctionType)
    assert isinstance(geom_dt  , FunctionType)
    assert find_indx([[250, 0, 0, 0], [600, 0, 0, 0]]) == 0
    assert geom_dt(250, 600) == approx(-314.093)


@mark.xfail
@mark.filterwarnings("ignore:Imported map")
def test_geom_loc_bar(TEST_DATA_DIR):
    test_name   = os.path.join(TEST_DATA_DIR, 'fake_SourcePos5_fake.ldat')
    map_file    = os.path.join(TEST_DATA_DIR, 'ringblah.feather')
    source_file = os.path.join(TEST_DATA_DIR, 'barBlah.yaml')

    ch_map = ChannelMap(map_file)
    with open(source_file) as sfile:
        source_yml = yaml.safe_load(sfile)
    ids, find_indx, geom_dt = geom_loc_bar(test_name, ch_map, source_yml)
