import os
import configparser

from pytest import approx, fixture, mark

from . skew_calc import np
from . skew_calc import pd
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
    test_file = os.path.join(TEST_DATA_DIR                              ,
                             '20221121_SourceSM1pos8_1000evt_coinc.ldat')
    test_conf = os.path.join(TMP_OUT      , 'skew.conf')
    mapfile   = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')

    with open(test_conf, 'w') as conf:
        conf.write(f'[mapping]\nmap_file = {mapfile}\nsetup = 2SM\n')
        conf.write( '[filter]\nmin_channels = 4,4\nelimits = 5,15\n')
        conf.write(f'[output]\nout_dir = {TMP_OUT}')
    
    conf = configparser.ConfigParser()
    conf.read(test_conf)
    proc_file = process_raw_data(conf)

    dt_files = list(proc_file([test_file]))
    assert len(dt_files) == 1
    assert os.path.isfile(dt_files[0])

    saved_dt = pd.read_feather(dt_files[0])
    exp_cols = ['ref_ch', 'coinc_ch', 'corr_dt']
    assert all(hasattr(saved_dt, col) for col in exp_cols)
    assert saved_dt.shape == (183, len(exp_cols))

    map_df  = pd.read_feather(mapfile)
    ref_chs = map_df[(map_df.supermodule == 0) & (map_df.minimodule == 15)].id
    coi_chs = map_df[ map_df.supermodule == 2 ].id
    assert all(saved_dt.ref_ch  .isin(ref_chs))
    assert all(saved_dt.coinc_ch.isin(coi_chs))
    assert saved_dt.corr_dt.mean() == approx(750.84)
