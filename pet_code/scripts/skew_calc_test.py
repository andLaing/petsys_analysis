from pytest import fixture

from . skew_calc import np
from . skew_calc import pd
from . skew_calc import peak_position


@fixture(scope = 'module')
def deltat_df():
    nsamples   = 10000
    dist_mean  =   100
    dist_sigma =    10
    dt_dict    = {'ref_ch'  : np.zeros(nsamples, dtype=int),
                  'coinc_ch': np.random.randint(512, 700, nsamples),
                  'corr_dt' : np.random.normal(dist_mean, dist_sigma, nsamples)}
    return dist_mean, dist_sigma, pd.DataFrame(dt_dict)


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
