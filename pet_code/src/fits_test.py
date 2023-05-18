from pytest import raises

from . fits import fit_gaussian
from . fits import gaussian
from . fits import mean_around_max
from . util import np


def test_gaussian():
    mu  = 0.0
    sig = 5.0
    x   = np.linspace(-2000, 2000, 4000)
    g   = gaussian(x, 1.0, mu, sig)

    std = np.average((x - mu)**2, weights=g)
    np.testing.assert_almost_equal(np.average(x, weights=g), mu )
    np.testing.assert_almost_equal(np.sqrt(std)            , sig)
    np.testing.assert_almost_equal(np.sum(g)               , 1.0, decimal=3)


def test_mean_around_max():
    mu = 0
    bin_vals, bin_edges = np.histogram(np.random.normal(mu, 2, 10000),
                                       bins=np.arange(-50, 50, 0.2))
    edges = bin_edges[:-1]
    mean_val, wsum, x, y, _ = mean_around_max(bin_vals, edges, 6)

    max_indx = np.argmax(bin_vals)
    np.testing.assert_allclose(x, edges   [max_indx-6:max_indx+6])
    np.testing.assert_allclose(y, bin_vals[max_indx-6:max_indx+6])
    #np.testing.assert_almost_equal(mean_val, edges[max_indx])
    np.testing.assert_almost_equal(wsum, np.sum(bin_vals[max_indx-6:max_indx+6]))


def test_mean_around_max_fails_on_neg():
    x = np.linspace(0, 10, 20)
    y = np.full(x.shape, -1)
    ret_vals = mean_around_max(y, x, len(x) // 2)
    assert all((v is None for v in ret_vals))


def test_fit_gaussian():
    amp     = 10000
    mu      =     3
    sig     =     2
    bin_wid =     0.2
    bin_vals, bin_edges = np.histogram(np.random.normal(mu, sig, amp),
                                       bins=np.arange(-50, 50, bin_wid))
    bin_centres, gvals, pars, pcov, chi_ndf = fit_gaussian(bin_vals, bin_edges)

    np.testing.assert_allclose(bin_centres, bin_edges[:-1] + bin_wid / 2)
    rel_errors = np.sqrt(np.diag(pcov)) / pars
    assert np.isclose(np.sum(gvals ), amp, rtol=3 * rel_errors[0])
    assert np.isclose(       pars[1], mu , rtol=3 * rel_errors[1])
    assert np.isclose(       pars[2], sig, rtol=3 * rel_errors[2])
    assert chi_ndf < 5


def test_fit_gaussian_raises():
    amp     = 100
    mu      =   3
    sig     =   2
    bin_wid =   0.2
    bin_vals, bin_edges = np.histogram(np.random.normal(mu, sig, amp),
                                       bins=np.arange(-50, 50, bin_wid))

    with raises(RuntimeError):
        dummy = fit_gaussian(bin_vals, bin_edges)


def test_fit_gaussian_peakfinder():
    amp     = 10000
    mu      =     3
    sig     =     2
    bin_wid =     0.2
    g_shift =    20
    bin_vals, bin_edges = np.histogram((np.random.normal(mu          , sig, amp),
                                        np.random.normal(mu + g_shift, sig, amp)),
                                       bins=np.arange(-50, 50, bin_wid)          )
    _, gvals, pars, pcov, chi_ndf = fit_gaussian(bin_vals, bin_edges, pk_finder='peak')

    rel_errors = np.sqrt(np.diag(pcov)) / pars
    assert np.isclose(np.sum(gvals ), amp         , rtol=3 * rel_errors[0])
    assert np.isclose(       pars[1], mu + g_shift, rtol=3 * rel_errors[1])
    assert np.isclose(       pars[2], sig         , rtol=3 * rel_errors[2])
    assert chi_ndf < 5
