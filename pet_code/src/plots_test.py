from types import FunctionType

import matplotlib.pyplot as plt

from . plots import hist1d
from . plots import mm_energy_spectra
from . util  import np

def test_hist1d():
    tdata = np.random.normal(150, size=1000)
    bin_edges, bin_vals = hist1d(plt, tdata)

    np.testing.assert_allclose(bin_edges, np.linspace(0, 300, 201))
    assert np.sum(bin_vals) == 1000


def test_mm_energy_spectra_noplots():
    gen_stats = 10000
    mm_e = {i: {'energy': np.random.normal(100 + 2 * i, 5 + i, gen_stats)} for i in range(1, 17)}

    exclusions = mm_energy_spectra(mm_e, 0)

    assert isinstance(exclusions, list)
    assert all(isinstance(ex, FunctionType) for ex in exclusions)
    ## Check that we have approximately +i 2 sigma passing cuts.
    accepted_prop = np.fromiter((len(v['energy'][exclusions[i-1](v['energy'])]) / gen_stats for i, v in mm_e.items()), float)
    assert all(accepted_prop > 0.9)
