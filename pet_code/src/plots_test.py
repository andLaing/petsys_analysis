import os

from types import FunctionType

import matplotlib.pyplot as plt

from . io    import read_ymlmapping
from . util  import np

from . plots import hist1d
from . plots import group_times
from . plots import group_times_slab
from . plots import mm_energy_spectra
from . plots import slab_energy_spectra


def test_hist1d():
    tdata = np.random.normal(150, size=1000)
    bin_edges, bin_vals = hist1d(plt, tdata)

    np.testing.assert_allclose(bin_edges, np.linspace(0, 300, 201))
    assert np.sum(bin_vals) == 1000


def test_mm_energy_spectra_noplots():
    gen_stats = 10000
    mm_e = {i: {'energy': np.random.normal(100 + 2 * i, 5, gen_stats)} for i in range(1, 17)}

    exclusions = mm_energy_spectra(mm_e, 0, min_peak=100)

    assert isinstance(exclusions, list)
    assert len(exclusions) == 16
    assert all(isinstance(ex, FunctionType) for ex in exclusions)
    ## Check that we have approximately +- 3 sigma passing cuts.
    accepted_prop = np.fromiter((len(v['energy'][exclusions[i-1](v['energy'])]) / gen_stats for i, v in mm_e.items()), float)
    assert all(accepted_prop > 0.9)


def test_mm_energy_spectra_plots(TMP_OUT):
    gen_stats = 10000
    mm_pitch  =    25.9
    mm_e = {i: {'energy': np.random.normal(100 + 2 * i, 5, gen_stats),
                'x'     : np.random.uniform(mm_pitch *  (i - 1) % 4      ,
                                            mm_pitch * ((i - 1) % 4 + 1) ,
                                            gen_stats                    ),
                'y'     : np.random.uniform(mm_pitch * (3 - (i - 1) // 4),
                                            mm_pitch * (4 - (i - 1) // 4),
                                            gen_stats                    )}
            for i in range(1, 17)                                          }

    out_base = os.path.join(TMP_OUT, 'testplots.ldat')
    _        = mm_energy_spectra(mm_e, 0, min_peak=100, plot_output=out_base)

    # mM energy spectra plots created?
    spec_plot  = out_base.replace('.ldat', '_EnergyModuleSMod0.png')
    flood_plot = out_base.replace(".ldat",      '_FloodModule0.png')
    assert os.path.isfile( spec_plot)
    assert os.path.isfile(flood_plot)


def test_slab_energy_spectra_noplots():
    nhists    =     5
    gen_stats = 10000
    slab_eng = {i: {'energy': np.random.normal(18, 1, gen_stats)} for i in range(nhists)}

    excl = slab_energy_spectra(slab_eng, min_peak=100)

    assert isinstance(excl, dict)
    assert len(excl) == nhists
    assert all(k in slab_eng.keys() for k in excl.keys())
    acc_prop = np.fromiter((len(v['energy'][excl[k](v['energy'])]) / gen_stats for k, v in slab_eng.items()), float)
    assert all(acc_prop > 0.9)


def test_slab_energy_spectra_plots(TMP_OUT):
    nhists    =     5
    gen_stats = 10000
    slab_eng  = {i: {'energy': np.random.normal(18, 1, gen_stats)} for i in range(nhists)}

    out_base = os.path.join(TMP_OUT, 'testplots.ldat')
    _        = slab_energy_spectra(slab_eng, plot_output=out_base, min_peak=100)

    plot_name = out_base.replace('.ldat', '_slab{}Spec.png')
    assert all(os.path.isfile(plot_name.format(k)) for k in slab_eng.keys())


def test_group_times(TEST_DATA_DIR, DUMMY_EVT):
    test_yml            = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    time_ch, eng_ch, *_ = read_ymlmapping(test_yml)

    # Dummy peak filters
    peak_sel = [[lambda x: (x >  50) & (x <  60) for _ in range(10)],
                [lambda x: (x > 100) & (x < 110) for _ in range(10)]]
    
    reco_dt  = group_times([DUMMY_EVT], peak_sel, eng_ch, time_ch, 1)

    assert len(reco_dt) == 1
    k, times = next(iter(reco_dt.items()))
    assert k == 64
    sm0_ch = set(x[0] for x in DUMMY_EVT[0])
    sm1_ch = set(x[0] for x in DUMMY_EVT[1])
    assert k in sm1_ch
    assert len(times) == 1
    assert isinstance(times[0], list)
    assert len(times[0]) == 3
    assert times[0][0] in sm0_ch
    assert times[0][2] - times[0][1] == -15


def test_group_times_slab(TEST_DATA_DIR, DUMMY_EVT):
    test_yml    = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    time_ch, *_ = read_ymlmapping(test_yml)

    # Dummy peak filters.
    peak_sel = [{682: lambda x: (x >  5) & (x <  7),
                 640: lambda x: (x >  5) & (x <  7)},
                { 64: lambda x: (x > 12) & (x < 14),
                  65: lambda x: (x > 12) & (x < 14)}]

    reco_dt = group_times_slab([DUMMY_EVT], peak_sel, time_ch, 1)

    assert len(reco_dt) == 1
    k, times = next(iter(reco_dt.items()))
    assert k == 64
    sm0_ch = set(x[0] for x in DUMMY_EVT[0])
    sm1_ch = set(x[0] for x in DUMMY_EVT[1])
    assert k in sm1_ch
    assert len(times) == 1
    assert isinstance(times[0], list)
    assert len(times[0]) == 3
    assert times[0][0] in sm0_ch
    assert times[0][2] - times[0][1] == -15
