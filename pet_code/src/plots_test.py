import os

from pytest    import mark
from functools import partial
from types     import FunctionType

import matplotlib.pyplot as plt

from . io    import read_ymlmapping
from . util  import np
from . util  import ChannelType

from . plots import ChannelEHistograms
from . plots import hist1d
from . plots import group_times
from . plots import group_times_list
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


@mark.filterwarnings("ignore::DeprecationWarning")
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


@mark.filterwarnings("ignore::DeprecationWarning")
def test_slab_energy_spectra_plots(TMP_OUT):
    nhists    =     5
    gen_stats = 10000
    slab_eng  = {i: {'energy': np.random.normal(18, 1, gen_stats)} for i in range(nhists)}

    out_base = os.path.join(TMP_OUT, 'testplots.ldat')
    _        = slab_energy_spectra(slab_eng, plot_output=out_base, min_peak=100)

    plot_name = out_base.replace('.ldat', '_slab{}Spec.png')
    assert all(os.path.isfile(plot_name.format(k)) for k in slab_eng.keys())


def enum_channels(SM):
    return list(map(lambda imp: [imp[0], ChannelType[imp[1]], imp[2], imp[3]], SM))


def test_group_times(TEST_DATA_DIR, DUMMY_EVT):
    test_yml         = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, _, mm_map, *_ = read_ymlmapping(test_yml)

    dummy_with_enum = tuple(map(enum_channels, DUMMY_EVT))
    # Dummy peak filters
    peak_sel = [[lambda x: (x >  50) & (x <  60) for _ in range(10)],
                [lambda x: (x > 100) & (x < 110) for _ in range(10)]]
    
    reco_dt  = group_times([dummy_with_enum], peak_sel, lambda id: mm_map[id], 1)

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


def test_group_times_slab(DUMMY_EVT):
    dummy_with_enum = tuple(map(enum_channels, DUMMY_EVT))

    # Dummy peak filters.
    peak_sel = [{682: lambda x: (x >  5) & (x <  7),
                 640: lambda x: (x >  5) & (x <  7)},
                { 64: lambda x: (x > 12) & (x < 14),
                  65: lambda x: (x > 12) & (x < 14)}]

    reco_dt = group_times_slab([dummy_with_enum], peak_sel, 1)

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


def test_group_times_list(DUMMY_EVT):
    dummy_with_enum = tuple(map(enum_channels, DUMMY_EVT))

    # Dummy peak filters.
    peak_sel = [{682: lambda x: (x >  5) & (x <  7),
                 640: lambda x: (x >  5) & (x <  7)},
                { 64: lambda x: (x > 12) & (x < 14),
                  65: lambda x: (x > 12) & (x < 14)}]

    reco_dt = group_times_list([dummy_with_enum], peak_sel, 1)

    assert len(reco_dt) == 1
    times0 = reco_dt[0]
    assert len( times0) == 4
    assert times0[0] ==  64
    assert times0[1] == 682
    assert times0[3] - times0[2] == -15


def test_ChannelEHistograms_thist(TEST_DATA_DIR, DUMMY_EVT):
    test_yml    = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    time_ch, *_ = read_ymlmapping(test_yml)

    trange   = 2,  24, 0.1
    tbins    = np.arange(*trange)
    e_histos = ChannelEHistograms(tbins, np.arange(4, 24, 0.2), np.arange(0, 200, 1.5))

    enum_evt = tuple(map(enum_channels, DUMMY_EVT))
    tchans   = list(filter(lambda x: x[0] in time_ch, enum_evt[0] + enum_evt[1]))

    # Expected over and underflow
    tunder = [imp[0] for imp in filter(lambda x: x[3] <  trange[0]            , tchans)]
    tover  = [imp[0] for imp in filter(lambda x: x[3] >= trange[1] - trange[2], tchans)]

    # numpy histogram
    thist = partial(np.histogram, bins=tbins)

    for imp in tchans:
        e_histos.fill_time_channel(imp)

    ntchan   = np.unique([imp[0] for imp in tchans]).shape[0]
    ch_under = len(e_histos.underflow)
    assert np.unique(tunder).shape[0] == ch_under
    assert set(tunder).issubset(e_histos.underflow.keys())
    assert len(tunder) ==   sum(e_histos.underflow.values())
    ch_over  = len(e_histos.overflow)
    assert np.unique(tover).shape[0] == ch_over
    assert set(tover).issubset(e_histos.overflow.keys())
    assert len(tover)  ==  sum(e_histos.overflow.values())
    assert len(e_histos.tdist) == ntchan - ch_over - ch_under
    assert sum(e_histos.tdist.values()).sum() == len(tchans) - ch_over - ch_under
    for id, hist in e_histos.tdist.items():
        allE_id  = list(map(lambda x: x[3], filter(lambda y: y[0] == id, tchans)))
        exp_hist = thist(allE_id)[0]
        np.testing.assert_array_equal(exp_hist, hist)


def test_ChannelEHistograms_ehist(TEST_DATA_DIR, DUMMY_EVT):
    test_yml      = os.path.join(TEST_DATA_DIR, "SM_mapping.yaml")
    _, eng_ch, *_ = read_ymlmapping(test_yml)

    erange   = 4,  24, 0.2
    ebins    = np.arange(*erange)
    e_histos = ChannelEHistograms(np.arange(2, 24, 0.1), ebins, np.arange(0, 200, 1.5))

    enum_evt = tuple(map(enum_channels, DUMMY_EVT))
    echans   = list(filter(lambda x: x[0] in  eng_ch, enum_evt[0] + enum_evt[1]))
    eunder   = [imp[0] for imp in filter(lambda x: x[3] <  erange[0]            , echans)]
    eover    = [imp[0] for imp in filter(lambda x: x[3] >= erange[1] - erange[2], echans)]
    ehist    = partial(np.histogram, bins=ebins)
    
    for imp in echans:
        e_histos.fill_energy_channel(imp)
    
    nechan   = np.unique([imp[0] for imp in echans]).shape[0]
    ch_under = len(e_histos.underflow)
    assert np.unique(eunder).shape[0] == ch_under
    assert set(eunder).issubset(e_histos.underflow.keys())
    assert len(eunder) ==   sum(e_histos.underflow.values())
    ch_over  = len(e_histos.overflow)
    assert np.unique(eover).shape[0] == ch_over
    assert set(eover).issubset(e_histos.overflow.keys())
    assert len(eover)  ==  sum(e_histos.overflow.values())
    assert len(e_histos.edist) == nechan - ch_over - ch_under
    assert sum(e_histos.edist.values()).sum() == len(echans) - ch_over - ch_under
    for id, hist in e_histos.edist.items():
        allE_id  = list(map(lambda x: x[3], filter(lambda y: y[0] == id, echans)))
        exp_hist = ehist(allE_id)[0]
        np.testing.assert_array_equal(exp_hist, hist)


def test_ChannelEHistograms_sumhist(DUMMY_EVT):
    sumbins  = np.arange(0, 200, 1.5)
    e_histos = ChannelEHistograms(np.arange(2, 24, 0.1), np.arange(4, 24, 0.2), sumbins)
    # Energy sum
    exp_id = 1008 # Def need a bigger number than this for full setup!
    enum_SM = enum_channels(DUMMY_EVT[1])
    e_histos.fill_esum(enum_SM, exp_id)

    eng_filt = filter(lambda y: y[1] is ChannelType.ENERGY, enum_SM)
    exp_sum  = sum(map(lambda x: x[3], eng_filt))
    assert len(e_histos.sum_dist) == 1
    assert exp_id     in e_histos.sum_dist .keys()
    assert exp_id not in e_histos.overflow .keys()
    assert exp_id not in e_histos.underflow.keys()
    np.testing.assert_array_equal(np.histogram(exp_sum, bins=sumbins)[0],
                                  e_histos.sum_dist[exp_id]             )


def test_ChannelEHistograms_maxes(DUMMY_EVT):
    e_histos = ChannelEHistograms(np.arange(2,  24, 0.1),
                                  np.arange(4,  24, 0.2),
                                  np.arange(0, 200, 1.5))

    enum_evt = tuple(map(enum_channels, DUMMY_EVT))
    e_histos.add_emax_evt(enum_evt)

    assert len(e_histos.tdist) == 2
    assert len(e_histos.edist) == 2
    exp_ids  =  64, 682
    exp_indx = 116,  44
    assert e_histos.tdist.keys() == set(exp_ids)
    for id, indx in zip(exp_ids, exp_indx):
        assert e_histos.tdist[id][indx] == 1
        mask       = np.ones(e_histos.tdist[id].size, bool)
        mask[indx] = False
        assert all(e_histos.tdist[id][mask] == 0)
    exp_ids  = 116, 700
    exp_indx =  80,  39
    assert e_histos.edist.keys() == set(exp_ids)
    for id, indx in zip(exp_ids, exp_indx):
        assert e_histos.edist[id][indx] == 1
        mask       = np.ones(e_histos.edist[id].size, bool)
        mask[indx] = False
        assert all(e_histos.edist[id][mask] == 0)
