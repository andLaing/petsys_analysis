import os

from pytest import fixture, mark

from .. src.plots import ChannelEHistograms
from .. src.io    import ChannelMap

from . cal_monitor import np
from . cal_monitor import output_energy_plots
from . cal_monitor import output_time_plots


@fixture(scope = 'module')
def test_plots(TEST_DATA_DIR):
    tbins   = np.arange(9,  25, 0.2)
    ebins   = np.arange(0, 300, 1.5)
    plotter = ChannelEHistograms(tbins, ebins, ebins)

    map_file  = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    ch_map    = ChannelMap(map_file)
    typ       = np.vectorize(ch_map.get_channel_type)
    ## Random values for ids and energies
    nevt      = 100 * ch_map.mapping.shape[0]
    ids       = np.random.choice(ch_map.mapping.index, size=(nevt, 1))
    test_data = np.hstack((ids, typ(ids), np.full((ids.shape[0], 1), 0),
                           np.random.normal(15, 3, (nevt, 1))))
    for evt in test_data.tolist():
        sm_mm  = (ch_map.get_supermodule(evt[0]),
                  ch_map.get_minimodule (evt[0]))
        plotter.add_all_energies(([evt], []), [sm_mm])
    return plotter    

@mark.filterwarnings("ignore:Imported map")
@mark.filterwarnings("ignore:divide by zero")
@mark.filterwarnings("ignore:Covariance of")
def test_output_time_plots(TMP_OUT, test_plots):
    cal_name   = 'test'
    dummy_file = 'test_file.ldat'
    output_time_plots(test_plots, cal_name, TMP_OUT, dummy_file, 10)

    time_mu_file  = os.path.join(TMP_OUT, 'test_filetest_timeEngMu.png')
    assert os.path.isfile( time_mu_file)
    time_sig_file = os.path.join(TMP_OUT, 'test_filetest_timeEngSig.png')
    assert os.path.isfile(time_sig_file)
    time_all_file = os.path.join(TMP_OUT, 'test_filetest_timeAllDist.png')
    assert os.path.isfile(time_all_file)


@mark.filterwarnings("ignore:Imported map")
@mark.filterwarnings("ignore:divide by zero")
@mark.filterwarnings("ignore:Covariance of")
def test_output_energy_plots(TMP_OUT, test_plots):
    cal_name   = 'test'
    dummy_file = 'test_file.ldat'
    sm_nums    = (0, 2)
    output_energy_plots(test_plots, cal_name, TMP_OUT, dummy_file, 'tb', sm_nums)

    txt_file = os.path.join(TMP_OUT, 'test_filetest_MMEngPeaks.txt')
    assert os.path.isfile(txt_file)
    mm_files = os.path.join(TMP_OUT, 'test_filetest_MMEngs_sm{}.png')
    assert all(os.path.isfile(mm_files.format(i)) for i in sm_nums)
    eng_mu_file  = os.path.join(TMP_OUT, 'test_filetest_mmEngMu.png')
    assert os.path.isfile( eng_mu_file)
    eng_sig_file = os.path.join(TMP_OUT, 'test_filetest_mmEngSig.png')
    assert os.path.isfile(eng_sig_file)
    eng_all_file = os.path.join(TMP_OUT, 'test_filetest_engAllDist.png')
    assert os.path.isfile(eng_all_file)

