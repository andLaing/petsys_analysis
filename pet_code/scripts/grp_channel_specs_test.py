import os
import configparser

from pytest import fixture, mark

from .. src.plots import ChannelEHistograms
from .. src.util  import ChannelType
from .. src.util  import get_absolute_id
from .. src.util  import pd

from . grp_channel_specs import channel_plots
from . grp_channel_specs import np
from . grp_channel_specs import energy_plots
from . grp_channel_specs import petsys_file
from . grp_channel_specs import slab_plots


@fixture(scope = 'module')
def cal_conf(TEST_DATA_DIR, TMP_OUT):
    config  = os.path.join(TMP_OUT, 'calib.conf')
    mapfile = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')

    with open(config, 'w') as conf:
        conf.write('[filter]\nmin_channels = 4\nmin_stats = 200\n')
        conf.write(f'[mapping]\nmap_file = {mapfile}\n')
        conf.write('[output]\nesum_binning = 0,200,1.5\ntbinning = 2,24,0.1\nebinning = 2,24,0.2')

    return config


@mark.filterwarnings("ignore:Imported map")
def test_channel_plots(TEST_DATA_DIR, cal_conf):
    inSource = os.path.join(TEST_DATA_DIR, 'chanCal_Source.ldat')
    inBack   = os.path.join(TEST_DATA_DIR, 'chanCal_woSource.ldat')

    conf = configparser.ConfigParser()
    conf.read(cal_conf)
    plotS, plotNS = channel_plots(conf, [inSource, inBack])

    assert len(plotS .tdist) == 255
    assert len(plotNS.tdist) == 245
    assert len(plotS .edist) == 252
    assert len(plotNS.edist) == 206
    nval_Sslab   = sum(plotS .tdist.values()).sum()
    nval_woSslab = sum(plotNS.tdist.values()).sum()
    nval_Semax   = sum(plotS .edist.values()).sum()
    nval_woSemax = sum(plotNS.edist.values()).sum()
    assert nval_Sslab   == 8807
    assert nval_woSslab ==  830
    assert nval_Semax   == 8756
    assert nval_woSemax ==  825


@fixture(scope = 'module')
def gauss_plots():
    tbins = np.arange(0, 50)
    ebins = np.arange(0, 50)
    sbins = np.arange(0, 10)

    plot_source  = ChannelEHistograms(tbins, ebins, sbins)
    plot_nsource = ChannelEHistograms(tbins, ebins, sbins)

    ## Fill source plots with 10000 Gaussian distributed values
    ## for 5 channels
    size_col  = (10000 * 5, 1)
    tmu, tsig = 26, 5
    id_gen    = (np.full((10000, 1), x, dtype=int) for x in range(5))
    dummy_t   = np.hstack((np.vstack(tuple(id_gen))             ,
                           np.full  (size_col, ChannelType.TIME),
                           np.zeros (size_col, int             ),
                           np.random.normal(tmu, tsig, size_col)))
    id_gen    = (np.full((10000, 1), x, dtype=int) for x in range(5, 10))
    emu, esig = 23, 5
    dummy_e   = np.hstack((np.vstack(tuple(id_gen))               ,
                           np.full  (size_col, ChannelType.ENERGY),
                           np.zeros (size_col, int               ),
                           np.random.normal(emu, esig, size_col)  ))
    for i, (impt, impe) in enumerate(zip(dummy_t, dummy_e)):
        plot_source.fill_time_channel  (impt)
        plot_source.fill_energy_channel(impe)
        if i % 10000 < 100:
            plot_nsource.fill_time_channel  (impt)
            plot_nsource.fill_energy_channel(impe)
    return tmu, tsig, emu, esig, plot_source, plot_nsource


def test_slab_plots(TMP_OUT, gauss_plots):
    out_file = os.path.join(TMP_OUT, 'tests_calplot_')

    tmu, tsig, _, _, plot_source, plot_nsource = gauss_plots

    bad_fits, _ = slab_plots(out_file, plot_source, plot_nsource, min_stats=100)

    assert bad_fits == 0
    time_fits = out_file + 'timeSlabPeaks.txt'
    assert os.path.isfile(time_fits)
    time_fit_vals = pd.read_csv(time_fits, sep='\t')
    assert time_fit_vals.shape == (5, 5)
    assert time_fit_vals.columns.isin(['ID', 'MU', 'MU_ERR', 'SIG', 'SIG_ERR']).all()
    np.testing.assert_allclose(time_fit_vals.MU ,  tmu, rtol=0.05)
    np.testing.assert_allclose(time_fit_vals.SIG, tsig, rtol=0.05)


def test_energy_plots(TMP_OUT, gauss_plots):
    out_file = os.path.join(TMP_OUT, 'tests_calplot_')

    _, _, Emu, _, plot_source, plot_nsource = gauss_plots

    bad_fits, _ = energy_plots(out_file, plot_source, plot_nsource, 100)

    assert bad_fits == 0
    eng_fits = out_file + 'eChannelPeaks.txt'
    assert os.path.isfile(eng_fits)
    eng_fit_vals = pd.read_csv(eng_fits, sep='\t')
    assert eng_fit_vals.shape == (5, 3)
    assert eng_fit_vals.columns.isin(['ID', 'MU', 'MU_ERR']).all()
    np.testing.assert_allclose(eng_fit_vals.MU, Emu, rtol=0.1)


def test_petsys_file(TMP_OUT, TEST_DATA_DIR):
    map_file = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    tchans   = os.path.join(TMP_OUT      ,    'fake_tPeaks.txt'    )
    echans   = os.path.join(TMP_OUT      ,    'fake_ePeaks.txt'    )
    eref     = 10
    out_name = os.path.join(TMP_OUT      ,    'test_petsys.tsv'    )

    map_type = pd.read_feather(map_file)[['id', 'type']]
    with open(tchans, 'w') as tpeak, open(echans, 'w') as epeak:
        tpeak.write('ID\tMU\tMU_ERR\tSIG\tSIG_ERR\n')
        epeak.write('ID\tMU\tMU_ERR\n')
        for _, id_typ in map_type.iterrows():
            if id_typ.type == 'TIME':
                tpeak.write(f'{id_typ.id}\t10\t1\t2\t0.5\n')
            else:
                epeak.write(f'{id_typ.id}\t10\t1\n')
    petsys_file(map_file, tchans, echans, eref, out_name)

    assert os.path.isfile(out_name)
    petsys_df = pd.read_csv(out_name, sep='\t')
    cols = ['#portID', 'slaveID', 'chipID', 'channelID', 'tacID', 'p0', 'p1', 'p2', 'p3']
    assert petsys_df.columns.isin(cols).all()
    assert petsys_df.shape == (map_type.shape[0] * 4, len(cols))
    petsys_df['id'] = petsys_df[cols[:4]].apply(lambda r: get_absolute_id(*r), axis=1)
    petsys_df.drop_duplicates('id', inplace=True)
    assert np.count_nonzero(petsys_df.isna()) == 0
    np.testing.assert_allclose(petsys_df[petsys_df.id.isin(map_type[map_type.type ==   'TIME'].id)].p3, 511 / 10)
    np.testing.assert_allclose(petsys_df[petsys_df.id.isin(map_type[map_type.type == 'ENERGY'].id)].p3,      1.0)
