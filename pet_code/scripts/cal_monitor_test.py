import os

from pytest import fixture, mark

from .. src.filters import filter_event_by_impacts
from .. src.plots   import ChannelEHistograms
from .. src.io      import ChannelMap
from .. src.io      import read_petsys_filebyfile
from .. src.util    import select_module

from . cal_monitor import np
from . cal_monitor import output_time_plots


@fixture(scope = 'module')
def test_plots(TEST_DATA_DIR):
    tbins   = np.arange(9,  25, 0.2)
    ebins   = np.arange(0, 300, 1.5)
    plotter = ChannelEHistograms(tbins, ebins, ebins)

    test_file = os.path.join(TEST_DATA_DIR, 'chanCal_Source.ldat'   )
    map_file  = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    ch_map    = ChannelMap(map_file)
    evt_filt  = filter_event_by_impacts(4, singles=True)
    reader    = read_petsys_filebyfile(ch_map.ch_type, sm_filter=evt_filt, singles=True)
    sel_mod   = select_module(ch_map.get_minimodule)
    for evt in reader(test_file):
        sel_mm = sel_mod(evt[0])
        sm_mm  = (ch_map.get_supermodule(sel_mm[0][0]),
                  ch_map.get_minimodule (sel_mm[0][0]))
        plotter.add_all_energies((sel_mm, []), sm_mm)

    ## Fill some values...
    return plotter    

@mark.skip
def test_output_time_plots(TMP_OUT, test_plots):
    cal_name = 'test'
    output_time_plots(test_plots, cal_name, TMP_OUT, 10)

