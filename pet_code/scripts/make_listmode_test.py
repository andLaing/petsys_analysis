import os
import configparser

from copy      import deepcopy
from itertools import chain
from pytest    import mark

from .. src.io   import ChannelMap
from .. src.io   import LMHeader
from .. src.util import ChannelType
from .. src.util import convert_to_kev

from .. src.util_test import enum_dummy

from . make_listmode import np
from . make_listmode import equal_and_select
from . make_listmode import local_pixel
from . make_listmode import supermod_energy
from . make_listmode import write_header


def test_local_pixel():
    xbins = np.linspace(0, 100, 101)
    ybins = np.linspace(0, 100, 101)
    xypos = ((50, 50), (1.3, 1.3), (1.3, 97.3), (97.3, 1.3), (97.3, 97.3))

    pixel_finder = local_pixel(xbins, ybins)

    found_pix    = tuple(map(lambda xy: pixel_finder(*xy), xypos))
    exp_pix      = ((50, 50), (1, 1), (1, 97), (97, 1), (97, 97))
    assert all(fnd == exp for fnd, exp in zip(found_pix, exp_pix))


@mark.filterwarnings("ignore:Imported map")
def test_equal_and_select(TEST_DATA_DIR, TMP_OUT, DUMMY_EVT):
    map_file = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    tchan_fn = os.path.join(TMP_OUT      ,     'Teng_peaks.tsv'    )
    echan_fn = os.path.join(TMP_OUT      ,     'Eeng_peaks.tsv'    )

    chan_map = ChannelMap(map_file)
    with open(tchan_fn, 'w') as tfile, open(echan_fn, 'w') as efile:
        tfile.write('ID\tMU\n')
        efile.write('ID\tMU\n')
        for imp in chain(*DUMMY_EVT):
            if chan_map.get_channel_type(imp[0]) is ChannelType.TIME:
                tfile.write(f'{imp[0]}\t10\n')
            else:
                efile.write(f'{imp[0]}\t10\n')

    # to avoid changing DUMMY_EVT in memory
    test_evt = deepcopy(DUMMY_EVT)
    cal_sel  = equal_and_select(chan_map, tchan_fn, echan_fn)

    sel_evt = cal_sel(test_evt)
    assert len(sel_evt[0]) <  len(DUMMY_EVT[0])
    assert len(sel_evt[0]) == 10
    assert len(sel_evt[1]) == len(DUMMY_EVT[1])
    ## Need to test values too!!


def test_write_header(TMP_OUT):
    tmp_conf = os.path.join(TMP_OUT, 'listmode.conf')
    tmp_LM   = os.path.join(TMP_OUT, 'LMheader.bin')

    ident = 'IMAS_1ring'
    acq   =  60
    iso   = 'Na22'
    detX  = 103.22
    detY  = 103.22
    modNo =  24
    rNo   =   1
    ringD = 820
    with open(tmp_conf, 'w') as conf:
        conf.write(f'[header]\n\
                     identifier = {ident}\n\
                     acqTime = {acq}\n\
                     isotope = {iso}\n\
                     detectorSizeX = {detX}\n\
                     detectorSizeY = {detY}\n\
                     moduleNumber = {modNo}\n\
                     ringNumber = {rNo}\n\
                     ringDistance = {ringD}')

    conf = configparser.ConfigParser()
    conf.read(tmp_conf)
    with open(tmp_LM, 'wb') as lm:
        write_header(lm, conf, np.linspace(0, 100, 101), np.linspace(0, 100, 101))

    assert os.path.isfile(tmp_LM)
    LMHead = LMHeader()
    with open(tmp_LM, 'rb') as lm:
        lm.readinto(LMHead)
    assert LMHead.identifier    == ident.encode('utf-8')
    assert LMHead.acqTime       == acq
    assert LMHead.isotope       == iso  .encode('utf-8')
    assert LMHead.detectorSizeX == detX
    assert LMHead.detectorSizeY == detY
    assert LMHead.moduleNumber  == modNo
    assert LMHead.ringNumber    == rNo
    assert LMHead.ringDistance  == ringD


@mark.filterwarnings("ignore:Imported map")
def test_supermod_energy(TEST_DATA_DIR, DUMMY_SM):
    map_file = os.path.join(TEST_DATA_DIR, '1ring_map.feather')
    kev_file = os.path.join(TEST_DATA_DIR, 'mM_calibratedEng_Peak.tsv')

    chan_map = ChannelMap(map_file)
    kev_conv = convert_to_kev(kev_file, chan_map.get_modules)

    kev_eng = supermod_energy(kev_conv)
    mm_eng  = kev_eng(enum_dummy(DUMMY_SM))

    assert np.isclose(round(mm_eng, 3), 599.570)
