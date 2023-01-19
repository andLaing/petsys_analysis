import os
import configparser

import numpy as np

from pytest import mark

from .. src.util import ChannelType

from . grp_channel_specs import channel_plots


@mark.filterwarnings("ignore:Imported map")
def test_channel_plots(TEST_DATA_DIR, TMP_OUT):
    inSource = os.path.join(TEST_DATA_DIR, 'chanCal_Source.ldat')
    inBack   = os.path.join(TEST_DATA_DIR, 'chanCal_woSource.ldat')
    mapfile  = os.path.join(TEST_DATA_DIR, 'twoSM_IMAS_map.feather')
    config   = os.path.join(TMP_OUT      , 'calib.conf')

    with open(config, 'w') as conf:
        conf.write('[filter]\nmin_channels = 4,4\nmin_stats = 200\n')
        conf.write(f'[mapping]\nmap_file = {mapfile}\n')
        conf.write('[output]\nesum_binning = 0,200,1.5\ntbinning = 2,24,0.1\nebinning = 2,24,0.2')

    conf = configparser.ConfigParser()
    conf.read(config)
    plotS, plotNS = channel_plots(conf, [inSource, inBack])

    assert len(plotS .tdist) == 255
    assert len(plotNS.tdist) == 246
    assert len(plotS .edist) == 252
    assert len(plotNS.edist) == 207
    # assert len(plotS .sum_dist) == 255
    # assert len(plotNS.sum_dist) == 246
    nval_Sslab   = sum(plotS .tdist.values()).sum()
    nval_woSslab = sum(plotNS.tdist.values()).sum()
    nval_Semax   = sum(plotS .edist.values()).sum()
    nval_woSemax = sum(plotNS.edist.values()).sum()
    # nval_Sesum   = sum(plotS .sum_dist.values()).sum()
    # nval_woSesum = sum(plotNS.sum_dist.values()).sum()
    assert nval_Sslab   == 8792
    # assert nval_Sesum   == 8820
    assert nval_Semax   == 8742
    assert nval_woSslab ==  829
    # assert nval_woSesum ==  831
    assert nval_woSemax ==  824
