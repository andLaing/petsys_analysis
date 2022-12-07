# Just to sort some values to test in configurations.

import sys
import configparser

import pandas as pd
import numpy  as np

import matplotlib.pyplot as plt

from pet_code.src.io   import read_ymlmapping
from pet_code.src.util import get_electronics_nums


if __name__ == '__main__':
    conf_name      = sys.argv[1]
    config         = configparser.ConfigParser()
    config.read(conf_name)
    time_name      = config.get('calibration',   'time_channels')
    eng_name       = config.get('calibration', 'energy_channels')
    time_Edefault  = config.getint('threshold',    'timeE')
    eng_Edefault   = config.getint('threshold',  'energyE')
    time_T1default = config.getint('threshold',   'timeT1')
    eng_T1default  = config.getint('threshold', 'energyT1')
    time_T2default = config.getint('threshold',   'timeT2')
    eng_T2default  = config.getint('threshold', 'energyT2')

    ## Maybe not the best.
    time_EOffset   = config.getint('offset',    'timeE', fallback=0)
    time_TOffset   = config.getint('offset',   'timeT1', fallback=0)
    time_T2Offset  = config.getint('offset',   'timeT2', fallback=0)
    eng_EOffset    = config.getint('offset',  'energyE', fallback=0)
    eng_TOffset    = config.getint('offset', 'energyT2', fallback=0)
    ##

    map_file     = 'pet_code/test_data/SM_mapping_corrected.yaml'
    time_ch, eng_ch, *_ = read_ymlmapping(map_file)

    timeCh = pd.read_csv(time_name, sep='\t ').set_index('ID')['MU']

    time_mu = np.mean(timeCh)
    print(f'Time mu average value = {time_mu}')
    Ebin_wid     = round(time_mu / (time_Edefault + 0.5), 2)
    Ebin_lim     = Ebin_wid * np.ceil((max(timeCh) + 1) / Ebin_wid) + Ebin_wid
    time_EThresh = np.arange(0, Ebin_lim, Ebin_wid)
    plt.hist(timeCh, bins=time_EThresh)
    plt.xlabel('Time channel 511 keV peak position, E thresh binning (au)')
    plt.show()
    Tbin_wid     = round(time_mu / (time_T1default + 0.5), 2)
    Tbin_lim     = Tbin_wid * np.ceil((max(timeCh) + 1) / Tbin_wid) + Tbin_wid
    time_TThresh = np.arange(0, Tbin_lim, Tbin_wid)
    plt.hist(timeCh, bins=time_TThresh)
    plt.xlabel('Time channel 511 keV peak position, T1 thresh binning (au)')
    plt.show()
    T2bin_wid     = round(time_mu / (time_T2default + 0.5), 2)
    T2bin_lim     = T2bin_wid * np.ceil((max(timeCh) + 1) / T2bin_wid) + T2bin_wid
    time_T2Thresh = np.arange(0, T2bin_lim, T2bin_wid)
    plt.hist(timeCh, bins=time_T2Thresh)
    plt.xlabel('Time channel 511 keV peak position, T2 thresh binning (au)')
    plt.show()

    ## Energy channels
    engCh  = pd.read_csv(eng_name, sep='\t ').set_index('ID')['MU']
    eng_mu = np.mean(engCh)
    print(f'Energy mu average value = {eng_mu}')
    EEbin_wid   = round(eng_mu / (eng_Edefault + 0.5), 2)
    EEbin_lim   = EEbin_wid * np.ceil((max(engCh) + 1) / EEbin_wid) + EEbin_wid
    eng_EThresh = np.arange(0, EEbin_lim, EEbin_wid)
    print("binning check: ", EEbin_wid, ", ", EEbin_lim)
    plt.hist(engCh, bins=eng_EThresh)
    plt.xlabel('Energy channel 511 keV peak position, E thresh binning (au)')
    plt.show()
    ETbin_wid   = round(eng_mu / (eng_T2default + 0.5), 2)
    ETbin_lim   = ETbin_wid * np.ceil((max(engCh) + 1) / ETbin_wid) + ETbin_wid
    eng_TThresh = np.arange(0, ETbin_lim, ETbin_wid)
    plt.hist(engCh, bins=eng_TThresh)
    plt.xlabel('Energy channel 511 keV peak position, T2 thresh binning (au)')
    plt.show()


    print('Thresh check: ', time_mu, ", EThresh = ", np.searchsorted(time_EThresh, time_mu), ", T1 = ", np.searchsorted(time_TThresh, time_mu))

    output = config.get('output', 'out_path')
    # id_limit = max(max(time_ch), max(eng_ch))
    id_limit = 767
    with open(output, 'w') as thresh_out:
        thresh_out.write('#portID\tslaveID\tchipID\tchannelID\tvth_t1\tvth_t2\tvth_e\n')
        for id in range(0, id_limit + 1):
            portID, slaveID, chipID, channelID = get_electronics_nums(id)
            if   id in time_ch:
                vth_t1, vth_t2, vth_e = time_T1default, time_T2default, time_Edefault
                peak_pos = timeCh.get(id)
                if peak_pos:
                    vth_t1 = max(0, np.searchsorted(time_TThresh, peak_pos) - 1 + time_TOffset)
                    vth_t2 = max(0, np.searchsorted(time_T2Thresh, peak_pos) - 1 + time_T2Offset)
                    vth_e  = max(0, np.searchsorted(time_EThresh, peak_pos) - 1 + time_EOffset)
            elif id in  eng_ch:
                vth_t1, vth_t2, vth_e = eng_T1default, eng_T2default, eng_Edefault
                peak_pos = engCh.get(id)
                if peak_pos:
                    vth_t2 = max(0, np.searchsorted(eng_TThresh, peak_pos) - 1 + eng_TOffset)
                    vth_e  = max(0, np.searchsorted(eng_EThresh, peak_pos) - 1 + eng_EOffset)
            thresh_out.write(f'{portID}\t{slaveID}\t{chipID}\t{channelID}\t{vth_t1}\t{vth_t2}\t{vth_e}\n')
