import sys

import numpy as np

from pet_code.src.io   import read_petsys_filebyfile
from pet_code.src.io   import read_ymlmapping
from pet_code.src.util import filter_impact

def filter_minch(min_ch, eng_ch):
    filt = filter_impact(min_ch, eng_ch)
    def valid_event(sm, _):
        return filt(sm)
    return valid_event

if __name__ == '__main__':
    chan_test = int(sys.argv[1])
    infile    =     sys.argv[2]
    q_low     =   0
    q_high    = 100
    if len(sys.argv) > 3:
        q_low  = float(sys.argv[3])
        q_high = float(sys.argv[4])

    time_ch, eng_ch, mm_map, *_ = read_ymlmapping('pet_code/test_data/SM_mapping_corrected.yaml')

    evt_filt = filter_minch(4, eng_ch)
    reader   = read_petsys_filebyfile(mm_map, sm_filter=evt_filt, singles=True)
    print(f'Checking for events with channel {chan_test} in Q range ({q_low}, {q_high})')
    occ_count = 0
    max_chan  = 0
    for sm, _ in reader(infile):
        chan_found = sum(map(lambda x: x[0] == chan_test and q_low < x[3] < q_high, sm))
        if chan_found:
            occ_count += 1
            sm_arr = np.asarray(sm)
            MMs    = set(sm_arr[:, 1])
            e_chan = np.fromiter(map(lambda x: x[0] in eng_ch, sm), bool)
            sums   = np.fromiter((sm_arr[(sm_arr[:, 1] == mm) & e_chan, 3].sum() for mm in MMs), float)
            time_E = [(x[0], x[3]) for x in filter(lambda x: x[0] in time_ch, sm)]
            if time_E[0][0] == chan_test:
                max_chan += 1
            print(f'Found with MMs {MMs} with info')
            print(f'MM energy sums = {sums}')
            print(f'Time channels (id, eng): {time_E}')
    print(f'Channel in event {occ_count} times of which {max_chan} it is the first/max channel')

