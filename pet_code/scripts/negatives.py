import sys

import numpy as np

from pet_code.src.io   import read_petsys_filebyfile
from pet_code.src.io   import read_ymlmapping
from pet_code.src.util import filter_event_by_impacts
from pet_code.src.util import select_max_energy


def print_info(all_channels, neg_channels, time_ch, eng_ch):
    print(f'{len(neg_channels)} negative channel(s) found for sm {(neg_channels[0][0] // 256) + 1}')
    total_chan = len(all_channels)
    total_time = sum(x[0] in time_ch for x in all_channels)
    nMM        = len(set(x[1] for x in all_channels))
    max_time   = select_max_energy(all_channels, time_ch)
    max_eng    = select_max_energy(all_channels,  eng_ch)
    neg_ids    = [x[0] for x in neg_channels]
    neg_engs   = [x[3] for x in neg_channels]
    time_neg   = sum(x[0] in time_ch for x in neg_channels)
    print(f'SM has {total_chan} channels, {total_time} of which are time channels.')
    print(f'{nMM} mini-modules have charge.')
    print(f'Time channel max is: {max_time}')
    print(f'Energy channel max is: {max_eng}')
    print(f'The negative ids are {neg_ids}, {time_neg} of them are time channels.')
    print(f'The negative signals are {neg_engs}')

if __name__ == '__main__':
    infile = sys.argv[1]

    singles = 'coinc' not in infile
    time_ch, eng_ch, mm_map, *_ = read_ymlmapping('pet_code/test_data/SM_mapping_corrected.yaml')
    evt_filter = filter_event_by_impacts(eng_ch, 4, 4, singles=singles)
    reader     = read_petsys_filebyfile(infile, mm_map, sm_filter=evt_filter, singles=singles)

    print(f'Checking for negative channels in {infile}')
    total_evt  = 0
    neg_in_sm1 = 0
    neg_in_sm2 = 0
    for sm1, sm2 in reader():
        total_evt += 1
        sm1_neg = list(filter(lambda x: x[3] < 0, sm1))
        sm2_neg = list(filter(lambda x: x[3] < 0, sm2))
        if sm1_neg:
            neg_in_sm1 += 1
            print_info(sm1, sm1_neg, time_ch, eng_ch)
        if sm2_neg:
            neg_in_sm2 += 1
            print_info(sm2, sm2_neg, time_ch, eng_ch)
    print(f'Total events {total_evt}')
    print(f'{neg_in_sm1} events with at least one negative first list(indx 0)')
    print(f'{neg_in_sm2} events with at least one negative second list(indx 1)')
