import sys

import numpy as np

from pet_code.src.filters import filter_event_by_impacts
from pet_code.src.io      import ChannelMap
from pet_code.src.io      import read_petsys_filebyfile
from pet_code.src.io      import read_ymlmapping
from pet_code.src.util    import ChannelType
from pet_code.src.util    import select_max_energy


def print_info(all_channels, neg_channels, ch_map):
    print(f'{len(neg_channels)} negative channel(s) found for sm {ch_map.get_supermodule(neg_channels[0][0])}')
    total_chan = len(all_channels)
    total_time = sum(x[1] is ChannelType.TIME for x in all_channels)
    nMM        = len(set(ch_map.get_minimodule(x[0]) for x in all_channels))
    neg_MM     = [ch_map.get_minimodule(x[0]) for x in neg_channels]
    max_time   = select_max_energy(all_channels, ChannelType.TIME  )
    max_eng    = select_max_energy(all_channels, ChannelType.ENERGY)
    neg_ids    = [x[0] for x in neg_channels]
    neg_engs   = [x[3] for x in neg_channels]
    time_neg   = sum(x[1] is ChannelType.TIME for x in neg_channels)
    print(f'SM has {total_chan} channels, {total_time} of which are time channels.')
    print(f'{nMM} mini-modules have charge.')
    print(f'Time channel max is: {max_time}')
    print(f'Energy channel max is: {max_eng}')
    print(f'The negative ids are {neg_ids}, {time_neg} of them are time channels.')
    print(f'The negative signals are {neg_engs}')
    return neg_MM

if __name__ == '__main__':
    infile   = sys.argv[1]
    map_file = sys.argv[2]

    singles  = 'coinc' not in infile
    chan_map = ChannelMap(map_file)
    # time_ch, eng_ch, mm_map, *_ = read_ymlmapping('pet_code/test_data/SM_mapping_corrected.yaml')
    evt_filter = filter_event_by_impacts(4, singles=singles)
    reader     = read_petsys_filebyfile(chan_map.ch_type, evt_filter, singles=singles)

    print(f'Checking for negative channels in {infile}')
    total_evt   = 0
    neg_in_sm1  = 0
    neg_in_sm2  = 0
    neg_in_both = 0
    neg_ch1     = []
    neg_ch2     = []
    neg_MMs1    = []
    neg_MMs2    = []
    for sm1, sm2 in reader(infile):
        total_evt += 1
        sm1_neg = list(filter(lambda x: x[3] < 0, sm1))
        sm2_neg = list(filter(lambda x: x[3] < 0, sm2))
        if sm1_neg:
            neg_in_sm1 += 1
            neg_MMs1.extend(print_info(sm1, sm1_neg, chan_map))
        if sm2_neg:
            neg_in_sm2 += 1
            neg_MMs2.extend(print_info(sm2, sm2_neg, chan_map))
        if sm1_neg and sm2_neg:
            neg_in_both += 1
        neg_ch1.extend(x[0] for x in sm1_neg)
        neg_ch2.extend(x[0] for x in sm2_neg)
    print(f'Total events {total_evt}')
    print(f'{neg_in_both} events have negatives in both SM')
    print(f'{neg_in_sm1} events with at least one negative first list(indx 0)')
    print(f'{neg_in_sm2} events with at least one negative second list(indx 1)')
    print(f'{np.unique(neg_ch1)} channels in first SM have negatives, {np.unique(neg_MMs1)} minimodules.')
    print(f'{np.unique(neg_ch2)} channels in second SM have negatives, {np.unique(neg_MMs2)} minimodules.')
