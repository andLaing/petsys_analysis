import sys

from pet_code.src.io   import read_petsys, read_ymlmapping
from pet_code.src.util import filter_impact


def read_and_filter(eng_map, min_sm1, min_sm2):
    """
    Name should be improved and it should
    be in another file.
    return a function to iterate on the
    event generator and return valid events.
    """
    m1_filter = filter_impact(min_sm1, eng_map)
    m2_filter = filter_impact(min_sm2, eng_map)
    # def valid_event(evt_tuple):
    #     return m1_filter(evt_tuple[0]) and m2_filter(evt_tuple[1])
    def valid_event(sm1, sm2):
        return m1_filter(sm1) and m2_filter(sm2)
    return valid_event

import time
if __name__ == '__main__':
    ## Should probably use docopt or config file.
    start = time.time()
    map_file  = 'pet_code/test_data/SM_mapping.yaml' # shouldn't be hardwired
    file_list = sys.argv[1:]

    time_ch, eng_ch, mm_map, centroid_map = read_ymlmapping(map_file)
    evt_select = read_and_filter(eng_ch, 5, 4) # minima should not be hardwired
    pet_reader = read_petsys(mm_map, evt_select)
    # filtered_events = list(filter(evt_select, pet_reader(file_list)))
    filtered_events = [tpl for tpl in pet_reader(file_list)]
    end_r = time.time()
    print("Time enlapsed reading: {} s".format(int(end_r - start)))
    print("length check: ", len(filtered_events))
