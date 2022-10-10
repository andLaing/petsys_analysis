import sys

import matplotlib.pyplot as plt
import numpy  as np
# import pandas as pd

from pet_code.src.io    import read_petsys, read_ymlmapping
from pet_code.src.plots import mm_energy_spectra, group_times
from pet_code.src.util  import filter_impacts_one_minimod, get_supermodule_eng
from pet_code.src.util  import centroid_calculation


# def module_centroids(event, c_calc):
#     """
#     Get all relevant centroids for each super module in the event.
#     Muy cutre ahora.
#     """
#     sm1_dict = pd.DataFrame(event[0], columns=['id', 'mm', 'tstp', 'eng']).groupby('mm').apply(lambda x: c_calc(x.values)).to_dict()
#     sm2_dict = pd.DataFrame(event[1], columns=['id', 'mm', 'tstp', 'eng']).groupby('mm').apply(lambda x: c_calc(x.values)).to_dict()
#     return sm1_dict, sm2_dict


# def range_filter(val, vmin, vmax):
#     """
#     Select
#     """


import time
if __name__ == '__main__':
    ## Should probably use docopt or config file.
    start = time.time()
    map_file  = 'pet_code/test_data/SM_mapping.yaml' # shouldn't be hardwired
    file_list = sys.argv[1:]

    time_ch, eng_ch, mm_map, centroid_map, _ = read_ymlmapping(map_file)
    evt_select = filter_impacts_one_minimod(eng_ch, 5, 4) # minima should not be hardwired
    pet_reader = read_petsys(mm_map, evt_select)
    filtered_events = [tpl for tpl in pet_reader(file_list)]
    end_r = time.time()
    print("Time enlapsed reading: {} s".format(int(end_r - start)))
    print("length check: ", len(filtered_events))
    ## Should we be filtering the events with multiple mini-modules in one sm?
    c_calc = centroid_calculation(centroid_map)
    # ## Must be a better way but...
    mod_dicts = [{}, {}]
    for evt in filtered_events:
        for i, ((x, y, _), (_, eng)) in enumerate(zip(map(c_calc, evt), map(get_supermodule_eng, evt, [eng_ch] * 2))):
            mm = evt[0][i][1]
            try:
                mod_dicts[i][mm]['x'].append(x)
                mod_dicts[i][mm]['y'].append(y)
                mod_dicts[i][mm]['energy'].append(eng)
            except KeyError:
                mod_dicts[i][mm] = {'x': [x], 'y': [y], 'energy': [eng]}

    # ## No file separation in case of multiple files, needs to be fixed.
    # print("Stats check: ", l1, ", ", l2)
    photo_peak = list(map(mm_energy_spectra, mod_dicts, [1, 2], [file_list[0]] * 2))

    ## CTR. Reference to a slab in sm2
    reco_dt = group_times(filtered_events, photo_peak, eng_ch, time_ch, 1)
    ref_count = 0
    for ref_ch, tstps in reco_dt.items():
        #print("length = ", len(tstps))
        ref_count += 1
        time_arr = np.array(tstps)
        plt.hist(np.diff(time_arr[:, 1:], axis=1).flatten(), bins=151, range=[-5000, 5000], histtype='step', fill=False, label = f"Ref ch {ref_ch}")
        plt.xlabel(f'tstp ch {ref_ch} - tstp coinc')
        # plt.show()
    print("Ref channels found: ", ref_count)
            


