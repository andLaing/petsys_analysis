import sys

import matplotlib.pyplot as plt
import numpy  as np
# import pandas as pd

from pet_code.src.io    import read_petsys, read_ymlmapping
from pet_code.src.plots import mm_energy_spectra, group_times
from pet_code.src.util  import filter_event_by_impacts
from pet_code.src.util  import filter_impacts_one_minimod, get_supermodule_eng
from pet_code.src.util  import filter_impacts_specific_mod
from pet_code.src.util  import select_module
from pet_code.src.util  import centroid_calculation
from pet_code.src.util  import mm_energy_centroids
from pet_code.src.util  import time_of_flight


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
    map_file  = 'pet_code/test_data/SM_mapping_corrected.yaml' # shouldn't be hardwired
    file_list = sys.argv[1:]

    time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    evt_select = filter_event_by_impacts(eng_ch, 4, 4)
    # evt_select = filter_impacts_one_minimod(eng_ch, 4, 4) # minima should not be hardwired
    # evt_select = filter_impacts_specific_mod(1, 10, eng_ch, 5, 4)
    pet_reader = read_petsys(mm_map, evt_select)
    filtered_events = [tpl for tpl in pet_reader(file_list)]
    end_r = time.time()
    print("Time enlapsed reading: {} s".format(int(end_r - start)))
    print("length check: ", len(filtered_events))
    ## Should we be filtering the events with multiple mini-modules in one sm?
    c_calc = centroid_calculation(centroid_map)
    # ## Must be a better way but...
    def wrap_mmsel(eng_ch):
        def sel(sm):
            return select_module(sm, eng_ch)
        return sel
    # mod_dicts = mm_energy_centroids(filtered_events, c_calc, eng_ch)
    mod_dicts = mm_energy_centroids(filtered_events, c_calc, eng_ch, mod_sel=wrap_mmsel(eng_ch))

    # ## No file separation in case of multiple files, needs to be fixed.
    # print("Stats check: ", l1, ", ", l2)
    out_base = 'test_floods/' + file_list[0].split('/')[-1]
    photo_peak = list(map(mm_energy_spectra, mod_dicts, [1, 2], [out_base] * 2, [100] * 2))

    ## CTR. Reference to a slab in sm1
    # reco_dt = group_times(filtered_events, photo_peak, eng_ch, time_ch, 1)
    # ref_count = 0
    # # Source pos hardwired for tests, extract from data file?
    # flight_time = time_of_flight(np.array([38.4, 38.4, 22.5986]))
    # for ref_ch, tstps in reco_dt.items():
    #     #print("length = ", len(tstps))
    #     ref_count += 1
    #     time_arr   = np.array(tstps)
    #     dt_th      = flight_time(slab_map[ref_ch]) - np.fromiter((flight_time(slab_map[id]) for id in time_arr[:, 0]), float)
    #     # print("Ref pos: ", slab_map[ref_ch], ", time: ", flight_time(slab_map[ref_ch]))
    #     # print("Whit? ", dt_th)
    #     tstp_diff  = np.diff(time_arr[:, 1:], axis=1).flatten()
    #     plt.hist(tstp_diff, bins=300, range=[-10000, 10000], histtype='step', fill=False, label = f"Ref ch {ref_ch}")
    #     plt.hist(tstp_diff - dt_th, bins=300, range=[-10000, 10000], histtype='step', fill=False, label = f"Ref ch {ref_ch} theory corr dt")
    #     plt.xlabel(f'tstp ch {ref_ch} - tstp coinc (ps)')
    #     plt.legend()
    #     ## Temp hardwire!!
    #     out_name = 'test_plots/' + file_list[0].split('/')[-1].replace('.ldat', '_timeCoincRef' + str(ref_ch) + '.png')
    #     plt.savefig(out_name)
    #     plt.clf()
    #     # plt.hist(tstp_diff - dt_th, bins=151, range=[-5000, 5000], histtype='step', fill=False, label = f"Ref ch {ref_ch}")
    #     # plt.xlabel(f'tstp ch {ref_ch} - tstp coinc  - dt (geom) (ps)')
    #     # plt.savefig(out_name.replace('timeCoincRef', 'timeCoincTheoryRef'))
    #     # plt.clf()
    #     # plt.show()
    # print("Ref channels found: ", ref_count)
            


