import sys
import time

import matplotlib.pyplot as plt
import numpy  as np

from pet_code.src.fits  import fit_gaussian
from pet_code.src.io    import read_petsys
from pet_code.src.io    import read_ymlmapping
from pet_code.src.plots import group_times_slab
from pet_code.src.plots import slab_energy_spectra
from pet_code.src.util  import centroid_calculation
from pet_code.src.util  import filter_impacts_specific_mod
from pet_code.src.util  import select_energy_range
from pet_code.src.util  import slab_energy_centroids
from pet_code.src.util  import time_of_flight

if __name__ == '__main__':
    ## Should probably use docopt or config file.
    start = time.time()
    map_file  = 'pet_code/test_data/SM_mapping.yaml' # shouldn't be hardwired
    file_list = sys.argv[1:]

    time_ch, eng_ch, mm_map, centroid_map, slab_map = read_ymlmapping(map_file)
    # Shouldn't be hardwired!!
    ref_indx   = 1
    evt_select = filter_impacts_specific_mod(ref_indx, 10, eng_ch, 5, 4)
    pet_reader = read_petsys(mm_map, evt_select)
    filtered_events = [tpl for tpl in pet_reader(file_list)]
    end_r = time.time()
    print("Time enlapsed reading: {} s".format(int(end_r - start)))
    print("length check: ", len(filtered_events))
    ## Should we be filtering the events with multiple mini-modules in one sm?
    c_calc     = centroid_calculation(centroid_map)
    slab_dicts = slab_energy_centroids(filtered_events, c_calc, time_ch)

    out_base   = 'test_plots/slab_filt/' + file_list[0].split('/')[-1]
    photo_peak = list(map(slab_energy_spectra, slab_dicts, [out_base] * 2))

    reco_dt = group_times_slab(filtered_events, photo_peak, time_ch, ref_indx)
    # Source pos hardwired for tests, extract from data file?
    flight_time = time_of_flight(np.array([38.4, 38.4, 22.5986]))
    for ref_ch, tstps in reco_dt.items():
        time_arr   = np.array(tstps)
        dt_th      = flight_time(slab_map[ref_ch]) - np.fromiter((flight_time(slab_map[id]) for id in time_arr[:, 0]), float)
        tstp_diff  = np.diff(time_arr[:, 1:], axis=1).flatten()
        plt.hist(tstp_diff, bins=300, range=[-10000, 10000], histtype='step', fill=False, label = f"Ref ch {ref_ch}")
        plt.hist(tstp_diff - dt_th, bins=300, range=[-10000, 10000], histtype='step', fill=False, label = f"Ref ch {ref_ch} theory corr dt")
        plt.xlabel(f'tstp ch {ref_ch} - tstp coinc (ps)')
        plt.legend()
        ## Temp hardwire!!
        out_name = 'test_plots/slab_filt/' + file_list[0].split('/')[-1].replace('.ldat', '_timeCoincRef' + str(ref_ch) + '.png')
        plt.savefig(out_name)
        plt.clf()
