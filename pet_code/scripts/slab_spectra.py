import sys
import time

import matplotlib.pyplot as plt
import numpy  as np

from pet_code.src.fits import fit_gaussian
from pet_code.src.io   import read_petsys
from pet_code.src.io   import read_ymlmapping
from pet_code.src.util import centroid_calculation
from pet_code.src.util import filter_impacts_specific_mod
from pet_code.src.util import slab_energy_centroids
from pet_code.src.util import time_of_flight


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

    photo_peak = [{}, {}]
    out_base   = 'test_plots/slab_filt/' + file_list[0].split('/')[-1]
    for i, sm in enumerate(slab_dicts):
        for slab, xye in sm.items():
            ## Limit range to avoid noise floor, can this be made more robust?
            bin_vals, bin_edges, _ = plt.hist(xye['energy'], bins=np.arange(8, 30, 0.2))
            plt.xlabel(f'Energy (au) slab {slab}')
            try:
                bcent, gvals, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6)
                minE, maxE = pars[1] - 2 * pars[2], pars[1] + 2 * pars[2]
            except RuntimeError:
                print(f'Failed fit, slab {slab}')
                minE, maxE = 10, 20
            photo_peak[i][slab] = lambda eng_val: (eng_val > minE) & (eng_val < maxE)
            try:
                plt.plot(bcent, gvals, label=f'fit $\mu$ = {round(pars[1], 3)},  $\sigma$ = {round(pars[2], 3)}')
                plt.axvspan(minE, maxE, facecolor='#00FF00' , alpha = 0.3, label='Selected range')
            except ValueError:
                print(f'Fit fucked, slab {slab}')
                minE, maxE = 10, 20
            plt.legend()
            # plt.show()
            plt.savefig(out_base.replace('.ldat', f'_slab{slab}Spec.png'))
            plt.clf()

    reco_dt = {}
    min_ch = [0, 0]
    coic_indx = 1 if ref_indx == 0 else 0
    for sm1, sm2 in filtered_events:
        try:
            min_ch[0] = next(filter(lambda x: x[0] in time_ch, sm1))
        except StopIteration:
            continue
        try:
            min_ch[1] = next(filter(lambda x: x[0] in time_ch, sm2))
        except StopIteration:
            continue
        if photo_peak[0][min_ch[0][0]](min_ch[0][3]) and photo_peak[1][min_ch[1][0]](min_ch[1][3]):
            ## Want to know the two channel numbers and timestamps.
            try:
                # Key is the reference channel/Slab, each has a
                # list of lists with [ch_other, t_other, t_ref]
                reco_dt[min_ch[ref_indx][0]].append([min_ch[coic_indx][0],
                                                     min_ch[coic_indx][2],
                                                     min_ch[ref_indx ][2]])
            except KeyError:
                reco_dt[min_ch[ref_indx][0]] = [[min_ch[coic_indx][0],
                                                 min_ch[coic_indx][2],
                                                 min_ch[ref_indx ][2]]]
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
