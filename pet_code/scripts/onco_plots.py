import os
import sys

import matplotlib.pyplot as plt
import numpy             as np

from pet_code.src.io   import read_petsys_filebyfile
from pet_code.src.io   import read_ymlmapping
from pet_code.src.util import filter_impact
from pet_code.src.util import shift_to_centres


def channel_engs():
    specs = {}
    def add_evt_info(sm):
        for imp in sm:
            try:
                specs[imp[0]].append(imp[3])
            except KeyError:
                specs[imp[0]] = [imp[3]]
    def gen_spacs():
        for id, spec in specs.items():
            yield id, spec
    def get_spec(id):
        try:
            return specs[id]
        except KeyError:
            return None
    return add_evt_info, gen_spacs, get_spec


def filter_minch(min_ch, eng_ch):
    filt = filter_impact(min_ch, eng_ch)
    def valid_event(sm, _):
        return filt(sm)
    return valid_event


if __name__ == '__main__':
    s_file  = sys.argv[1]
    ns_file = sys.argv[2]

    # Mentiras
    _, _, mm_map, _, _ = read_ymlmapping('pet_code/test_data/SM_mapping_corrected.yaml')
    #

    eng_ch    = {}
    evt_filt  = filter_minch(4, eng_ch)
    reader_s  = read_petsys_filebyfile( s_file, mm_map, sm_filter=evt_filt, singles=True)
    reader_ns = read_petsys_filebyfile(ns_file, mm_map, sm_filter=evt_filt, singles=True)

    spec_build_s , spec_read_s,        _ = channel_engs()
    spec_build_ns,           _, spec_get = channel_engs()
    for sm, _ in reader_s():
        spec_build_s(sm)
    for sm, _ in reader_ns():
        spec_build_ns(sm)
    bins    = np.arange(0, 30, 0.2)
    out_dir = 'onco_channel_tests'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_base = s_file.split('/')[-1]
    for id, s_spec in spec_read_s():
        bin_vals, bin_edges = np.histogram(s_spec, bins=bins)
        # bin_vals, bin_edges, _ = plt.hist(s_spec, bins=bins, label='Source spectrum')
        ns_spec = spec_get(id)
        if ns_spec is not None:
            ns_vals, _ = np.histogram(ns_spec, bins=bins)
            # ns_vals, *_ = plt.hist(ns_spec, bins=bins, label='No source spectrum')
            spec_diff = bin_vals - ns_vals
            specD_err = np.sqrt(bin_vals + ns_vals)
            plt.errorbar(shift_to_centres(bin_edges), spec_diff, yerr=specD_err, label='Difference')
            plt.xlabel('Charge (au)')
            plt.ylabel('au')
            plt.legend()
            plt.savefig(os.path.join(out_dir, out_base.replace('.ldat', f'_EspecCh{id}.png')))

