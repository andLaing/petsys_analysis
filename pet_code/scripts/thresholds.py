# Just to sort some values to test in configurations.

import sys

import pandas as pd
import numpy  as np

from pet_code.src.io import read_ymlmapping

def rel_mu(channels, fallback):
    def get_mu(id):
        try:
            return channels.loc[id].MU
        except KeyError:
            return fallback
    return get_mu


if __name__ == '__main__':
    file_name = sys.argv[1]

    map_file     = 'pet_code/test_data/SM_mapping_corrected.yaml'
    *_, slab_map = read_ymlmapping(map_file)
    time_ordered = sorted(slab_map.keys())

    timeCh = pd.read_csv(file_name, sep='\t ')
    timeCh.set_index('ID', inplace=True)

    mean_mu = np.mean(timeCh.MU)
    val_getter = rel_mu(timeCh, mean_mu)
    rel_thresh = [round(5 * val_getter(id) / mean_mu, 3) for id in time_ordered]
    print(rel_thresh)
