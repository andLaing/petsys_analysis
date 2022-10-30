#!/usr/bin/env python3

"""Generate energy spectra for all channels that appear in a petsys singles mode file

Usage: channel_specs.py [--mm] [--out OUTFILE] (--map MFILE) INPUT ...

Arguments:
    INPUT  Input file name or list of names

Required:
    --map=MFILE  Name of the module mapping yml file

Options:
    --mm   Output mini-module level energy channel sum spectra?
    --out  Name base for output image files [default: slabSpec]
"""

from docopt import docopt

import matplotlib.pyplot as plt
import numpy             as np

from pet_code.src.io import read_petsys_singles
from pet_code.src.io import read_ymlmapping


def channel_energies():
    specs = {}
    def add_value(channel):
        try:
            specs[channel[0]].append(channel[3])
        except KeyError:
            specs[channel[0]] = [channel[3]]
    # def get_spec(ch_id):
    #     try:
    #         return specs[ch_id]
    #     except KeyError:
    #         return []
    def loop_items():
        for id, engs in specs.items():
            yield id, engs
    return add_value, loop_items#get_spec


if __name__ == '__main__':
    args     = docopt(__doc__)
    mm_spec  = args['--mm']
    map_file = args['--map']
    out_file = args['--out']
    infiles  = args['INPUT']

    _, _, mm_map, _, _ = read_ymlmapping(map_file)
    add_val, spec_loop = channel_energies()
    for fn in infiles:
        reader = read_petsys_singles(fn, mm_map)
        _      = tuple(map(add_val, reader()))
    for id, engs in spec_loop():
        plt.hist(engs, bins=np.arange(30))
        plt.xlabel(f'Energy channel {id}')
        plt.savefig(out_file  + f'{id}.png')
        plt.clf()
