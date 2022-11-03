#!/usr/bin/env python3

"""Generate energy spectra for all channels that appear in a petsys group mode file

Usage: channel_specs.py [--mm] [--out OUTFILE] (--map MFILE) INPUT ...

Arguments:
    INPUT  Input file name or list of names

Required:
    --map=MFILE  Name of the module mapping yml file

Options:
    --mm   Output mini-module level energy channel sum spectra?
    --out=OUTFILE  Name base for output image files [default: slabSpec]
"""

from docopt      import docopt
from collections import Counter

import matplotlib.pyplot as plt
import numpy             as np

from pet_code.src.io   import read_petsys_filebyfile
from pet_code.src.io   import read_ymlmapping
from pet_code.src.util import filter_impact, filter_multihit
from pet_code.src.util import select_module


class ChannelCal:
    def __init__(self, time_ch, mm_indx) -> None:
        self.slab_energies = {}
        self.eng_accum     = {}
        self.saved_evt     = []
        self.time_id       = time_ch
        self.mm_indx       = mm_indx

    def add_evt(self, supermods):
        mms = {ch[1] for ch in supermods[0]}
        # try:
        #     t_ch = next(filter(lambda ch: ch[0] in self.time_id, supermods[0]))
        # except StopIteration:
        #     return len(mms)
        n_ch = -1
        for n_ch, ch in enumerate(filter(lambda x: x[0] in self.time_id, supermods[0])):
            try:
                self.slab_energies[ch[0]].append(ch[3])
            except KeyError:
                self.slab_energies[ch[0]] = [ch[3]]
        if n_ch >= 0:
            self.saved_evt.append(supermods[0])
        return len(mms)
        # for ch in supermods[0]:
        #     if ch[0] in time_ch:
        #         mms.add(ch[1])
        #         try:
        #             self.slab_energies[ch[0]].append(ch[3])
        #         except KeyError:
        #             self.slab_energies[ch[0]] = [ch[3]]
        #     else:
        #         mms.add(ch[1])
        #         sm = ch[0] // 256
        #         i  = np.argwhere(self.mm_indx == ch[0] % 256)[0][1]
        #         try:
        #             self.eng_accum[sm][ch[1]][i] += ch[3]
        #         except KeyError:
        #             try:
        #                 self.eng_accum[sm][ch[1]]    = np.zeros(8)
        #                 self.eng_accum[sm][ch[1]][i] = ch[3]
        #             except KeyError:
        #                 self.eng_accum[sm] = {ch[1]: np.zeros(8)}
        #                 self.eng_accum[sm][ch[1]][i] = ch[3]
        # return len(mms)


def channel_energies(eng_ch, time_ch):
    specs = {}
    def add_value(channels):
        for impact in channels[0]:
            try:
                specs[impact[1]][impact[0]].append(impact[3])
            except KeyError:
                try:
                    specs[impact[1]][impact[0]] = [impact[3]]
                except KeyError:
                    specs[impact[1]] = {impact[0]: [impact[3]]}
    def add_mm_max(channels):
        arr_imps = np.asarray(channels[0])
        # Shite code, improve
        for mm in np.unique(arr_imps[:, 1]):
            mm_info  = arr_imps[arr_imps[:, 1] == mm]
            max_vals = mm_info[np.argmax(mm_info[:, 3])]
            try:
                specs[mm][max_vals[0]].append(max_vals[3])
            except KeyError:
                try:
                    specs[mm][max_vals[0]] = [max_vals[3]]
                except KeyError:
                    specs[mm] = {max_vals[0]: [max_vals[3]]}
    def add_max_mm(channels):
        max_mm = select_module(channels[0])
        mm_id  = max_mm[0][1]
        for impact in max_mm:
            try:
                specs[mm_id][impact[0]].append(impact[3])
            except KeyError:
                try:
                    specs[mm_id][impact[0]] = [impact[3]]
                except KeyError:
                    specs[mm_id] = {impact[0]: [impact[3]]}
    def add_engtime_max(channels):
        """
        Only use the max of the energy channels and that of time ch
        separately.
        """
        e_chans   = list(map(lambda x: x[3], filter(lambda x: x[0] in  eng_ch, channels[0])))
        t_chans   = list(map(lambda x: x[3], filter(lambda x: x[0] in time_ch, channels[0])))
        
        if e_chans:
            eng_max  = channels[0][np.argmax(e_chans)]
            try:
                specs[eng_max[1]][eng_max[0]].append(eng_max[3])
            except KeyError:
                try:
                    specs[eng_max[1]][eng_max[0]] = [eng_max[3]]
                except KeyError:
                    specs[eng_max[1]] = {eng_max[0]: [eng_max[3]]}
        if t_chans:
            time_max = channels[0][np.argmax(t_chans)]
            try:
                specs[time_max[1]][time_max[0]].append(time_max[3])
            except KeyError:
                try:
                    specs[time_max[1]][time_max[0]] = [time_max[3]]
                except KeyError:
                    specs[time_max[1]] = {time_max[0]: [time_max[3]]}

    def loop_channels():
        for ids in specs.values():
            for id, engs in ids.items():
                yield id, engs
    def loop_mms():
        for mm, vals in specs.items():
            yield mm, vals
    # return add_value, loop_channels, loop_mms#get_spec
    # return add_mm_max, loop_channels, loop_mms
    return add_engtime_max, loop_channels, loop_mms


def filter_minch(min_ch, eng_ch):
    filt = filter_impact(min_ch, eng_ch)
    def valid_event(sm, _):
        return filt(sm)
    return valid_event


def filter_oneMM():
    def valid_event(sm, _):
        return filter_multihit(sm)
    return valid_event


if __name__ == '__main__':
    args     = docopt(__doc__)
    mm_spec  = args['--mm']
    map_file = args['--map']
    out_file = args['--out']
    infiles  = args['INPUT']

    time_ch, eng_ch, mm_map, _, _ = read_ymlmapping(map_file)
    # add_val, spec_loop, mm_loop   = channel_energies()
    # add_val, spec_loop, mm_loop   = channel_energies(eng_ch, time_ch)
    # filt = filter_minch(4, eng_ch)
    # filt = filter_oneMM()
    plotter = ChannelCal(time_ch, mm_index)
    # Maybe a bit dangerous memory wise, review
    for fn in infiles:
        # reader = read_petsys_filebyfile(fn, mm_map, sm_filter=filt, singles=True)
        reader  = read_petsys_filebyfile(fn, mm_map, singles=True)
        # _      = tuple(map(add_val, reader()))
        num_mms = tuple(map(plotter.add_evt, reader()))
    print("First pass complete, mm multiplicities: ", Counter(num_mms))
    for id, engs in plotter.slab_energies.items():
        plt.hist(engs, bins=np.arange(0, 30, 0.2))
        plt.xlabel(f'Energy time channel {id}')
        plt.savefig(out_file  + f'{id}.png')
        plt.clf()
    # for id, engs in spec_loop():
    #     plt.hist(engs, bins=np.arange(0, 30, 0.2))
    #     ch_type = 'time' if id in time_ch else 'energy'
    #     plt.xlabel(f'Energy {ch_type} channel {id}')
    #     plt.savefig(out_file  + f'{id}.png')
    #     plt.clf()
    # # The following isn't really what we're after.
    # if mm_spec:
    #     for mm, engs in mm_loop():
    #         all_engs = np.concatenate([e for _, e in filter(lambda k: k[0] in eng_ch, engs.items())])
    #         plt.hist(all_engs, np.arange(0, 30, 0.2))
    #         plt.xlabel(f'Energy spec all channels mm{mm}')
    #         plt.savefig(out_file + f'mm{mm}.png')
    #         plt.clf()

