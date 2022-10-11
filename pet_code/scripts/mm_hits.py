import sys

from pet_code.src.io   import read_petsys_filebyfile, read_ymlmapping
from pet_code.src.io   import write_event_trace
from pet_code.src.util import filter_event_by_impacts


# Is there a better way?
def sort_and_write_mm(writer, sm_num):
    """
    Get info for each mini module in
    super module of interest and write
    to file.
    """
    def sort_and_write(evt):
        mm_dict = {}
        for hit in evt[sm_num -1]:
            try:
                mm_dict[hit[1]].append(hit)
            except KeyError:
                mm_dict[hit[1]] = [hit]
        for vals in mm_dict.values():
            writer(vals)
    return sort_and_write


if __name__ == '__main__':
    ## Should probably use docopt or config file.
    map_file   = 'pet_code/test_data/SM_mapping.yaml' # shouldn't be hardwired
    control_sm = int(sys.argv[1])
    valid_sm   = {1, 2}
    if control_sm not in valid_sm:
        print("Invalid control super module.")
        sys.exit()
    file_list  = sys.argv[2:]

    _, eng_ch, mm_map, centroid_map, _ = read_ymlmapping(map_file)

    evt_filter = filter_event_by_impacts(eng_ch, 5, 4)
    mm_check = 0
    all_evt  = 0
    for fn in file_list:
        out_file = fn.replace('.ldat', '_NN.txt')
        # Need to protect from overwrite? Will add output folder when using docopt/config or both
        with open(out_file, 'w') as tout:
            sort_writer = sort_and_write_mm(write_event_trace(tout, centroid_map), control_sm)
            reader      = read_petsys_filebyfile(fn, mm_map, evt_filter)
            for evt in reader():
                all_evt += 1
                n_mm = len(set(x[1] for x in evt[control_sm-1]))
                if n_mm > 1:
                    mm_check += 1
                sort_writer(evt)
    print("Proportion of events with multihit in sm: ", mm_check / all_evt)

