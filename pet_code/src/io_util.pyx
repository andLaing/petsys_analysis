from itertools import islice, chain
import numpy as np
cimport numpy as np

# cpdef tuple coinc_evt_loop(tuple first_line, object line_it, dict type_dict):
cpdef tuple coinc_evt_loop(np.ndarray first_line, object line_it, dict type_dict):
    cdef:
        list sm1 = []
        list sm2 = []
        set ch_sm1 = set()
        set ch_sm2 = set()
        int nlines = first_line['f0'].item() + first_line['f5'].item() - 2
        np.ndarray evt
    for evt in chain([first_line], islice(line_it, nlines)):
        id1 = evt['f4'].item()
        if id1 not in ch_sm1:
            # sm1.append([evt[4], type_dict[evt[4]], evt[2], evt[3]])
            sm1.append([id1, type_dict[id1], evt['f2'].item(), evt['f3'].item()])
            ch_sm1.add(id1)
        id2 = evt['f9'].item()
        if id2 not in ch_sm2:
            sm2.append([id2, type_dict[id2], evt['f7'].item(), evt['f8'].item()])
            ch_sm2.add(id2)
    return sm1, sm2

cpdef tuple singles_evt(np.ndarray first_line, object filemap, dict type_dict):
    cdef:
        # np.ndarray dummy = np.empty(0)
        int nlines = first_line['f0'].item() - 1
        # np.ndarray sm = np.empty((nlines+1, 4))
        list sm = []
        list dummy = []
        np.ndarray evt
        int id
    # for i, evt in enumerate(chain([first_line], islice(filemap, nlines))):
    for evt in chain([first_line], islice(filemap, nlines)):
        # sm[i] = evt['f4'].item(), type_dict[evt['f4'].item()], evt['f2'].item(), evt['f3'].item()
        id = evt['f4'].item()
        sm.append([id, type_dict[id], evt['f2'].item(), evt['f3'].item()])

    return sm, dummy


# cdef packed struct petsysGroup_t:
#     # unsigned char nline
#     # unsigned char indx
#     # signed long long tstp
#     # float energy
#     # signed int id
#     uint8_t nline
#     uint8_t indx
#     int64_t tstp
#     float32_t energy
#     int32_t id
