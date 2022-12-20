from itertools import islice, chain

cpdef tuple coinc_evt_loop(tuple first_line, object line_it, object type_func):
    cdef:
        list sm1 = []
        list sm2 = []
        set ch_sm1 = set()
        set ch_sm2 = set()
        int nlines = first_line[0] + first_line[5] - 2
        tuple evt
    for evt in chain([first_line], islice(line_it, nlines)):
        if evt[4] not in ch_sm1:
            sm1.append([evt[4], type_func(evt[4]), evt[2], evt[3]])
            ch_sm1.add(evt[4])
        if evt[-1] not in ch_sm2:
            sm2.append([evt[-1], type_func(evt[-1]), evt[7], evt[8]])
            ch_sm2.add(evt[-1])
    return sm1, sm2
