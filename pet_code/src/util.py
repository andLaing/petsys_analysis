def get_no_eng_channels(mod_data, energy_chid):
    """
    Return the number of channels for energy
    measurement in the module data list.
    """
    return sum(x[0] in energy_chid for x in mod_data)


def get_supermodule_eng(mod_data, energy_chid):
    """
    Return the number of channels for energy
    measurement in the module data and the
    total energy deposited.
    """
    eng_ch = list(filter(lambda x: x[0] in energy_chid, mod_data))
    return len(eng_ch), sum(hit[3] for hit in eng_ch)


def filter_impact(min_ch, energy_chid):
    """
    Make a filter to check impacts recorded
    in sufficient channels.
    """
    def valid_impact(mod_data):
        neng = get_no_eng_channels(mod_data, energy_chid)
        return min_ch < neng < len(mod_data)
    return valid_impact


def centroid_calculation(centroid_map, offset_x=0.00001, offset_y=0.00001):
    """
    Calculates the centroid of a set of module
    data according to a centroid map.
    """
    powers  = [1, 2]
    offsets = [offset_x, offset_y]
    def centroid(data):
        """
        Calculate the average position of the time
        and energy channels and return them plus
        the total energy channel deposit.
        """
        sums    = [0.0, 0.0]
        weights = [0.0, 0.0]
        for imp in data:
            en_t, pos      = centroid_map[imp[0]]
            weight         = (imp[3] + offsets[en_t])**powers[en_t]
            sums   [en_t] += weight * pos
            weights[en_t] += weight
        return (sums[0] / weights[0] if weights[0] else 0.0,
                sums[1] / weights[1] if weights[1] else 0.0, weights[1])
    return centroid


def slab_position(pos):
    """
    Calculate the index of
    a slab in a mini-module using
    its position.
    """
    slab_num = round((pos - 1.6) / 3.2) # Safe to have these hardwired?
    indx     = slab_num - slab_num // 8 * 8
    return indx if indx < 8 else indx - 8
