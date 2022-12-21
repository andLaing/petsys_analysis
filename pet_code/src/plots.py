import matplotlib.pyplot as plt

from . fits import fit_gaussian
from . util import get_supermodule_eng
from . util import np
from . util import pd
from . util import select_energy_range
from . util import select_max_energy

def plot_settings():
    plt.rcParams[ 'lines.linewidth' ] =  2
    plt.rcParams[ 'font.size'       ] = 11
    plt.rcParams[ 'axes.titlesize'  ] = 19
    plt.rcParams[ 'axes.labelsize'  ] = 16
    plt.rcParams[ 'ytick.major.pad' ] = 14
    plt.rcParams[ 'xtick.major.pad' ] = 14
    plt.rcParams[ 'legend.fontsize' ] = 11

def hist1d(axis, data, bins=200, range=(0, 300), histtype='step', label='histo'):
    """
    plot a 1d histogram and return its
    """
    weights, pbins, _ = axis.hist(data, bins=bins, range=range, histtype=histtype, label=label)
    return pbins, weights


def mm_energy_spectra(module_xye, sm_label, plot_output=None, min_peak=150, brange=(0, 300), nsigma=2):
    """
    Generate the energy spectra and select the photopeak
    for each module. Optionally plot and save spectra
    and floodmaps.

    module_xye  : Dict
                  Supermodule xyz lists for each mm
                  key is mm number (1--),
    sm_label    : int
                  A label for the plots of which SM
                  is being processed.
    plot_output : None or String
                  If not None, the output base name for
                  the plots. When None, no plots made.
    min_peak    : int
                  Minimum entries in peak bin for fit.
    brange      : tuple
                  Lower and upper bounds for energy histograms.
    nsigma      : int
                  Number of sigmas in peak selection.
    return
                 List of energy selection filters.
    """
    photo_peak = []
    if plot_output:
        plot_settings()
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
        xfilt = None
        yfilt = None
        for j, ax in enumerate(axes.flatten()):
            ## mmini-module numbers start at 1
            try:
                bin_edges, bin_vals = hist1d(ax, module_xye[j+1]['energy'], range=brange, label=f'Det: {sm_label}\n mM: {j+1}')
            except KeyError:
                print(f'No data for super module {sm_label}, mini module {j+1}, skipping')
                photo_peak.append(lambda x: False)
                continue
            try:
                bcent, gvals, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6, min_peak=min_peak)
                minE, maxE = pars[1] - nsigma * pars[2], pars[1] + nsigma * pars[2]
            except RuntimeError:
                minE, maxE = 0, 300
            eng_arr = np.array(module_xye[j+1]['energy'])
            photo_peak.append(select_energy_range(minE, maxE))
            ax.plot(bcent, gvals, label=f'fit $\mu$ = {round(pars[1], 3)},  $\sigma$ = {round(pars[2], 3)}')
            ax.set_xlabel('Energy (au)')
            ## Filters for floodmaps
            ax.axvspan(minE, maxE, facecolor='#00FF00' , alpha = 0.3, label='Selected range')
            ax.legend()
            ## Filter the positions
            if xfilt is not None:
                xfilt = np.hstack((xfilt, np.array(module_xye[j+1]['x'])[photo_peak[-1](eng_arr)]))
                yfilt = np.hstack((yfilt, np.array(module_xye[j+1]['y'])[photo_peak[-1](eng_arr)]))
            else:
                xfilt = np.array(module_xye[j+1]['x'])[photo_peak[-1](eng_arr)]
                yfilt = np.array(module_xye[j+1]['y'])[photo_peak[-1](eng_arr)]
        ## Temporary for tests.
        out_name = plot_output.replace(".ldat","_EnergyModuleSMod" + str(sm_label) + ".png")
        fig.savefig(out_name)
        plt.clf()
        plt.hist2d(xfilt, yfilt, bins = 500, range=[[0, 104], [0, 104]], cmap="Reds", cmax=250)
        plt.xlabel('X position (pixelated) [mm]')
        plt.ylabel('Y position (monolithic) [mm]')
        plt.colorbar()
        plt.tight_layout()
        out_name = plot_output.replace(".ldat","_FloodModule" + str(sm_label) + ".png")
        plt.savefig(out_name)
        plt.clf()
    else:
        for j in range(1, 17):
            try:
                bin_vals, bin_edges = np.histogram(module_xye[j]['energy'], bins=200, range=brange)
            except KeyError:
                print(f'No data for super module {sm_label}, mini module {j+1}, skipping')
                photo_peak.append(lambda x: False)
                continue
            try:
                *_, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6, min_peak=min_peak)
                minE, maxE = pars[1] - nsigma * pars[2], pars[1] + nsigma * pars[2]
            except RuntimeError:
                minE, maxE = 0, 300
            photo_peak.append(select_energy_range(minE, maxE))
    return photo_peak


def slab_energy_spectra(slab_xye, plot_output=None, min_peak=150, bins=np.arange(9, 25, 0.2)):
    """
    Make energy spectra of slab time channels.
    slab_xye : Dict
               Keys slab channel id,
               values x, y and eng keyed lists.
    plot_output : String
                  If not None, output plots to this
                  name base.
    min_peak    : int
                  Minimum entries in peak bin for fit.
    returns
        Dict of energy selection filters
    """
    photo_peak = {}
    ## Limit range to avoid noise floor, can this be made more robust?
    # bins = np.arange(9, 25, 0.2)
    if plot_output:
        for slab, xye in slab_xye.items():
            #Try to exclude more noise
            first_bin = 0 if max(xye['energy']) < bins[-1] else int(3 / np.diff(bins[:2])[0])
            bin_vals, bin_edges, _ = plt.hist(xye['energy'], bins=bins[first_bin:])
            plt.xlabel(f'Energy (au) slab {slab}')
            try:
                bcent, gvals, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6, min_peak=min_peak)
                # Cutre protection
                if pars[1] <= bins[first_bin]:
                    minE, maxE = -1, 1
                else:
                    minE, maxE = pars[1] - 2 * pars[2], pars[1] + 2 * pars[2]
                plt.plot(bcent, gvals, label=f'fit $\mu$ = {round(pars[1], 3)},  $\sigma$ = {round(pars[2], 3)}')
            except RuntimeError:
                print(f'Failed fit, slab {slab}')
                minE, maxE = -1, 0
            photo_peak[slab] = select_energy_range(minE, maxE)
            plt.axvspan(minE, maxE, facecolor='#00FF00' , alpha = 0.3, label='Selected range')
            plt.legend()
            plt.savefig(plot_output.replace('.ldat', f'_slab{slab}Spec.png'))
            plt.clf()
    else:
        for slab, xye in slab_xye.items():
            first_bin = 0 if max(xye['energy']) < bins[-1] else int(3 / np.diff(bins[:2])[0])
            bin_vals, bin_edges = np.histogram(xye['energy'], bins=bins[first_bin:])
            try:
                *_, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6, min_peak=min_peak)
                # Cutre protection
                if pars[1] <= bins[first_bin]:
                    minE, maxE = -1, 0
                else:
                    minE, maxE = pars[1] - 2 * pars[2], pars[1] + 2 * pars[2]
            except RuntimeError:
                minE, maxE = -1, 0
            photo_peak[slab] = select_energy_range(minE, maxE)
    return photo_peak


def group_times(filtered_events, peak_select, mm_map, eng_ch, time_ch, ref_indx):
    """
    Group the first time signals for each slab
    in a reference super module with all slabs
    in other supermodules.
    filered_events: List of tuple of coincidence lists
                    The information for each event in structure
                    ([[id, mm, tstp, eng], ...], [...])
    peak_select   : List of Lists
                    Energy filter functions for each minimodule
    mm_map        : Callable
                    Maps channel id to minimodule number.
    eng_ch        : Set
                    Set of all channels for energy measurement.
    time_ch       : Set
                    Set of all channels for time measurement.
    ref_indx      : int
                    Index (0 or 1) of the reference SM.
    returns
    A dictionary of tuples (coinc id, coinc tstp, ref tstp)
    with key ref id.
    """
    if ref_indx not in (0, 1):
        # For calibration setup.
        raise ValueError('Currently only accepts 0 and 1 as ref_indx')
    reco_dt  = {}
    coinc_sm = 0 if ref_indx == 1 else 1
    min_ch   = [0, 0]
    for sm1, sm2 in filtered_events:
        mm1   = mm_map(sm1[0][0])
        _, e1 = get_supermodule_eng(sm1)
        mm2   = mm_map(sm2[0][0])
        _, e2 = get_supermodule_eng(sm2)
        if peak_select[0][mm1-1](e1) and peak_select[1][mm2-1](e2):
            try:
                min_ch[0] = select_max_energy(sm1, time_ch)
            except ValueError:
                continue
            try:
                min_ch[1] = select_max_energy(sm2, time_ch)
            except ValueError:
                continue
            ## Want to know the two channel numbers and timestamps.
            try:
                # Key is the reference channel/Slab, each has a
                # list of lists with [ch_other, t_other, t_ref]
                reco_dt[min_ch[ref_indx][0]].append([min_ch[coinc_sm][0],
                                                     min_ch[coinc_sm][2],
                                                     min_ch[ref_indx][2]])
            except KeyError:
                reco_dt[min_ch[ref_indx][0]] = [[min_ch[coinc_sm][0],
                                                 min_ch[coinc_sm][2],
                                                 min_ch[ref_indx][2]]]
    return reco_dt


def group_times_slab(filtered_events, peak_select, time_ch, ref_indx):
    """
    Group the first time signals for each slab
    in a reference super module with all slabs
    in other supermodules.
    ! Very similar to other group function
    ! How to combine?
    filered_events: List of tuple of coincidence lists
                    The information for each event in structure
                    ([[id, mm, tstp, eng], ...], [...])
    peak_select   : List of Dicts
                    Energy filter functions for each minimodule
    time_ch       : Set
                    Set of all channels for time measurement.
    ref_indx      : int
                    Index (0 or 1) of the reference SM.
    returns
    A dictionary of tuples (coinc id, coinc tstp, ref tstp)
    with key ref id.
    """
    reco_dt  = {}
    coinc_indx = 0 if ref_indx == 1 else 1
    min_ch   = [0, 0]
    for sm1, sm2 in filtered_events:
        try:
            min_ch[0] = select_max_energy(sm1, time_ch)
        except ValueError:
            continue
        try:
            min_ch[1] = select_max_energy(sm2, time_ch)
        except ValueError:
            continue
        if peak_select[0][min_ch[0][0]](min_ch[0][3]) and\
           peak_select[1][min_ch[1][0]](min_ch[1][3]):
            ## Want to know the two channel numbers and timestamps.
            try:
                # Key is the reference channel/Slab, each has a
                # list of lists with [ch_other, t_other, t_ref]
                reco_dt[min_ch[ref_indx][0]].append([min_ch[coinc_indx][0],
                                                     min_ch[coinc_indx][2],
                                                     min_ch[ref_indx  ][2]])
            except KeyError:
                reco_dt[min_ch[ref_indx][0]] = [[min_ch[coinc_indx][0],
                                                 min_ch[coinc_indx][2],
                                                 min_ch[ref_indx  ][2]]]
    return reco_dt


def group_times_list(filtered_events, peaks, time_ch, ref_indx):
    coinc_indx = 0 if ref_indx == 1 else 1
    def get_times(evt):
        try:
            chns = list(map(select_max_energy, evt, [time_ch] * 2))
        except ValueError:
            return
        if peaks[0][chns[0][0]](chns[0][3]) and peaks[1][chns[1][0]](chns[1][3]):
            return [chns[  ref_indx][0], chns[coinc_indx][0],
                    chns[coinc_indx][2], chns[  ref_indx][2]]

    return list(filter(lambda x: x, map(get_times, filtered_events)))


# Need to review all of this. Clearly repetition.
# Should be a general energy selection for all? Lookup?
def ctr(time_ch, peaks, skew=pd.Series(dtype=float)):
    """
    CTR
    """
    def timestamp_difference(evt):
        try:
            chns = [select_max_energy(sm, time_ch) for sm in evt]
        except ValueError:
            return
        if peaks[0][chns[0][0]](chns[0][3]) and peaks[1][chns[1][0]](chns[1][3]):
            return chns[0][2] - chns[1][2] + skew.get(chns[1][0], 0.0) - skew.get(chns[0][0], 0.0)
        return
    return timestamp_difference

