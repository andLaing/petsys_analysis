import matplotlib.pyplot as plt

from . fits import fit_gaussian
from . util import get_supermodule_eng
from . util import np

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


def mm_energy_spectra(module_xye, sm_label, plot_output=None):
    """
    Generate the energy spectra and select the photopeak
    for each module. Optionally plot and save spectra
    and floodmaps.

    module_xye : Dict
                 Supermodule xyz lists for each mm
                 key is mm number (1--),
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
            bin_edges, bin_vals = hist1d(ax, module_xye[j+1]['energy'], label=f'Det: {sm_label}\n mM: {j+1}')
            try:
                bcent, gvals, pars, cov = fit_gaussian(bin_vals, bin_edges, cb=6)
                minE, maxE = pars[1] - 2 * pars[2], pars[1] + 2 * pars[2]
            except RuntimeError:
                minE, maxE = 0, 300
            eng_arr = np.array(module_xye[j+1]['energy'])
            photo_peak.append(lambda eng_val: (eng_val > minE) & (eng_val < maxE))
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
        fig.savefig(plot_output.replace(".ldat","_EnergyModuleSMod" + str(sm_label) + ".png"))
        plt.clf()
        plt.hist2d(xfilt, yfilt, bins = 500, range=[[0, 104], [0, 104]], cmap="Reds", cmax=250)
        plt.xlabel('X position (pixelated) [mm]')
        plt.ylabel('Y position (monolithic) [mm]')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(plot_output.replace(".ldat","_FloodModule" + str(sm_label) + ".png"))
        plt.clf()
    else:
        for j in range(1, 17):
            bin_vals, bin_edges = np.histogram(module_xye[j]['energy'], bins=200, range=(0, 300))
            try:
                *_, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6)
                minE, maxE = pars[1] - 2 * pars[2], pars[1] + 2 * pars[2]
            except RuntimeError:
                minE, maxE = 0, 300
            photo_peak.append(lambda eng_val: (eng_val > minE) & (eng_val < maxE))
    return photo_peak


def group_times(filtered_events, peak_select, eng_ch, time_ch, ref_supermod):
    """
    Group the first time signals for each slab
    in a reference super module with all slabs
    in other supermodules.
    filered_events: List of tuple of coincidence lists
                    The information for each event in structure
                    ([[id, mm, tstp, eng], ...], [...])
    peak_select   : List of Lists
                    Energy filter functions for each minimodule
    eng_ch        : Set
                    Set of all channels for energy measurement.
    time_ch       : Set
                    Set of all channels for time measurement.
    ref_supermod  : int
                    Index (0 or 1) of the reference SM.
    returns
    A dictionary of tuples (coinc id, coinc tstp, ref tstp)
    with key ref id.
    """
    reco_dt  = {}
    coinc_sm = 0 if ref_supermod == 1 else 1
    min_ch   = [0, 0]
    for sm1, sm2 in filtered_events:
        mm1   = sm1[0][1]
        _, e1 = get_supermodule_eng(sm1, eng_ch)
        mm2   = sm2[0][1]
        _, e2 = get_supermodule_eng(sm2, eng_ch)
        if peak_select[0][mm1-1](e1) and peak_select[1][mm2-1](e2):
            try:
                min_ch[0] = next(filter(lambda x: x[0] in time_ch, sm1))
            except StopIteration:
                continue
            try:
                min_ch[1] = next(filter(lambda x: x[0] in time_ch, sm2))
            except StopIteration:
                continue
            ## Want to know the two channel numbers and timestamps.
            try:
                # Key is the reference channel/Slab, each has a
                # list of lists with [ch_other, t_other, t_ref]
                reco_dt[min_ch[ref_supermod][0]].append([min_ch[coinc_sm    ][0],
                                                         min_ch[coinc_sm    ][2],
                                                         min_ch[ref_supermod][2]])
            except KeyError:
                reco_dt[min_ch[ref_supermod][0]] = [[min_ch[coinc_sm    ][0],
                                                     min_ch[coinc_sm    ][2],
                                                     min_ch[ref_supermod][2]]]
    return reco_dt