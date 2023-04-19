from typing import Callable

import matplotlib.pyplot as plt

from . fits import fit_gaussian
from . util import get_supermodule_eng
from . util import np
from . util import pd
from . util import ChannelType
from . util import select_energy_range
from . util import select_max_energy
from . util import shift_to_centres

def plot_settings() -> None:
    plt.rcParams[ 'lines.linewidth' ] =  2
    plt.rcParams[ 'font.size'       ] = 11
    plt.rcParams[ 'axes.titlesize'  ] = 19
    plt.rcParams[ 'axes.labelsize'  ] = 16
    plt.rcParams[ 'ytick.major.pad' ] = 14
    plt.rcParams[ 'xtick.major.pad' ] = 14
    plt.rcParams[ 'legend.fontsize' ] = 11

def hist1d(axis    : plt.Axes                  ,
           data    : np.ndarray                ,
           bins    : int             = 200     ,
           range   : tuple[int, int] = (0, 300),
           histtype: str             = 'step'  ,
           label   : str             = 'histo'
           ) -> tuple[np.ndarray, np.ndarray]:
    """
    plot a 1d histogram and return its
    """
    weights, pbins, _ = axis.hist(data, bins=bins, range=range, histtype=histtype, label=label)
    return pbins, weights


def sm_floodmaps(setup   : str        = 'tbpet'                 ,
                 out_base: str        = 'maps.ldat'             ,
                 min_peak: int        = 100                     ,
                 nsigma  : int        =   2                     ,
                 xbins   : np.ndarray = np.linspace(0, 104, 200),
                 ybins   : np.ndarray = np.linspace(0, 104, 200),
                 ebins   : np.ndarray = np.linspace(0, 300, 200),
                 log     : bool       = False                   ,
                 cmap    : str        = 'Reds'
                 ) -> Callable:
    """
    Plot energy spectra for each minimodule
    and a floodmap for the supermodule based
    on the peak selection in each of the spectra
    using the setup and binning set and a 3D (XYE)
    histogram for each minimodule.

    setup    : str
               Name of the setup: tbpet, ebrain.
    out_base : str
               The output base name for
               the plots. When None, no plots made.
    min_peak : int
               Minimum entries in peak bin for fit.
    nsigma   : int
               Number of sigmas in peak selection.
    xbins    : np.ndarray
               x axis bin edges
    ybins    : np.ndarray
               y axis bin edges
    ebins    : np.ndarray
               energy axis bin edges
    log      : bool
               Log scale for floodmap
    cmap     : str
               Floodmap color scale
    return
                 Callable for SM level plotting.
    """
    if   setup == 'tbpet' :
        nrow  = 4
        ncol  = 4
        psize = (15, 15)
    elif setup == 'ebrain':
        nrow  = 8
        ncol  = 2
        psize = (20, 15)
    else:
        print('Setup not found, defaulting to tbpet')
        nrow    = 4
        ncol    = 4
        psize   = (15, 15)
    flood_extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
    flood_norm   = 'linear'
    if log:
        flood_norm = 'log'
    plot_settings()
    def _make_plot(sm_label: int, module_xye: dict) -> None:
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=psize)
        flmap     = np.zeros((xbins.size - 1, ybins.size - 1), np.uint)
        for j, ax in enumerate(axes.flatten()):
            if not module_xye[j].any():
                print(f'No data for super module {sm_label}, mini module {j}, skipping')
                continue
            espec = module_xye[j].sum(axis=(0, 1))
            ax.plot(shift_to_centres(ebins), espec, label=f'Det: {sm_label}\n mM: {j}')
            try:
                bcent, gvals, pars, _ = fit_gaussian(espec, ebins,
                                                     cb=6, min_peak=min_peak, pk_finder='peak')
                minE, maxE = pars[1] - nsigma * pars[2], pars[1] + nsigma * pars[2]
                ax.plot(bcent, gvals, label=f'fit mu = {round(pars[1], 3)},  sigma = {round(pars[2], 3)}')
            except RuntimeError:
                minE, maxE = 0, 300
            ax.set_xlabel('Energy (au)')
            ## Filters for floodmaps
            ax.axvspan(minE, maxE, facecolor='#00FF00' , alpha = 0.3, label='Selected range')
            ax.legend()
            eslice = slice(np.searchsorted(ebins, minE, side='right') - 1,
                           np.searchsorted(ebins, maxE, side='right')    )
            flmap  = np.add(flmap, module_xye[j][:, :, eslice].sum(axis=2))
        out_name = out_base.replace(".ldat","_EnergyModuleSMod" + str(sm_label) + ".png")
        fig.savefig(out_name)
        plt.clf()

        plt.imshow(flmap.T, cmap=cmap, interpolation='none',
                   origin='lower', extent=flood_extent, norm=flood_norm)
        plt.xlabel('X position (pixelated) [mm]')
        plt.ylabel('Y position (monolithic) [mm]')
        plt.colorbar()
        plt.tight_layout()
        out_name = out_base.replace(".ldat","_FloodModule" + str(sm_label) + ".png")
        plt.savefig(out_name)
        plt.clf()
        plt.close('all')
    return _make_plot


def mm_energy_spectra(setup      : str             = 'tbpet' ,
                      plot_output: str | None      = None    ,
                      min_peak   : int             = 150     ,
                      brange     : tuple[int, int] = (0, 300),
                      nsigma     : int             = 2
                      ) -> Callable:
    """
    Generate the energy spectra and select the photopeak
    for each module. Optionally plot and save spectra
    and floodmaps.

    setup       : str
                  Name of the setup: tbpet, ebrain.
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
    if   setup == 'tbpet' :
        nrow    = 4
        ncol    = 4
        psize   = (15, 15)
        flbins  = 500
        flrange = [[0, 104], [0, 104]]
    elif setup == 'ebrain':
        nrow  = 8
        ncol  = 2
        psize = (20, 15)
        flbins  = 500
        flrange = [[0, 52], [0, 208]]
    elif plot_output:
        print('Setup not found, defaulting to tbpet')
        nrow    = 4
        ncol    = 4
        psize   = (15, 15)
        flbins  = 500
        flrange = [[0, 104], [0, 104]]
    def _make_plot(sm_label: int, module_xye: dict) -> list:
        """
        module_xye  : Dict
                      Supermodule xyz lists for each mm
                      key is mm number (0--),
        sm_label    : int
                      A label for the plots of which SM
                      is being processed.
        """
        photo_peak = []
        if plot_output:
            plot_settings()
            fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=psize)
            xfilt     = None
            yfilt     = None
            for j, ax in enumerate(axes.flatten()):
                ## mmini-module numbers start at 1
                try:
                    bin_edges, bin_vals = hist1d(ax, module_xye[j]['energy'],
                                                 range=brange, label=f'Det: {sm_label}\n mM: {j}')
                except KeyError:
                    print(f'No data for super module {sm_label}, mini module {j}, skipping')
                    photo_peak.append(lambda x: False)
                    continue
                try:
                    bcent, gvals, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6, min_peak=min_peak)
                    minE, maxE = pars[1] - nsigma * pars[2], pars[1] + nsigma * pars[2]
                    ax.plot(bcent, gvals, label=f'fit mu = {round(pars[1], 3)},  sigma = {round(pars[2], 3)}')
                except RuntimeError:
                    minE, maxE = 0, 300
                eng_arr = np.array(module_xye[j]['energy'])
                photo_peak.append(select_energy_range(minE, maxE))
                ax.set_xlabel('Energy (au)')
                ## Filters for floodmaps
                ax.axvspan(minE, maxE, facecolor='#00FF00' , alpha = 0.3, label='Selected range')
                ax.legend()
                ## Filter the positions
                if xfilt is not None:
                    xfilt = np.hstack((xfilt, np.array(module_xye[j]['x'])[photo_peak[-1](eng_arr)]))
                    yfilt = np.hstack((yfilt, np.array(module_xye[j]['y'])[photo_peak[-1](eng_arr)]))
                else:
                    xfilt = np.array(module_xye[j]['x'])[photo_peak[-1](eng_arr)]
                    yfilt = np.array(module_xye[j]['y'])[photo_peak[-1](eng_arr)]
            ## Temporary for tests.
            out_name = plot_output.replace(".ldat","_EnergyModuleSMod" + str(sm_label) + ".png")
            fig.savefig(out_name)
            plt.clf()
            plt.hist2d(xfilt, yfilt, bins=flbins, range=flrange, cmap="Reds", cmax=250)
            plt.xlabel('X position (pixelated) [mm]')
            plt.ylabel('Y position (monolithic) [mm]')
            plt.colorbar()
            plt.tight_layout()
            out_name = plot_output.replace(".ldat","_FloodModule" + str(sm_label) + ".png")
            plt.savefig(out_name)
            plt.clf()
        else:
            for j in range(nrow * ncol):
                try:
                    bin_vals, bin_edges = np.histogram(module_xye[j]['energy'], bins=200, range=brange)
                except KeyError:
                    print(f'No data for super module {sm_label}, mini module {j}, skipping')
                    photo_peak.append(lambda x: False)
                    continue
                try:
                    *_, pars, _ = fit_gaussian(bin_vals, bin_edges, cb=6, min_peak=min_peak)
                    minE, maxE = pars[1] - nsigma * pars[2], pars[1] + nsigma * pars[2]
                except RuntimeError:
                    minE, maxE = 0, 300
                photo_peak.append(select_energy_range(minE, maxE))
        return photo_peak
    return _make_plot


def slab_energy_spectra(slab_xye   : dict                              ,
                        plot_output: str        = None                 ,
                        min_peak   : int        = 150                  ,
                        bins       : np.ndarray = np.arange(9, 25, 0.2)
                        ) -> dict:
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
                plt.plot(bcent, gvals, label=f'fit mu = {round(pars[1], 3)},  sigma = {round(pars[2], 3)}')
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


def group_times(filtered_events: list    ,
                peak_select    : list    ,
                mm_map         : Callable,
                ref_indx       : int
                ) -> dict:
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
                min_ch[0] = select_max_energy(sm1, ChannelType.TIME)
            except ValueError:
                continue
            try:
                min_ch[1] = select_max_energy(sm2, ChannelType.TIME)
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


def group_times_slab(filtered_events: list,
                     peak_select    : list,
                     ref_indx       : int
                     ) -> dict:
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
            min_ch[0] = select_max_energy(sm1, ChannelType.TIME)
        except ValueError:
            continue
        try:
            min_ch[1] = select_max_energy(sm2, ChannelType.TIME)
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


def group_times_list(filtered_events: list,
                     peaks          : list,
                     ref_indx       : int
                     ) -> list:
    """
    Find the time channel with highest charge
    for each side of every event in
    filtered_events.
    """
    coinc_indx = 0 if ref_indx == 1 else 1
    def get_times(evt):
        try:
            chns = list(map(select_max_energy, evt, [ChannelType.TIME] * 2))
        except ValueError:
            return
        if peaks[0][chns[0][0]](chns[0][3]) and peaks[1][chns[1][0]](chns[1][3]):
            return [chns[  ref_indx][0], chns[coinc_indx][0],
                    chns[coinc_indx][2], chns[  ref_indx][2]]

    return list(filter(lambda x: x, map(get_times, filtered_events)))


def corrected_time_difference(impact_sel: Callable[[   tuple], tuple],
                              ref_ch    : Callable[[   tuple], int  ],
                              energy_sel: Callable[[   float], bool ],
                              geom_dt   : Callable[[int, int], float]
                              ) -> Callable:
    """
    Calculate the time stamp difference of the
    two impacts corrected for the expectation
    given the source position.

    impact_select : Callable
                    Calibration/selection function
    ref_ch        : Callable
                    Get which 'side' of the event
                    contains the reference channel.
    energy_sel    : Callable
                    True if channel energy in
                    expected charge range.
    geom_dt       : Callable
                    Calculates the geometric dt for
                    the two channels.
    """
    def _correct_dt(evt: tuple[list]) -> tuple[int, int, float]:
        sel_sms = impact_sel(evt)
        chns    = list(map(select_max_energy, sel_sms, [ChannelType.TIME] * 2))
        if any(ch is None for ch in chns) or not all(energy_sel(ch[3]) for ch in chns):
            return
        ref_indx   = ref_ch(chns)
        coinc_indx = 0 if ref_indx == 1 else 1
        ref_id     = chns[  ref_indx][0]
        coinc_id   = chns[coinc_indx][0]
        dt_geom    = geom_dt(ref_id, coinc_id)
        if dt_geom is None:
            # No overlap found, ignore.
            return
        dt_tstp    = chns[ref_indx][2] - chns[coinc_indx][2]
        return ref_id, coinc_id, dt_tstp - dt_geom
    return _correct_dt


def ctr(eselect: Callable                        ,
        skew   : pd.Series=pd.Series(dtype=float)
        ) -> Callable:
    """
    CTR function.
    Returns a function that calculates the timestamp
    difference for a coincidence event given the
    energy selection (eselect) and skew correction values (skew)
    """
    def timestamp_difference(evt: tuple[list]) -> float:
        chns = [chn for sm in evt if (chn := select_max_energy(sm, ChannelType.TIME))]
        if len(chns) == 2 and eselect(chns[0][3]) and eselect(chns[1][3]):
            skew_corr = skew.get(chns[1][0], 0.0) - skew.get(chns[0][0], 0.0)
            return chns[0][2] - chns[1][2] + skew_corr
        return
    return timestamp_difference


class ChannelEHistograms:
    def __init__(self, tbins: np.ndarray, ebins: np.ndarray, esum_bins: np.ndarray) -> None:
        self.edges      = {ChannelType.TIME  :     tbins,
                           ChannelType.ENERGY:     ebins,
                           'ESUM'            : esum_bins}
        self.nbin_time  = len(tbins)     - 1
        self.nbin_eng   = len(ebins)     - 1
        self.nbin_sum   = len(esum_bins) - 1
        self.tdist      = {}
        self.edist      = {}
        self.sum_dist   = {}
        self.underflow  = {}
        self.overflow   = {}

    def add_overflow(self, id: int) -> None:
        try:
            self.overflow[id] += 1
        except KeyError:
            self.overflow[id]  = 1

    def add_underflow(self, id: int) -> None:
        try:
            self.underflow[id] += 1
        except KeyError:
            self.underflow[id]  = 1

    @staticmethod
    def __fill_histo(id: int, indx: int, hist_dict: dict, nbins: int) -> None:
        try:
            hist_dict[id][indx] += 1
        except KeyError:
            hist_dict[id]        = np.zeros(nbins, int)
            hist_dict[id][indx] += 1

    def __get_bin(self, id: int, type: ChannelType | str, eng: float) -> int | None:
        if eng >= self.edges[type][-1]:
            self.add_overflow(id)
            return
        bin_indx = np.searchsorted(self.edges[type], eng, side='right') - 1
        if bin_indx < 0:
            self.add_underflow(id)
            return
        return bin_indx

    def fill_time_channel(self, impact: list) -> None:
        bin_indx = self.__get_bin(impact[0], impact[1], impact[3])
        if bin_indx is not None:
            self.__fill_histo(impact[0], bin_indx, self.tdist, self.nbin_time)

    def fill_energy_channel(self, impact: list) -> None:
        bin_indx = self.__get_bin(impact[0], impact[1], impact[3])
        if bin_indx is not None:
            self.__fill_histo(impact[0], bin_indx, self.edist, self.nbin_eng)

    def fill_esum(self, sm_impacts: list, id_val: int) -> None:
        ## Needs to be sorted for new types and improved.
        efilt    = filter(lambda y: y[1] is ChannelType.ENERGY, sm_impacts)
        esum     = sum(map(lambda x: x[3], efilt))
        bin_indx = self.__get_bin(id_val, 'ESUM', esum)
        self.__fill_histo(id_val, bin_indx, self.sum_dist, self.nbin_sum)

    # Used for channel equalization/peak value plots
    # Will normally be used with singles but general just in case
    def add_emax_evt(self, evt: tuple[list, list]) -> None:
        # time channels
        tchn_map = map(select_max_energy, evt, [ChannelType.TIME]*2)
        chns = list(filter(lambda x: x, tchn_map))
        for slab in chns:
            self.fill_time_channel(slab)

        # energy channels
        if chns:
            echn_map = map(select_max_energy, evt, [ChannelType.ENERGY]*2)
            for echn in filter(lambda x: x, echn_map):
                self.fill_energy_channel(echn)

    def add_all_energies(self, evt: tuple[list, list], sms: list[tuple]) -> None:
        """
        Time channel energy plots for all occurences and
        energy sums for each minimodule.
        """
        for imps, sm_mm in zip(evt, sms):
            # time channels
            for imp in filter(lambda i: i[1] is ChannelType.TIME, imps):
                self.fill_time_channel(imp)

            self.fill_esum(imps, sm_mm)
