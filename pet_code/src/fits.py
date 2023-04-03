from scipy.optimize import curve_fit
from scipy.signal   import find_peaks
from typing         import Callable

from . util import np
from . util import shift_to_centres


def gaussian(x    : float | np.ndarray,
             amp  : float             ,
             mu   : float             ,
             sigma: float
             ) -> float | np.ndarray:
    if sigma <= 0.:
        return np.inf
    return amp * np.exp(-0.5 * (x - mu)**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)


def lorentzian(x    : float | np.ndarray,
               amp  : float             ,
               x0   : float             ,
               gamma: float
               ) -> float | np.ndarray:
    if gamma <= 0:
        return np.inf
    return amp * gamma**2 / ((x - x0)**2 + gamma**2)


def fit_gaussian(data     : np.ndarray              ,
                 bins     : np.ndarray              ,
                 cb       : int               =   8 ,
                 min_peak : int               = 150 ,
                 yerr     : np.ndarray | None = None,
                 pk_finder: str = 'max'
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tidy of existing function.
    Probably want to generalise so
    other functions could be used (Gaus+poly?)

    cb: number of bins +- from max used in fit
    """
    bin_centres = shift_to_centres(bins)

    # Define limits around maximum.
    if data[np.argmax(data)] < min_peak:
        raise RuntimeError('Peak max below requirement.')

    if 'peak' in pk_finder:
        max_val  = data.max()
        peaks, _ = find_peaks(data                      ,
                              height     = max_val  /  2,
                              distance   = cb           ,
                              prominence = min_peak // 2,
                              width      = cb       // 2)
        if peaks.shape[0] == 0:
            mu0, wsum, x, y, err = mean_around_max(data, bin_centres, cb, yerr)
        else:
            pk_indx  = peaks.max()# Use the peak furthest to the right
            mu0      = bin_centres[pk_indx]
            min_indx = max(pk_indx - cb, 0)
            max_indx = min(pk_indx + cb, len(bin_centres))
            x        = bin_centres[min_indx:max_indx]
            y        = data       [min_indx:max_indx]
            wsum     = y.sum()
            if yerr is None:
                err = np.sqrt(y, out=np.abs(y).astype('float'), where=y>=0)
            else:
                err = yerr[min_indx:max_indx]
    else:
        mu0, wsum, x, y, err = mean_around_max(data, bin_centres, cb, yerr)

    if mu0 is None:
        raise RuntimeError('No useful data available.')

    ## Initial values
    sig0 = np.average((x - mu0)**2, weights=y)
    if wsum > 1:
        sig0 *= wsum / (wsum - 1)
    sig0 = np.sqrt(sig0)

    pars, pcov = curve_fit(gaussian, x, y, sigma=err, p0=[wsum, mu0, sig0])
    return bin_centres, gaussian(bin_centres, *pars), pars, pcov


def curve_fit_fn(fn  : Callable  ,
                 x   : np.ndarray,
                 y   : np.ndarray,
                 yerr: np.ndarray,
                 p0  : list
                 ) -> tuple[np.ndarray, np.ndarray]:
    pars, pcov = curve_fit(fn, x, y, sigma=yerr, p0=p0)
    return pars, pcov


def mean_around_max(data: np.ndarray     ,
                    bins: np.ndarray     ,
                    cb  : int            ,
                    yerr: np.ndarray=None
                    ) -> tuple:
    max_indx  = np.argmax(data)
    first_bin = max(max_indx - cb, 0)
    last_bin  = min(max_indx + cb, len(bins))
    x         = bins[first_bin:last_bin]
    y         = data[first_bin:last_bin]
    if sum(y) <= 0:
        return None, None, None, None
    if yerr is not None:
        yerr = yerr[first_bin:last_bin]
    else:
        yerr = np.sqrt(y, out=np.abs(y).astype('float'), where=y>=0)
    return *np.average(x, weights=y, returned=True), x, y, yerr
