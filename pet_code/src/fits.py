from random import gauss
from scipy.optimize import curve_fit

from . util import np
from . util import shift_to_centres


def gaussian(x, amp, mu, sigma):
    if sigma <= 0.:
        return np.inf
    return amp * np.exp(-0.5 * (x - mu)**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)


def fit_gaussian(data, bins, cb=8):
    """
    Tidy of existing function.
    Probably want to generalise so
    other functions could be used (Gaus+poly?)

    cb: number of bins +- from max used in fit
    """
    bin_centres = shift_to_centres(bins)

    # Define limits around maximum.
    max_indx  = np.argmax(data)
    first_bin = max(max_indx - cb, 0)
    last_bin  = min(max_indx + cb, len(bin_centres))
    x         = bin_centres[first_bin:last_bin]
    y         = data[first_bin:last_bin]
    if sum(y) <= 0:
        raise RuntimeError('No useful data available.')

    ## Initial values
    mu0 , wsum = np.average(x           , weights=y, returned=True)
    sig0       = np.average((x - mu0)**2, weights=y)
    if wsum > 1:
        sig0 *= wsum / (wsum - 1)

    pars, pcov = curve_fit(gaussian, x, y, p0=[wsum, mu0, sig0])
    return bin_centres, gaussian(bin_centres, *pars), pars, pcov
