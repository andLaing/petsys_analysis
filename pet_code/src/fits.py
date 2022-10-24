from scipy.optimize import curve_fit

from . util import np
from . util import shift_to_centres


def gaussian(x, amp, mu, sigma):
    if sigma <= 0.:
        return np.inf
    return amp * np.exp(-0.5 * (x - mu)**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)


def fit_gaussian(data, bins, cb=8, min_peak=150):
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
    mu0, wsum, x, y = mean_around_max(data, bin_centres, cb)
    if mu0 is None:
        raise RuntimeError('No useful data available.')

    ## Initial values
    sig0 = np.average((x - mu0)**2, weights=y)
    if wsum > 1:
        sig0 *= wsum / (wsum - 1)

    pars, pcov = curve_fit(gaussian, x, y, p0=[wsum, mu0, sig0])
    return bin_centres, gaussian(bin_centres, *pars), pars, pcov


def mean_around_max(data, bins, cb):
    max_indx  = np.argmax(data)
    first_bin = max(max_indx - cb, 0)
    last_bin  = min(max_indx + cb, len(bins))
    x         = bins[first_bin:last_bin]
    y         = data[first_bin:last_bin]
    if sum(y) <= 0:
        return None, None, None, None
    return *np.average(x, weights=y, returned=True), x, y
