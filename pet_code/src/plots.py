import matplotlib.pyplot as plt

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