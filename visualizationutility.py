import numpy as np
import datautility as du
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def plot_distribution(ar, breaks=None, save_file=None, title='',xlabel='',ylabel='',color='red', show=True):
    lab = None
    ar = du.nan_omit(ar)

    if du.infer_if_string(ar,100):
        ar, lab = du.as_factor(ar, return_labels=True)

    ar = np.array(ar, dtype=np.float32)

    if breaks is None:
        breaks = len(np.unique(np.round(ar, 3)))

    fig = plt.hist(ar, breaks, color=color, edgecolor='black', linewidth=0.5, alpha=.75)
    plt.title(title)
    if lab is not None:
        plt.xticks(ar, lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show:
        plt.show()
    if save_file is not None:
        plt.savefig(save_file, format='png')

if __name__ == "__main__":
    x = np.random.normal(0,10,1000)
    n, bins, patches = plt.hist(x,100,color='red', edgecolor='black', linewidth=0.5, alpha=.75)
    # y = mlab.normpdf(bins,0,10)
    # plt.plot(bins,y,linewidth=1)

    plt.show()