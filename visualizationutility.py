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
        plt.savefig(save_file, figsize=(8,6), dpi=80, format='png')

    plt.close()



def __get_cmap(n, name='gist_rainbow'):
    # https: // stackoverflow.com / questions / 14720331 / how - to - generate - random - colors - in -matplotlib
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_lines(ar, labels=None, save_file=None, title='', xlabel='', ylabel='', xticks=None, show=True):

    cmap = __get_cmap(len(ar), 'tab10' if len(ar) <= 10 else 'gist_rainbow')

    lines = dict()
    ax = plt.subplot(111)
    lab = labels

    show_legend = True
    if labels is None:
        if type(ar) is dict:
            lab = [str(i) for i in ar]
        else:
            lab = ['' for _ in range(len(ar))]
            show_legend = False
    j = 0
    for i in ar:
        if type(ar) is dict:
            lines[j], = ax.plot(list(range(len(ar[i]))), ar[i], color=cmap(j), label=lab[j])
        else:
            lines[j], = ax.plot(list(range(len(i))), i, color=cmap(j), label=lab[j])
        j += 1

    if show_legend:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

    plt.title(title)
    if xticks is not None:
        plt.xticks(list(range(len(xticks))), xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show:
        plt.show()
    if save_file is not None:
        plt.savefig(save_file, figsize=(8, 6), dpi=80, format='png')

    plt.close()

if __name__ == "__main__":
    x = np.random.normal(0,10,1000)
    n, bins, patches = plt.hist(x,100,color='red', edgecolor='black', linewidth=0.5, alpha=.75)
    # y = mlab.normpdf(bins,0,10)
    # plt.plot(bins,y,linewidth=1)

    plt.show()