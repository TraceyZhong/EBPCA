# a function to plot a series of boxplots along x axis

import matplotlib.pyplot as plt
import numpy as np
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plot_boxplot_series(stats, ax, color_panel = ['tab:red', 'tab:grey'],
                        sep_width = [-0.15, 0.15], labels = ['iter NPMLE off', 'iter NPMLE on'],
                        title = 'Estimation error (subspace distance)'):
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['font.size'] = 18

    n_groups = len(stats)
    n_iters = len(stats[0])
    colors = color_panel[:n_groups]

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), constrained_layout=True)
    # plot boxplots with specified separation widths and colors
    bp_list = []
    for i in range(n_groups):
        bp_list.append(ax.boxplot(stats[i], positions=np.array([j + sep_width[i] for j in range(1, n_iters + 1, 1)]),
                                  widths=0.2))
        set_box_color(bp_list[i], colors[i])
        # add medians to each boxplot
        ax.plot([j + sep_width[i] for j in range(1, n_iters+1, 1)],
                [np.median(st) for st in stats[i]], '-', alpha=0.5,  c=colors[i], label=labels[i])
        ax.legend()
        ax.set_xticks([i for i in range(1, n_iters+1, 1)])
        ax.set_xticklabels([i for i in range(0, n_iters+1, 1)])
        ax.set_ylabel('Subspace distance')
        ax.set_xlabel('Iteration')
        ax.set_title(title)
    return ax
