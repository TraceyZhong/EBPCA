# a function to plot a series of boxplots along x axis

import matplotlib.pyplot as plt
import numpy as np
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plot_boxplot_series(stats, color_panel = ['tab:red', 'tab:grey'],
                        sep_width = [-0.15, 0.15], labels = ['iter NPMLE off', 'iter NPMLE on']):
    n_groups = len(stats)
    n_iters = len(stats[0])
    colors = color_panel[:n_groups]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.3, 4.3), constrained_layout=True)
    # plot boxplots with specified separation widths and colors
    bp_list = []
    for i in range(n_groups):
        bp_list.append(ax.boxplot(stats[i], positions=np.array([i - sep_width[i] for i in range(1, n_iters + 2, 1)]),
                                     widths=0.2))
        set_box_color(bp_list[i], colors[i])
        # add medians to each boxplot
        ax.plot([i - sep_width[i] for i in range(1, n_iters+2, 1)],
                [np.median(st) for st in stats[i]], '-', alpha=0.5,  c=colors[i], label=labels[i])
        ax.legend()
        ax.set_xticks([i for i in range(1, n_iters+2, 1)])
        ax.set_xticklabels([i for i in range(1, n_iters+2, 1)])
        ax.set_title('Estimation error (subspace distance)')
    plt.show()
