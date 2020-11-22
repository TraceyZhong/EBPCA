import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])

# ----------------------------------------------
# Figure 3:
# alignment comparison across methods
# boxplot
# ----------------------------------------------

def load_alignments(prior, method, PC = 'left', s = 1.3, rm_na=False):
    align_dir = 'output/univariate/%s/alignments' % prior
    if PC == 'left':
        align = np.load('%s/%s_u_s_%.1f_n_rep_50.npy' % (align_dir, method, s)).T
    else:
        align = np.load('%s/%s_v_s_%.1f_n_rep_50.npy' % (align_dir, method, s)).T
    if rm_na:
        align = [(align[i])[~np.isnan(align[i])] for i in range(len(align))]
    else:
        align = align.tolist()
    return align

def group_alignments(prior, rm_na=False):
    res = [load_alignments(prior, method, rm_na=rm_na) for method in ['EB-PCA', 'EBMF']]
    return res

def alignment_boxplots(res, ticks, plot_seps = [-0.5, 0.5], bp_colors = ['#de2d26', '#3182bd']):
    ebpca, ebmf = res
    n = 2
    bp_width = 0.3

    # reference: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots 2nd answer
    f, ax = plt.subplots(figsize=(8, 6))
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    bp1 = ax.boxplot(ebpca, positions=np.array(range(len(ebpca))) * float(n) + plot_seps[0], sym='', widths=bp_width)
    bp2 = ax.boxplot(ebmf, positions=np.array(range(len(ebmf))) * float(n) + plot_seps[1], sym='', widths=bp_width)

    set_box_color(bp1, bp_colors[0])
    set_box_color(bp2, bp_colors[1])

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c=bp_colors[0], label='EB-PCA')
    ax.plot([], c=bp_colors[1], label='EBMF')

    ax.legend(loc='lower right')
    plt.xticks(range(0, len(ticks) * n, n), ticks)
    ax.set_xlabel("iteration")
    ax.set_ylabel('alignment with ground truth')
    ax.set_xlim(-2, len(ticks) * n)
    ax.set_ylim(0 - 0.05, 1 + 0.05)
    return ax

def add_line_to_boxplots(ax, res, plot_seps = [-0.5, 0.5], bp_colors = ['#fc9272', '#de2d26']):
    # evaluate medians
    y = np.nanmedian(res, axis=2)
    x = [ax.get_xticks() + plot_seps[i] for i in range(2)]
    # Plot a line between the means of each dataset
    [plt.plot(x[i], y[i], 'b-', c=bp_colors[i], linestyle = '--') for i in range(2)]

def make_iter_plot(prior, PCname, iters, s = 1.3, plot_seps = [-0.5, 0.5], bp_colors = ['#de2d26', '#3182bd']):
    res_clean = group_alignments(prior, True)
    res = group_alignments(prior, False)
    ax = alignment_boxplots(res_clean, [i + 1 for i in range(iters)], plot_seps, bp_colors)
    add_line_to_boxplots(ax, res, plot_seps, bp_colors)
    plt.title('%s, %s, alignment across iterations (s=%.1f)' % (prior.replace('_', ' '), PCname, s))
    plt.savefig('figures/univariate/Figure1/%s_%s_iterations_boxplots.png' % (prior, PCname))
    plt.close()

if __name__ == '__main__':
    # Figure 3
    s = 1.3
    for prior in ['Point_normal', 'Two_points', 'Uniform']:
        for PCname in ['U', 'V']:
            make_iter_plot(prior, PCname, 10, s, bp_colors=['tab:red', 'tab:blue'])
