import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.state_evolution import get_state_evolution, get_alignment_evolution, \
    uniform, point_normal, two_points

# ----------------------------------------------
# Figure 3:
# alignment comparison across methods
# boxplot
# ----------------------------------------------

def load_alignments(prior, method, PCname = 'U', rm_na=False, prefix = ''):
    prefix = prefix + '/'
    align_dir = 'output/univariate/%s/%salignments' % (prior, prefix)
    if PCname == 'U':
        align = np.load('%s/%s_u_s_%.1f_n_rep_%i.npy' % (align_dir, method, s, n_rep)).T
    else:
        align = np.load('%s/%s_v_s_%.1f_n_rep_%i.npy' % (align_dir, method, s, n_rep)).T
    if rm_na:
        align = [(align[i])[~np.isnan(align[i])] for i in range(len(align))]
    else:
        align = align.tolist()
    return align

def group_alignments(prior, PCname, rm_na=False, prefix = ''):
    res = [load_alignments(prior, method, PCname, rm_na=rm_na, prefix=prefix) for method in ['EB-PCA', 'EBMF']]
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
    # ax.set_ylim(0 - 0.05, 1 + 0.05)
    return ax

def add_line_to_boxplots(ax, res, plot_seps = [-0.5, 0.5], bp_colors = ['#fc9272', '#de2d26']):
    # evaluate medians
    y = np.nanmedian(res, axis=2)
    x = [ax.get_xticks() + plot_seps[i] for i in range(2)]
    # make labels
    labels = ['EB-PCA', 'EBMF'] #, label = labels[i]
    # Plot a line between the means of each dataset
    [plt.plot(x[i], y[i], 'b-', c=bp_colors[i]) for i in range(2)]

def make_iter_plot(prior, PCname, iters, plot_seps = [-0.5, 0.5],
                   bp_colors = ['#de2d26', '#3182bd'], suffix = '', prefix = ''):
    res_clean = group_alignments(prior, PCname, True, prefix=prefix)
    res = group_alignments(prior, PCname, False, prefix=prefix)
    ax = alignment_boxplots(res_clean, [i + 1 for i in range(iters)], plot_seps, bp_colors)
    add_line_to_boxplots(ax, res, plot_seps, bp_colors)
    plt.title('%s, %s, alignment across iterations (s=%.1f) \n %s' % (prior.replace('_', ' '), PCname, s, suffix))


if __name__ == '__main__':
    prefix = 'n_2000_gamma_2.0_nsupp_ratio_0.5_0.5_useEM_True' # 'n_1000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_True'
    suffix = prefix # 'useEM_pilot'
    n_rep = 50
    iters = 10
    gamma = 2
    mmse_funcs = {
        "Uniform": uniform,
        "Two_points": two_points,
        "Point_normal": point_normal
    }

    # Figure 3
    for s in [1.3]: #[1.1, 1.5, 2.0]
        for prior in ['Point_normal', 'Two_points', 'Uniform']:
            for PCname in ['U', 'V']:
                make_iter_plot(prior, PCname, 10, bp_colors=['tab:red', 'tab:blue'], suffix=suffix, prefix=prefix)
                se = get_state_evolution(s, gamma, mmse_funcs[prior], mmse_funcs[prior], iters)
                ae = get_alignment_evolution(se)
                if PCname == 'U':
                    plt.plot([i + 1 - 1 for i in range(0, 2 * iters, 2)], ae.ualigns[:-1],
                             c='grey', linestyle='--', label='Bayes Optimal')
                else:
                    plt.plot([i + 1 - 1 for i in range(0, 2 * iters, 2)], ae.valigns[:-1],
                             c='grey', linestyle='--', label='Bayes Optimal')
                plt.legend(loc='lower right')
                plt.savefig('figures/univariate/Figure3/%s_%s_%.1f_iterations_boxplots_%s.png' % (prior, PCname, s, suffix))
                plt.close()

    exit()
    prior = 'Two_points'
    PCname = 'V'
    s = 1.5
    make_iter_plot(prior, PCname, 10, bp_colors=['tab:red', 'tab:blue'], suffix=suffix, prefix=prefix)
    se = get_state_evolution(s, gamma, mmse_funcs[prior], mmse_funcs[prior], iters)
    ae = get_alignment_evolution(se)
    if PCname == 'U':
        plt.plot([i + 1 - 1 for i in range(0, 2 * iters, 2)], ae.ualigns[:-1],
                 c='grey', linestyle='--', label='Bayes Optimal')
    else:
        plt.plot([i + 1 - 1 for i in range(0, 2 * iters, 2)], ae.valigns[:-1],
                 c='grey', linestyle='--', label='Bayes Optimal')
    plt.legend(loc='lower right')
    plt.show()

    exit()