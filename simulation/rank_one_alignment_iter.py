import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.extend(['../../generalAMP'])
from ebpca.state_evolution import get_state_evolution, get_alignment_evolution, \
    uniform, point_normal, two_points, point_normal_p1, point_normal_p5, uniform_centered

# ----------------------------------------------
# Figure 3:
# alignment comparison across methods
# boxplot
# ----------------------------------------------

plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

def load_alignments(prior, method, PC = 'u', rm_na=False, prefix = '', suffix = ''):
    prefix = prefix + '/'
    align_dir = 'output/univariate/%s/%salignments/%s' % (prior, prefix, method)
    align = np.load('%s_%s_s_%.1f_n_rep_%i%s.npy' % (align_dir, PC, s, n_rep, suffix)).T
    if rm_na:
        align = [(align[i])[~np.isnan(align[i])] for i in range(len(align))]
    else:
        align = align.tolist()
    return align

def group_alignments(prior, PCname, rm_na=False, prefix = '', suffix = ''):
    res = [load_alignments(prior, method, PCname, rm_na=rm_na, prefix=prefix, suffix = suffix) \
           for method in ['EB-PCA', 'EBMF']]
    return res

def alignment_boxplots(res, ticks, plot_seps = [-0.5, 0.5], bp_colors = ['#de2d26', '#3182bd']):
    ebpca, ebmf = res
    n = 2
    bp_width = 0.3

    # reference: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots 2nd answer
    f, ax = plt.subplots(figsize=(6, 4), constrained_layout = True)
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
    ax.plot([], c=bp_colors[1], label='Mean-field VB')

    ax.legend(loc='lower right', scatterpoints=3, fontsize=20)
    plt.xticks(range(0, len(ticks) * n, n), ticks)
    ax.set_xlabel("Iteration")
    ax.set_ylabel('Accuracy')
    ax.set_xlim(-2, len(ticks) * n)
    return ax

def add_line_to_boxplots(ax, res, plot_seps = [-0.5, 0.5], bp_colors = ['#fc9272', '#de2d26']):
    # evaluate medians
    y = np.nanmedian(res, axis=2)
    x = [ax.get_xticks() + plot_seps[i] for i in range(2)]
    # make labels
    labels = ['EB-PCA', 'Mean-field VB']
    # Plot a line between the means of each dataset
    [plt.plot(x[i], y[i], 'b-', c=bp_colors[i]) for i in range(2)]

def make_iter_plot(prior, PCname, iters, plot_seps = [-0.5, 0.5],
                   bp_colors = ['#de2d26', '#3182bd'], suffix = '', prefix = '', data_suffix = ''):
    # res_clean = group_alignments(prior, PCname, True, prefix=prefix, suffix=suffix)
    res = group_alignments(prior, PCname, False, prefix=prefix, suffix=data_suffix)
    ax = alignment_boxplots(res, [i for i in range(iters + 1)], plot_seps, bp_colors)
    add_line_to_boxplots(ax, res, plot_seps, bp_colors)
    plt.title('Alignment across iterations, %s (s=%.1f)' % (PCname, s))


if __name__ == '__main__':
    prefixes = {'n_2000': 'n_2000_gamma_2.0_nsupp_ratio_0.5_0.5_useEM_True',
                'n_1000': 'n_1000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_True',
                'MOSEK_pilot': 'n_2000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_False',
                'MOSEK_exper': 'n_2000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_False'}
    n_reps = {'n_2000': 50, 'n_1000': 50, 'MOSEK_pilot': 15, 'MOSEK_exper': 50}
    priors = {'n_2000': ['Point_normal', 'Two_points', 'Uniform'],
              'n_1000': ['Point_normal', 'Two_points', 'Uniform'],
              'MOSEK_pilot': ['Point_normal_0.1', 'Point_normal_0.5',  'Two_points', 'Uniform_centered'],
              'MOSEK_exper': ['Point_normal_0.1', 'Point_normal_0.5',  'Two_points', 'Uniform_centered', 'Normal']}
    s_lists = {'n_2000': [1.1, 1.3, 1.5, 2.0],
              'n_1000': [1.1, 1.3, 1.5, 2.0],
              'MOSEK_pilot': [1.1, 1.3, 1.5],
               'MOSEK_exper': [1.1, 1.3, 1.5, 2.0]}

    exper_name = 'MOSEK_exper'
    prefix = prefixes[exper_name]
    n_rep = n_reps[exper_name]
    iters = 10
    gamma = 2
    mmse_funcs = {
        "Uniform": uniform,
        "Two_points": two_points,
        "Point_normal_0.1": point_normal_p1,
        "Point_normal_0.5": point_normal_p5,
        "Point_normal": point_normal,
        "Uniform_centered": uniform_centered
    }
    data_suffix = '_by_5'

    if not os.path.exists('figures/univariate/Figure3/%s/%s' % (prefix, exper_name)):
        os.mkdir('figures/univariate/Figure3/%s/%s' % (prefix, exper_name))

    plot_se = True

    # Figure 3
    for s in [1.3]:
        for prior in ['Uniform_centered', 'Two_points']:
            for PC in ['u', 'v']:
                make_iter_plot(prior, PC, 10, bp_colors=['tab:red', 'tab:blue'], \
                               suffix=exper_name, prefix=prefix, data_suffix=data_suffix)
                # remove Bayes optimal
                # if prior == 'Two_points':
                #     se = get_state_evolution(s, gamma, mmse_funcs[prior], mmse_funcs[prior], iters)
                #     ae = get_alignment_evolution(se)
                #     if PC == 'u':
                #         plt.plot([i + 1 - 1 for i in range(0, 2 * (iters + 1), 2)], ae.ualigns,
                #                  c='grey', linestyle='--', label='Bayes Optimal')
                #     else:
                #         plt.plot([i + 1 - 1 for i in range(0, 2 * (iters + 1), 2)], ae.valigns,
                #                  c='grey', linestyle='--', label='Bayes Optimal')
                plt.legend(loc='lower right', fontsize=15)
                plt.savefig('figures/univariate/Figure3/%s/%s/%s_%s_%.1f.png' % \
                            (prefix, exper_name, prior, PC, s))
                plt.close()

    exit()
