import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
sys.path.extend(['../../generalAMP'])
from ebpca.state_evolution import get_state_evolution, uniform, point_normal, two_points
# ----------------------------------------------
# Figure 1:
# alignment comparison across methods
# boxplot
# ----------------------------------------------

def alignment_boxplots(res, ticks):
    n = len(res)
    if n == 5:
        pca, bayesamp, ebpca, ebmf, spca = res
        plot_seps = [-1.2, -0.6, 0.0, 0.6, 1.2]
        bp_width = 0.45
    else:
        pca, bayesamp, ebpca, ebmf = res
        n = 4
        plot_seps = [-1.2, -0.4, 0.4, 1.2]
        bp_width = 0.45
    bp_colors = ['tab:grey', 'tab:orange', 'tab:red', 'tab:blue', 'tab:green']
    # reference: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots 2nd answer
    f, ax = plt.subplots(figsize=(8, 6))
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    bp1 = ax.boxplot(pca, positions=np.array(range(len(pca))) * float(n) + plot_seps[0], sym='', widths=bp_width)
    bp2 = ax.boxplot(bayesamp, positions=np.array(range(len(bayesamp))) * float(n) + plot_seps[1], sym='', widths=bp_width)
    bp3 = ax.boxplot(ebpca, positions=np.array(range(len(ebpca))) * float(n) + plot_seps[2], sym='', widths=bp_width)
    bp4 = ax.boxplot(ebmf, positions=np.array(range(len(ebmf))) * float(n) + plot_seps[3], sym='', widths=bp_width)
    if n == 5:
        bp5 = ax.boxplot(spca, positions=np.array(range(len(spca))) * float(n) + plot_seps[4], sym='', widths=bp_width)

    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    set_box_color(bp1, bp_colors[0])
    set_box_color(bp2, bp_colors[1])
    set_box_color(bp3, bp_colors[2])
    set_box_color(bp4, bp_colors[3])
    if n == 5:
        set_box_color(bp5, bp_colors[4])  # colors are from http://colorbrewer2.org/

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c=bp_colors[0], label='PCA')
    ax.plot([], c=bp_colors[1], label='Oracle Bayes AMP')
    ax.plot([], c=bp_colors[2], label='EB-PCA')
    ax.plot([], c=bp_colors[3], label='EBMF')
    if n == 5:
        ax.plot([], c=bp_colors[4], label='sPCA')

    ax.legend(loc='lower right')
    plt.xticks(range(0, len(ticks) * n, n), ticks)
    ax.set_xlabel("signal strength")
    ax.set_ylabel('alignment with ground truth')
    ax.set_xlim(-2, len(ticks) * n)
    ax.set_ylim(np.min([np.min(res[i]) for i in range(len(res))]) - 0.05, 1 + 0.05)
    return ax
# os.chdir('/Users/chang/PycharmProjects/generalAMP/simulation/')

# load alignments from 50 replications
def load_alignments(prior, method, s_list, ind=-1, PC='u', rm_na =True, \
                    n_rep=50, prefix = '', suffix = '_by_5'):
    n_par = len(s_list)
    prefix = prefix + '/'
    align_dir = 'output/univariate/%s/%salignments/%s_%s' % (prior, prefix, method, PC)
    n_rep_suffix = 'n_rep_%i%s' % (n_rep, suffix)

    if method == 'spca':
        aligns = [(pd.read_table('%s_s_%.1f_%s.txt' % (align_dir, s, n_rep_suffix)).values.reshape(-1)) \
            for s in s_list]
    else:
        # take the result from the last iterate
        aligns = [(np.load('%s_s_%.1f_%s.npy' % (align_dir, s, n_rep_suffix))[:, ind]) for s in s_list]

    if rm_na:
        # remove nan values:
        print('\n %s, %s\n' % (prior, method))
        print('Number of NA alignments for %s, %s' % (method, prior), [np.sum(np.isnan(aligns[i])) for i in range(n_par)])
        aligns = [np.array(aligns[i])[~np.isnan(aligns[i])] for i in range(n_par)]
    else:
        aligns = np.array(aligns)
    return aligns

def group_alignments(prior, PC, rm_na = True, s_list=[1.1, 1.3, 1.5, 2.0], n_rep=50, prefix='', suffix=''):
    pca = load_alignments(prior, 'EB-PCA', s_list, 0, PC, rm_na, n_rep, prefix, suffix)
    bayesamp = load_alignments(prior, 'BayesAMP', s_list, -1, PC, rm_na, n_rep, prefix, suffix)
    ebpca = load_alignments(prior, 'EB-PCA', s_list, 5, PC, rm_na, n_rep, prefix, suffix)
    ebmf = load_alignments(prior, 'EBMF', s_list, -1, PC, rm_na, n_rep, prefix, suffix)
    if prior == 'Point_normal_0.1':
        spca = load_alignments(prior, 'spca', s_list, -1, PC, rm_na, n_rep, prefix, suffix)
        res = [pca, bayesamp, ebpca, ebmf, spca]
    else:
        res = [pca, bayesamp, ebpca, ebmf]
    return res

def eval_se(prior, s_list, gamma, iters):
    mmse_funcs = {
        "Uniform": uniform,
        "Two_points": two_points,
        "Point_normal": point_normal
    }
    se = [get_state_evolution(s, gamma, mmse_funcs[prior], mmse_funcs[prior], iters) for s in s_list]
    return se

# prior = 'Point_normal' # 'Two_points' # 'Uniform'
def make_comp_plot(res, prior, PC, s_list, to_save=True, prefix = '', suffix=''):
    alignment_boxplots(res, s_list)
    # locs, labels = plt.xticks()
    # plt.hlines(se, locs-0.5, locs+0.5)
    plt.title('%s, %s, method comparison \n %s' % (prior.replace('_', ' '), PC, suffix))
    if to_save:
        print('figures/univariate/Figure1/%s/%s_%s_%s.png' % (prefix, suffix, prior, PC))
        plt.savefig('figures/univariate/Figure1/%s/%s/%s_%s.png' % (prefix, suffix, prior, PC))
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    # Figure 1
    # res = group_alignments('Point_normal', 'V', s_list=[1.3, 1.4, 1.5, 1.6, 3.0])
    # make_comp_plot(res, 'Point_normal', 'V', s_list=[1.3, 1.4, 1.5, 1.6, 3.0])

    prefixes = {'n_2000': 'n_2000_gamma_2.0_nsupp_ratio_0.5_0.5_useEM_True',
                'n_1000': 'n_1000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_True',
                'MOSEK_pilot': 'n_2000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_False',
                'MOSEK_exper': 'n_2000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_False'}
    n_reps = {'n_2000': 50, 'n_1000': 50, 'MOSEK_pilot': 15, 'MOSEK_exper': 50}
    priors = {'n_2000': ['Point_normal', 'Two_points', 'Uniform'],
              'n_1000': ['Point_normal', 'Two_points', 'Uniform'],
              'MOSEK_pilot': ['Point_normal_0.1', 'Point_normal_0.5', 'Two_points', 'Uniform_centered'],
              'MOSEK_exper': ['Point_normal_0.1', 'Point_normal_0.5', 'Two_points', 'Uniform_centered', 'Normal']}
    s_lists = {'n_2000': [1.1, 1.3, 1.5, 2.0],
               'n_1000': [1.1, 1.3, 1.5, 2.0],
               'MOSEK_pilot': [1.1, 1.3, 1.5, 2.0],
               'MOSEK_exper': [1.1, 1.3, 1.5, 2.0]}

    exper_name = 'MOSEK_exper'
    prefix = prefixes[exper_name]
    s_list = s_lists[exper_name]
    n_rep = n_reps[exper_name]
    gamma = 2
    iters = 10

    if not os.path.exists('figures/univariate/Figure1/%s/%s' % (prefix, exper_name)):
        os.mkdir('figures/univariate/Figure1/%s/%s' % (prefix, exper_name))

    f1 = True
    if f1:
        for prior in priors[exper_name]:
            for PC in ['U', 'V']:
                # if not os.path.exists('output/univariate/%s/%s')
                res = group_alignments(prior, PC, s_list=s_list, n_rep=n_rep,
                                       prefix=prefix, suffix='_by_5')
                # se = eval_se(prior, s_list, gamma, iters)
                make_comp_plot(res, prior, PC, s_list=s_list, to_save=True, \
                               prefix = prefix, suffix=exper_name)

    # prior = 'Point_normal'
    # PC = 'U'
    # res = group_alignments(prior, PC, s_list=s_list, n_rep=n_rep, prefix=prefix)  # [0.9, 1.0, 1.1, 1.2, 1.3]
    # se = eval_se(prior, s_list, gamma, iters)
    # print(se)
    # make_comp_plot(res, se, prior, PC, s_list=s_list, to_save=False, suffix=suffix)  # 'min_s_pilot'

