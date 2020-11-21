import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.extend(['../../generalAMP'])

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
    ax.plot([], c=bp_colors[1], label='BayesAMP')
    ax.plot([], c=bp_colors[2], label='EB-PCA')
    ax.plot([], c=bp_colors[3], label='EBMF')
    if n == 5:
        ax.plot([], c=bp_colors[4], label='sPCA')

    ax.legend(loc='lower right')
    plt.xticks(range(0, len(ticks) * n, n), ticks)
    ax.set_xlabel("signal strength")
    ax.set_ylabel('alignment with ground truth')
    ax.set_xlim(-2, len(ticks) * n)
    ax.set_ylim(0 - 0.05, 1 + 0.05)
    return ax
# os.chdir('/Users/chang/PycharmProjects/generalAMP/simulation/')

# load alignments from 50 replications
def load_alignments(prior, method, ind=-1, PC='U', rm_na =True, s_list=[1.3, 1.8, 3.0]):
    n_par = len(s_list)
    align_dir = 'output/univariate/%s/alignments' % prior
    if method == 'spca':
        if PC == 'U':
            aligns = [(pd.read_table('%s/%s_u_s_%.1f_n_rep_50.txt' % (align_dir, method, s)).values.reshape(-1)) \
                      for s in s_list]
        else:
            aligns = [(pd.read_table('%s/%s_v_s_%.1f_n_rep_50.txt' % (align_dir, method, s)).values.reshape(-1)) \
                      for s in s_list]
    else:
        if PC=='U':
            aligns = [(np.load('%s/%s_u_s_%.1f_n_rep_50.npy' % (align_dir, method, s))[:, ind]) for s in s_list]
        else:
            aligns = [(np.load('%s/%s_v_s_%.1f_n_rep_50.npy' % (align_dir, method, s))[:, ind]) for s in s_list]

    if rm_na:
        # remove nan values:
        print('Number of NA alignments for %s, %s' % (method, prior), [np.sum(np.isnan(aligns[i])) for i in range(n_par)])
        aligns = [np.array(aligns[i])[~np.isnan(aligns[i])] for i in range(n_par)]
    else:
        aligns = np.array(aligns)
    return aligns

def group_alignments(prior, PC, rm_na = True, s_list=[1.3, 1.8, 3.0]):
    pca = load_alignments(prior, 'EB-PCA', 0, PC, rm_na, s_list)
    bayesamp = load_alignments(prior, 'BayesAMP', -1, PC, rm_na, s_list)
    ebpca = load_alignments(prior, 'EB-PCA', 5, PC, rm_na, s_list)
    ebmf = load_alignments(prior, 'EBMF', -1, PC, rm_na, s_list)
    if prior == 'Point_normal':
        spca = load_alignments(prior, 'spca', -1, PC, rm_na, s_list)
        res = [pca, bayesamp, ebpca, ebmf, spca]
    else:
        res = [pca, bayesamp, ebpca, ebmf]
    return res

# prior = 'Point_normal' # 'Two_points' # 'Uniform'
def make_comp_plot(res, prior, PC, s_list = [1.3, 1.8, 3.0]):
    alignment_boxplots(res, s_list)
    plt.title('%s, %s, method comparison' % (prior.replace('_', ' '), PC))
    plt.savefig('figures/univariate/Figure1/%s_%s_method_comp_boxplots.png' % (prior, PC))
    plt.close()

if __name__ == '__main__':
    # Figure 1
    # res = group_alignments('Point_normal', 'V', s_list=[1.3, 1.4, 1.5, 1.6, 3.0])
    # make_comp_plot(res, 'Point_normal', 'V', s_list=[1.3, 1.4, 1.5, 1.6, 3.0])
    f1= True
    if f1:
        for prior in ['Point_normal', 'Two_points', 'Uniform']:
            for PC in ['U', 'V']:
                res = group_alignments(prior, PC, s_list = [1.3, 1.5, 1.7, 3.0]) # [1.3, 1.4, 1.5, 1.6, 3.0]
                make_comp_plot(res, prior, PC, s_list = [1.3, 1.5, 1.7, 3.0])
