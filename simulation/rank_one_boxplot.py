import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prior", type=str, help="enter the prior to visualize",
                    default='Two_points', const='Two_points', nargs='?')
parser.add_argument("--PC", type=str, help="enter which PC to visualize",
                    default='left', const='left', nargs='?')
args = parser.parse_args()
prior = args.prior
PC = args.PC

# ----------------------------------------------
# Figure 1:
# alignment comparison across methods
# boxplot
# ----------------------------------------------

def alignment_boxplots(pca, bayesamp, ebpca, ebmf, spca, ticks):

    # reference: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots 2nd answer
    f, ax = plt.subplots(figsize=(8, 6))
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    bp1 = ax.boxplot(pca, positions=np.array(range(len(pca))) * 5.0 - 1.2, sym='', widths=0.45)
    bp2 = ax.boxplot(bayesamp, positions=np.array(range(len(ebpca))) * 5.0 - 0.6, sym='', widths=0.45)
    bp3 = ax.boxplot(ebpca, positions=np.array(range(len(ebmf))) * 5.0 + 0.0, sym='', widths=0.45)
    bp4 = ax.boxplot(ebmf, positions=np.array(range(len(spca))) * 5.0 + 0.6, sym='', widths=0.45)
    bp5 = ax.boxplot(spca, positions=np.array(range(len(spca))) * 5.0 + 1.2, sym='', widths=0.45)

    set_box_color(bp1, '#636363')
    set_box_color(bp2, '#fc9272')
    set_box_color(bp3, '#de2d26')
    set_box_color(bp4, '#3182bd')
    set_box_color(bp5, '#31a354')  # colors are from http://colorbrewer2.org/

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c='#636363', label='PCA')
    ax.plot([], c='#fc9272', label='BayesAMP')
    ax.plot([], c='#de2d26', label='EB-PCA')
    ax.plot([], c='#3182bd', label='EBMF')
    ax.plot([], c='#31a354', label='sPCA')

    ax.legend(loc='lower right')
    plt.xticks(range(0, len(ticks) * 5, 5), ticks)
    ax.set_xlabel("signal strength")
    ax.set_ylabel('alignment with ground truth')
    ax.set_xlim(-2, len(ticks) * 5)
    # plt.ylim(0, 1)
    # plt.title('Sqaured alignment, {} \n n={}, sparsity={}, alpha={}, {} runs'.format(vector, n, sparsity, gamma, nruns))
    return ax
    # plt.savefig(
    #     'result/point_normal/acccompare/acccompare_amp_ebmf_{}_sparsity_{}_alpha_{}.png'.format(vector, sparsity,
    #                                                                                             alpha))

# os.chdir('/Users/chang/PycharmProjects/generalAMP/simulation/')

# load alignments from 50 replications
def load_alignments(prior, method, ind=-1, PC='left', s_list=[1.3, 1.8, 3.0]):
    n_par = len(s_list)
    align_dir = 'output/univariate/%s/alignments' % prior
    if PC=='left':
        aligns = [(np.load('%s/%s_u_s_%.1f_n_rep_50.npy' % (align_dir, method, s))[:, ind]).tolist() for s in s_list]
    else:
        aligns = [(np.load('%s/%s_v_s_%.1f_n_rep_50.npy' % (align_dir, method, s))[:, ind]).tolist() for s in s_list]
    # remove nan values:
    print('Number of NA alignments for %s, %s' % (method, prior), [np.sum(np.isnan(aligns[i])) for i in range(n_par)])
    aligns = [np.array(aligns[i])[~np.isnan(aligns[i])].tolist() for i in range(n_par)]
    return aligns

# prior = 'Point_normal' # 'Two_points' # 'Uniform'
def make_comp_plot(prior, PC, s_list = [1.3, 1.8, 3.0]):
    pca = load_alignments(prior, 'EB-PCA', 0, PC)
    bayesamp = load_alignments(prior, 'BayesAMP', -1, PC)
    ebpca = load_alignments(prior, 'EB-PCA', -1, PC)
    ebmf = load_alignments(prior, 'EBMF', -1, PC)
    alignment_boxplots(pca, bayesamp, ebpca, ebmf, ebpca, s_list)
    plt.title('%s, %s PC, method comparison' % (prior, PC))
    plt.savefig('figures/univariate/Figure1/%s_%sPC_method_comp_boxplots.png' % (prior, PC))

# ----------------------------------------------
# Figure 3:
# alignment comparison across methods
# boxplot
# ----------------------------------------------



if __name__ == '__main__':
    for prior in ['Point_normal', 'Two_points', 'Uniform']:
        for PC in ['left', 'right']:
            make_comp_plot(prior, PC)