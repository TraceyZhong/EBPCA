import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from simulation.helpers import get_joint_alignment, get_marginal_alignment, align_pc

# ----------------------------------------------
# Figure 4:
# visualize denoised PC
# dot plot
# ----------------------------------------------

def load_PC_star(prior, s_star, n_copy=0):
    data_prefix = 'output/bivariate/%s/data/s_%.1f_%.1f' % (prior, s_star[0], s_star[1])
    u_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, n_copy), allow_pickle=False)
    return u_star

def load_dePC(prior, method, s_star, n_copy=0):
    dePC_prefix = 'output/bivariate/%s/denoisedPC/%s_U_s_%.1f_%.1f' % (prior, method, s_star[0], s_star[1])
    X = np.load('%s_n_copy_%i.npy' % (dePC_prefix, n_copy), allow_pickle=False)
    return X

def plot_rank2_dePC(star, mar, joint, prior, s_star):
    # tune aesthetics
    plt.rcParams['font.size'] = 14

    # align estimated PC with true PC
    # in both direction and magnitude
    pca_est = align_pc(mar[:, :, 0], star)
    mar_est = align_pc(mar[:, :, -1], star)
    joint_est = align_pc(joint[:, :, -1], star)

    # get marginal alignment
    pca_align = get_marginal_alignment(pca_est, star)
    mar_align = get_marginal_alignment(mar_est, star)
    joint_align = get_marginal_alignment(joint_est, star)

    plot_est = [star, pca_est, mar_est, joint_est]
    plot_aligns = [[1, 1], pca_align, mar_align, joint_align]
    plot_method = ['Ground truth', 'PCA', 'EB-PCA marginal estimation', 'EB-PCA joint estimation']

    # start plotting
    for i in range(4):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
        ax.scatter(plot_est[i][:, 0], plot_est[i][:, 1], s = 2, alpha = 0.8)
        if i > 0:
            ax.set_title('%s \n bivariate alignment=%.2f' % \
                         (plot_method[i], get_joint_alignment(plot_aligns[i], False)))
            ax.set_xlabel('PC 1, alignment={:.2f}'.format(plot_aligns[i][0]))
            ax.set_ylabel('PC 2, alignment={:.2f}'.format(plot_aligns[i][1]))
        if i == 0:
            ax.set_title('%s' % (plot_method[i]))
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plt.savefig('figures/bivariate/%s_%s_s_%.1f_%.1f.png' % \
                    (prior, plot_method[i].replace(' ', '_'), s_star[0], s_star[1]))
        plt.close()
    # plt.suptitle('{}'.format(prior.replace('_', ' ')))


# ----------------------------------------------
# Table 2:
# make table of alignment statistics
# ----------------------------------------------

# load alignments from 50 replications
def eval_align_stats(prior, method, s_star, ind=-1):
    align_dir = 'output/bivariate/%s/alignments' % prior
    aligns = np.load('%s/%s_u_s_%.1f_%.1f_n_rep_50.npy' %
                     (align_dir, method, s_star[0], s_star[1]))

    print('Print alignment statistics for %s: ' % method)
    print('#simulations that returns NA: ', np.sum(np.isnan(aligns[:, :, ind])))
    print('marginal alignments: \n \t mean:', np.nanmean(aligns[:, :, ind], axis=0))
    print('\t std:', np.nanstd(aligns[:, :, ind], axis=0))

    joint_aligns = [get_joint_alignment(aligns[i, :, ind]) for i in range(aligns.shape[0])]

    print('joint alignments: \n \t mean: ',
          np.nanmean(joint_aligns))
    print('\t std:', np.nanstd(joint_aligns))

if __name__ == '__main__':
    # --------------------------
    # rank-2 simulation settings
    # --------------------------
    iters = 5
    n_rep = 50
    rank = 2
    n = 1000
    d = 1000

    # take named argument from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--s_star_1", type=float, help="true signal strength 1",
                        default=2.5, const=2.5, nargs='?')
    parser.add_argument("--s_star_2", type=float, help="true signal strength 2",
                        default=2, const=2, nargs='?')
    args = parser.parse_args()
    s_star = [args.s_star_1, args.s_star_2]

    for prior in ['Three_points']: # 'Uniform_circle',
        # load data
        # assuming results from rank_two.py is available
        i = 0
        star = load_PC_star(prior, s_star, i)
        mar = load_dePC(prior, "marginal", s_star, i)

        plt.scatter(mar[:, 0, -1], mar[:, 1, -1], s = 0.5, alpha = 0.5)
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.savefig('figures/bivariate/tmp_%s.png' % prior)
        plt.close()

        joint = load_dePC(prior, "joint", s_star, i)
        # make plots
        plot_rank2_dePC(star, mar, joint, prior, s_star)

        # print table statistics
        # for method in ['marginal', 'joint']:
        #     eval_align_stats(prior, method)



