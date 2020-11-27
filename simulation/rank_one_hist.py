import numpy as np
import os
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.pca import get_pca
from ebpca.empbayes import NonparEBChecker
from ebpca.amp import ebamp_gaussian
from ebpca.misc import ebmf
from simulation.helpers import approx_prior, simulate_prior, signal_plus_noise_model, fill_alignment
from tutorial import redirect_pc
from ebpca.preprocessing import normalize_obs

# create directories to save the figures
fig_prefix = 'figures/univariate'
if not os.path.exists(fig_prefix):
    for i in range(3):
        os.makedirs('%s/Figure%i' % (fig_prefix, i + 1))

# ------------------------------------------------------
# Figure 2:
# predicted density, fitted density versus observed data
# contrasting EBMF and EB-PCA
# histogram
# ------------------------------------------------------


def get_marginal_plots(prior, prefix, s_star = 1.3, i = 0, gamma = None, to_save = True, iters = 5):
    if gamma is None:
        # use data from replication 0
        # generated in n_rep simulations
        data_prefix = 'output/univariate/%s/%s/data/s_%.1f' % (prior, prefix, s_star)
        u_star = np.load('%s_copy_%i_u_star_n_2000_gamma_2.0.npy' % (data_prefix, i), allow_pickle=False)
        v_star = np.load('%s_copy_%i_v_star_n_2000_gamma_2.0.npy' % (data_prefix, i), allow_pickle=False)
        X = np.load('%s_copy_%i_n_2000_gamma_2.0.npy' % (data_prefix, i), allow_pickle=False)
        fig_suffix = ''
    else:
        # regenerate data for gamma other than 2
        # to see model performance under other gamma value
        np.random.seed(100)
        n = 1000
        d = int(n * gamma)
        rank = 2
        u_star = simulate_prior(prior, n, seed=np.random.randint(0, 10000, 1)[0])
        v_star = simulate_prior(prior, d, seed=np.random.randint(0, 10000, 1)[0])
        Y = signal_plus_noise_model(u_star, v_star, s_star)
        X = normalize_obs(Y, rank)
        fig_suffix = 'gamma_%.1f' % gamma

    # Figure 2 folder
    f2_prefix = 'univariate/Figure2/%s/%s/' % (prior, prefix)
    if not os.path.exists('figures/' + f2_prefix):
        os.makedirs('figures/' + f2_prefix)

    # make pca pack
    pcapack = get_pca(X, 1)
    # approximate true prior with empirical distribution
    [uTruePriorLoc, uTruePriorWeight] = approx_prior(u_star, pcapack.U)
    [vTruePriorLoc, vTruePriorWeight] = approx_prior(v_star, pcapack.V)

    # exchange U and V to make EBMF denoise V first
    pcapack_t = pcapack
    pcapack_t = pcapack_t._replace(U = pcapack.V)
    pcapack_t = pcapack_t._replace(V = pcapack.U)

    # run EBMF
    # exchange u and v true prior
    ldenoiser = NonparEBChecker(vTruePriorLoc, vTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EBMF' + fig_suffix,
                                histcol='tab:blue', PCname = 'V',
                                xRange = xRanges_V[prior], yRange = yRanges_V[prior])
    fdenoiser = NonparEBChecker(uTruePriorLoc, uTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EBMF' + fig_suffix,
                                histcol='tab:blue', xRange=xRanges_U[prior], yRange = yRanges_U[prior])
    U_ebmf, V_ebmf, _ = ebmf(pcapack_t, ldenoiser=ldenoiser, fdenoiser=fdenoiser,
                             iters=iters, ebpca_scaling=True, tau_by_row=False)

    # for left PC in U
    print('EBMF alignments:', fill_alignment(V_ebmf, u_star, iters))

    # run EB-PCA
    udenoiser = NonparEBChecker(uTruePriorLoc, uTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EB-PCA' + fig_suffix,
                                histcol='tab:red', xRange=xRanges_U[prior], yRange = yRanges_U[prior])
    vdenoiser = NonparEBChecker(vTruePriorLoc, vTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EB-PCA' + fig_suffix,
                                histcol='tab:red', PCname = 'V',
                                xRange = xRanges_V[prior], yRange = yRanges_V[prior])
    U_ebpca, V_ebpca = ebamp_gaussian(pcapack, iters=iters, udenoiser=udenoiser, vdenoiser=vdenoiser)

    print('EB-PCA alignments:', fill_alignment(U_ebpca, u_star, iters))

    return fill_alignment(U_ebpca, u_star, iters), fill_alignment(V_ebpca, v_star, iters),\
           fill_alignment(V_ebmf, u_star, iters), fill_alignment(U_ebmf, v_star, iters)

if __name__ == '__main__':
    # take argument: number of iterations
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, help="number of iterations",
                        default=2, const=2, nargs='?')
    parser.add_argument("--s_star", type=float, help="true signal strength",
                        default=1.1, const=1.1, nargs='?')
    args = parser.parse_args()
    iters = args.iters
    s_star = args.s_star

    # load experiment results
    prefixes = {'n_2000': 'n_2000_gamma_2.0_nsupp_ratio_0.5_0.5_useEM_True',
                'n_1000': 'n_1000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_True',
                'MOSEK_pilot': 'n_2000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_False'}
    n_reps = {'n_2000': 50, 'n_1000': 50, 'MOSEK_pilot': 15}
    priors = {'n_2000': ['Point_normal', 'Two_points', 'Uniform'],
              'n_1000': ['Point_normal', 'Two_points', 'Uniform'],
              'MOSEK_pilot': ['Point_normal_0.1', 'Point_normal_0.5', 'Two_points', 'Uniform_centered']}
    s_lists = {'n_2000': [1.1, 1.3, 1.5, 2.0],
               'n_1000': [1.1, 1.3, 1.5, 2.0],
               'MOSEK_pilot': [1.1, 1.3, 1.5]}

    exper_name = 'MOSEK_pilot'
    n_rep = n_reps[exper_name]
    prefix = prefixes[exper_name]
    s_list = s_lists[exper_name]

    # load scaling specification
    xRanges_U = {'Point_normal_0.1': [-15, 15],
                 'Point_normal_0.5': [-8, 8],
                 'Two_points': [-4, 4],
                 'Uniform_centered': [-5, 5]}
    yRanges_U = {'Point_normal_0.1': [0, 0.8],
                 'Point_normal_0.5': [0, 0.7],
                 'Two_points': [0, 0.55],
                 'Uniform_centered': [0, 0.6]}
    xRanges_V = {'Point_normal_0.1': [-17, 17],
                 'Point_normal_0.5': [-10, 10],
                 'Two_points': [-6, 6],
                 'Uniform_centered': [-6, 6]}
    yRanges_V = {'Point_normal_0.1': [0, 0.45],
                 'Point_normal_0.5': [0, 0.35],
                 'Two_points': [0, 0.27],
                 'Uniform_centered': [0, 0.28]}

    for prior in priors[exper_name]:
        print(prior)
        # plot marginal plots
        get_marginal_plots(prior, prefix, s_star, iters=iters)
        # double check with alignments from the experiments
        align_dir = 'output/univariate/%s/%s/alignments' % (prior, prefix)
        for method in ['EBMF', 'EB-PCA']:
            print(method)
            aligns = np.load('%s/%s_u_s_%.1f_n_rep_%i.npy' %
                             (align_dir, method, s_star, n_rep))
            print(aligns[0])

    # for prior in ['Two_points', 'Uniform', 'Point_normal']:
    #     get_marginal_plots(prior)
    # prior = 'Two_points'
    # get_marginal_plots(prior)
