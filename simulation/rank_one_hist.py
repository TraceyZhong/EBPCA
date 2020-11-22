import numpy as np
import os
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.pca import get_pca
from ebpca.empbayes import NonparEBChecker
from ebpca.amp import ebamp_gaussian
from ebpca.misc import ebmf
from simulation.helpers import approx_prior, simulate_prior, signal_plus_noise_model, fill_alignment
from tutorial import get_alignment
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


def get_marginal_plots(prior, s_star = 1.3, i = 0, gamma = None, to_save = True, iters = 5):
    if not '%s/Figure2/%s' % (fig_prefix, prior):
        os.mkdir('%s/Figure2/%s' % (fig_prefix, prior))
    if gamma is None:
        # use data from replication 0
        # generated in n_rep simulations
        data_prefix = 'output/univariate/%s/data/s_%.1f' % (prior, s_star)
        u_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, i), allow_pickle=False)
        v_star = np.load('%s_copy_%i_v_star.npy' % (data_prefix, i), allow_pickle=False)
        X = np.load('%s_copy_%i.npy' % (data_prefix, i), allow_pickle=False)
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

    # make pca pack
    pcapack = get_pca(X, 1)

    # approximate true prior with empirical distribution
    [uTruePriorLoc, uTruePriorWeight] = approx_prior(u_star, pcapack.U)
    [vTruePriorLoc, vTruePriorWeight] = approx_prior(v_star, pcapack.V)

    # Figure 2 folder
    f2_prefix = 'univariate/Figure2/' + prior + '/'
    if not os.path.exists('figures/' + f2_prefix):
        os.mkdir('figures/' + f2_prefix)


    # run EBMF
    ldenoiser = NonparEBChecker(uTruePriorLoc, uTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EBMF' + fig_suffix,
                                histcol='tab:blue')
    fdenoiser = NonparEBChecker(vTruePriorLoc, vTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EBMF' + fig_suffix,
                                histcol='tab:blue', PCname = 'V')
    U_ebmf, V_ebmf, _ = ebmf(pcapack, ldenoiser=ldenoiser, fdenoiser=fdenoiser,
                             iters=iters, ebpca_scaling=True)

    print('EBMF alignments:', fill_alignment(U_ebmf, u_star, iters))

    # run EB-PCA
    udenoiser = NonparEBChecker(uTruePriorLoc, uTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EB-PCA' + fig_suffix,
                                histcol='tab:red')
    vdenoiser = NonparEBChecker(vTruePriorLoc, vTruePriorWeight, optimizer="Mosek",
                                to_save=to_save, fig_prefix=f2_prefix + 'EB-PCA' + fig_suffix,
                                histcol='tab:red', PCname = 'V')
    U_ebpca, V_ebpca = ebamp_gaussian(pcapack, iters=iters, udenoiser=udenoiser, vdenoiser=vdenoiser)

    print('EB-PCA alignments:', fill_alignment(U_ebmf, u_star, iters))

    return fill_alignment(U_ebpca, u_star, iters), fill_alignment(V_ebpca, v_star, iters),\
           fill_alignment(U_ebmf, u_star, iters), fill_alignment(V_ebmf, v_star, iters)


if __name__ == '__main__':
    get_marginal_plots('Point_normal')
    # for prior in ['Two_points', 'Uniform', 'Point_normal']:
    #     get_marginal_plots(prior)
    # prior = 'Two_points'
    # get_marginal_plots(prior)
