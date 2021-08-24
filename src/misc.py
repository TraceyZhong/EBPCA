'''
==========
Mean-field Variational Bayes method
==========
This module includes implementations of the method in https://arxiv.org/abs/1802.06931 ,
  which denoises PCs with a variational Bayes method.
We include it here to compare with EB-PCA.
There are two functions:
1. ebmf:
    This function is an exact implementation of the EBMF method for rank-1 model. When
    supplied appropriate parametric MLE methodS, it produces exactly the same results
    as the original implementation of EBMF in R (flashr: https://github.com/stephenslab/flashr).
2. MeanFieldVB
    This function is an extension of the EBMF method for rank-k models in the following
    two aspects: 1. It applies NPMLE for prior estimation; 2. It estimates multivariate
    priors jointly and performs joint update for k PCs. Both of these extensions are
    to remove subtle differences between EB-PCA and mean-field VB and to focus on the
    distinction between AMP and CAVI updates.
'''

import numpy as np
from ebpca.empbayes import _gaussian_pdf, NonparEB

def ebmf(pcapack, ldenoiser = NonparEB(), fdenoiser = NonparEB(),
         update_family = 'nonparametric', iters = 5, tol=1e-1,
         ebpca_scaling=True, tau_by_row=True):
    """
    Implement Empirical Bayes Matrix Factorization as described in Wang and Stephens, 2021.
    Args:
        pcapack (tuple): output from pca.get_pca, containing normalized data and signal strength
        estimates
        ldenoiser (_BaseEmpiricalBayes): a PointNormalEB or NonparEB denoiser
        fdenoiser (_BaseEmpiricalBayes): a PointNormalEB or NonparEB denoiser
        update_family (str): use Point Normal or nonparametric distributions to model prior,
        denoise PCs and evaluate objective function.
        iters (int): maximum number of iterations
        tol (float): 1e-2, the same as the default setting in flashr
        ebpca_scaling (bool): whether or not to apply the same scaling as EB-PCA, i.e. satisfy
        (2.5) on Zhong et al., 2019. If True, the correspondence
        between EBMF and EB-PCA can be described by the equations on P11, Zhong et al., 2019.
        If False, the scaling is consistent with the original EBMF method.
        Note that re-scaling doesn't affect the estimates of EBMF, as EBMF model does not
        impose any assumption on the scaling of initialization right / left PC
    Returns:
        left PC estimates, right PC estimates, objective function
    Reference:
        Wang, Wei, and Matthew Stephens. "Empirical bayes matrix factorization." Journal of Machine Learning Research 22.120 (2021): 1-40.
        https://github.com/stephenslab/flashr
    Remarks:
        1. This implementation supportS both parametric and nonparametric priors. We implemented PointNormalEB
        in empbayes.PY as an example for parametric denoiser.
        2. A test showing that this implementation is the same as flashr R package is in ???
    Example:
        L, F = ebmf(pcapack, update_family = 'nonparametric', iters=5)
    """

    X = pcapack.X
    u, v = pcapack.U, pcapack.V
    # get dimension
    (n, d) = X.shape

    # initialize parameter tau
    if tau_by_row:
        tau = n
        pc1 = 'u'
        pc2 = 'v'
    else:
        tau = d
        pc1 = 'v'
        pc2 = 'u'

    if ebpca_scaling:
        print('Apply rescaling to match the scale with EB-PCA in marginal plots')
        # get signal
        signals = pcapack.signals
        # apply the same scaling in EB-PCA
        u = u / np.sqrt((u ** 2).sum(axis=0)) * np.sqrt(n)
        v = v / np.sqrt((v ** 2).sum(axis=0)) * np.sqrt(d)
        print('signal: %.2f' % signals)
        mu_constant = np.float(signals / tau)
    else:
        mu_constant = 1
    print('mu constant: %.2f' % mu_constant)

    # re-label u, v with l, f
    l_hat = u
    f_hat = v

    # initialize placeholder for l, f update results
    L = l_hat[:,:, np.newaxis]
    F = f_hat[:,:, np.newaxis]

    # use the same scaling as EBMF
    # initialize the first update with svd
    # first denoise the loadings
    l_hat = X.dot(f_hat) / np.sum(f_hat ** 2)
    mu = np.diag([mu_constant])
    sigma_sq = np.diag([1 / (np.sum(f_hat ** 2) * tau)])
    print('fhat %.2f' % np.sum(f_hat ** 2))
    if ebpca_scaling:
        l_hat = l_hat * np.sum(f_hat ** 2)
        mu = mu * np.sum(f_hat ** 2)
        sigma_sq = sigma_sq * np.sum(f_hat ** 2)**2

    print('mu=%.4f' % mu)
    print('sigma_sq=%.4f' % sigma_sq)
    print('initial SNR: %.4f' % (sigma_sq / mu**2))

    obj_funcs = []
    t = 0
    new_flag = False
    while t < iters:
        old_flag = new_flag
        print("at ebmf iter {}".format(t))
        ldenoiser.fit(l_hat, mu, sigma_sq, figname='_%s_iter%02d.png' % (pc1, t))
        El = ldenoiser.denoise(l_hat, mu, sigma_sq)
        Varl = ldenoiser.ddenoise(l_hat, mu, sigma_sq) * (sigma_sq / mu)
        El2 = El**2 + Varl.reshape(-1,1)
        L = np.dstack((L, np.reshape(El,(-1,1,1))))
        # Evaluate log likelihood
        [par1, par2] = ldenoiser.get_estimate()
        KL_l = marginal_lik_F_func([par1, par2.reshape(-1)],
                                   l_hat, np.sqrt(sigma_sq), mu, update_family) - \
               NM_posterior_e_loglik(l_hat, mu, sigma_sq, El, El2)
        # Update the estimate of the factor
        f_hat = X.T.dot(El) / np.sum(El2)
        mu_bar = np.diag([mu_constant])
        sigma_bar_sq = np.diag([1 / (np.sum(El2) * tau)])
        if ebpca_scaling:
            f_hat = f_hat * np.sum(El2)
            mu_bar = mu_bar * np.sum(El2)
            sigma_bar_sq = sigma_bar_sq * np.sum(El2)**2
        # print('sigma2_bar/mu_bar**2 %.2f' % (sigma_bar_sq / np.power(mu_bar,2)))
        fdenoiser.fit(f_hat, mu_bar, sigma_bar_sq, figname='_%s_iter%02d.png' % (pc2, t))
        Ef = fdenoiser.denoise(f_hat, mu_bar, sigma_bar_sq)
        Varf = fdenoiser.ddenoise(f_hat, mu_bar, sigma_bar_sq) * (sigma_bar_sq / mu_bar)
        Ef2 = Ef**2 + Varf.reshape(-1,1) # [:,:,0]
        F = np.dstack((F, np.reshape(Ef, (-1,1,1))))
        # Evaluate log likelihood
        [par1, par2] = fdenoiser.get_estimate()
        KL_f = marginal_lik_F_func([par1, par2.reshape(-1)],
                                   f_hat, np.sqrt(sigma_bar_sq), mu_bar, update_family) - \
               NM_posterior_e_loglik(f_hat, mu_bar, sigma_bar_sq, Ef, Ef2)
        # Update the estimate of the loading
        l_hat = X.dot(Ef) / np.sum(Ef2)
        mu = np.diag([mu_constant])
        sigma_sq = np.diag([1 / (np.sum(Ef2) * tau)])
        if ebpca_scaling:
            l_hat = l_hat * np.sum(Ef2)
            mu = mu * np.sum(Ef2)
            sigma_sq = sigma_sq * np.sum(Ef2)**2
        # print('sigma2/mu**2 %.2f' % (sigma_sq / np.power(mu, 2)))
        # Evaluate objective function
        obj_func = get_cond_logl(El, El2, Ef, Ef2, X, tau) + KL_l + KL_f
        obj_funcs.append(obj_func)
        print('Objective F function: {:.5f}'.format(obj_func))
        t += 1
        if t == 1:
            new_flag = False
        else:
            # Use change in objective function as convergence threshold
            new_flag = abs(obj_funcs[-1] - obj_funcs[-2]) < tol
            if new_flag == True and old_flag == False:
                print('EBMF converged in {} iterations, tol={:.1e}.'.format(t, tol))

    if not new_flag:
        print('EBMF failed to converge in {} iterations.'.format(iters))

    return L, F, obj_funcs

def MeanFieldVB(pcapack, ldenoiser = NonparEB(), fdenoiser = NonparEB(),
                iters = 5, ebpca_scaling=True, start_from_f=False,
                ebpca_ini = False):
    """
    Run the naive mean-field variational Bayes algorithm as described in Zhong et al., 2021
    for a fixed number of iterations.
    Args:
        pcapack (tuple): output from pca.get_pca, containing normalized data and signal strength
        estimates
        ldenoiser (_BaseEmpiricalBayes): a PointNormalEB or NonparEB denoiser
        fdenoiser (_BaseEmpiricalBayes): a PointNormalEB or NonparEB denoiser
        iters (int): maximum number of iterations
        ebpca_scaling (bool): whether or not to apply the same scaling as EB-PCA, i.e. satisfy
        (2.5) on Zhong et al., 2019. If True, the correspondence
        between EBMF and EB-PCA can be described by the equations on P11, Zhong et al., 2019.
        If False, the scaling is consistent with the original EBMF method.
        Note that re-scaling doesn't affect the estimates of EBMF, as EBMF model does not
        impose any assumption on the scaling of initialization right / left PC.
        start_from_f (bool): if True, initiate the algorithm from denoising the right PC (f); if False,
        initiate from the left PC (g). EB-PCA starts from the right PC, so setting
        start_from_f=True aligns this algorithm better with EB-PCA. Yet the original
        implementation of EBMF starts from the left PC.
        ebpca_ini (bool): if True, initialize mean-field VB with the same initialization as EB-PCA;
        if False, initialize with the same sample PC based statistics as EBMF.
    Returns:
        left PC estimates, right PC estimates, objective function
    Reference:
        Wang, Wei, and Matthew Stephens. "Empirical bayes matrix factorization." Journal of Machine Learning Research 22.120 (2021): 1-40.
        https://github.com/stephenslab/flashr
    Example:
        L, F = ebmf(pcapack, iters=5)
    """
    X = pcapack.X
    u, v = pcapack.U, pcapack.V
    # get dimension
    (n, d) = X.shape
    k = pcapack.K

    if ebpca_scaling:
        print('Apply rescaling to match the scale with EB-PCA in marginal plots')
        # get signal
        signals = pcapack.signals
        # apply the same scaling in EB-PCA
        u = u / np.sqrt((u ** 2).sum(axis=0)) * np.sqrt(n)
        v = v / np.sqrt((v ** 2).sum(axis=0)) * np.sqrt(d)
        print('rescaling to match EB-PCA scaling')
    else:
        mu_constant = np.diag(np.repeat(1, k))

    # initialize parameter tau
    # re-label u, v with l, f, to be consistent with EBMF notations
    if start_from_f:
        # the original EBMF implementation starts from l
        # to start from f, we flip left and right PCs
        l_hat = v
        f_hat = u
        tau = d
        init_aligns = pcapack.feature_aligns
        X = X.T
        pc2 = 'u'
        alpha = n/d
    else:
        l_hat = u
        f_hat = v
        tau = n
        init_aligns = pcapack.sample_aligns
        pc2 = 'v'
        alpha = d/n
    # initialize placeholder for l, f update results
    L = l_hat[:,:, np.newaxis]
    F = f_hat[:,:, np.newaxis]

    Omega_mat = f_hat.T @ f_hat
    if ebpca_scaling:
        l_hat = X @ f_hat
        if ebpca_ini:
            # initialize with compound decision model predicted by RMT
            # X=1/sqrt(nd)*u*v
            # X@f_hat=\|v\|_2^2 / sqrt(nd) * u
            # alpha=\|v\|_2^2 / sqrt(nd) = d/n if not transpose; n/d if transpose
            mu = np.sqrt(alpha) * np.diag(init_aligns) * pcapack.mu
            sigma_sq = alpha * \
                       np.diag(1 - init_aligns ** 2) * (pcapack.mu)**2
        else:
            # initialize with sample PCs based statistics
            # as Algorithm 2, step 1-2 in Wang and Stephens, 2021
            mu = (1 / tau) * Omega_mat * signals
            sigma_sq = (1 / tau) * Omega_mat
    else:
        Sigma_mat = np.linalg.pinv(Omega_mat)
        l_hat = X @ f_hat @ Sigma_mat
        mu = mu_constant
        sigma_sq = Sigma_mat / tau

    t = 0
    while t < iters:
        print("at Mean Field VB iter {}".format(t))
        # Denoise l_hat to get l
        ldenoiser.fit(l_hat, mu, sigma_sq)
        El = ldenoiser.denoise(l_hat, mu, sigma_sq)
        # Two equivalent way of evaluating posterior 2nd moment: var+sq first moment
        # b  = ldenoiser.ddenoise(l_hat, mu, sigma_sq)
        # Varl = b @ sigma_sq @ np.linalg.pinv(mu).T
        # El2 = El.T @ El + np.sum(Varl, axis=0) # El**2 + Varl.reshape(-1,1)
        # or directly
        El2 = np.sum(ldenoiser.pos2m(l_hat, mu, sigma_sq), axis=0)
        L = np.dstack((L, np.reshape(El,(-1,k,1))))
        # Update the estimate of the factor
        Omega_bar_mat = El2
        if ebpca_scaling:
            f_hat = X.T @ El
            mu_bar = (1 / tau) * Omega_bar_mat * signals
            sigma_bar_sq = (1 / tau) * Omega_bar_mat
        else:
            Sigma_bar_mat = np.linalg.pinv(Omega_bar_mat)
            f_hat = X.T @ El @ Sigma_bar_mat
            mu_bar = mu_constant
            sigma_bar_sq = Sigma_bar_mat / tau
        fdenoiser.fit(f_hat, mu_bar, sigma_bar_sq, figname='_%s_iter%02d.png' % (pc2, t))
        Ef = fdenoiser.denoise(f_hat, mu_bar, sigma_bar_sq)
        Ef2 = np.sum(fdenoiser.pos2m(f_hat, mu_bar, sigma_bar_sq), axis=0)
        F = np.dstack((F, np.reshape(Ef, (-1,k,1))))
        # Update the estimate of the loading
        Omega_mat = Ef2
        if ebpca_scaling:
            l_hat = X @ Ef
            mu = (1 / tau) * Omega_mat * signals
            sigma_sq = (1 / tau) * Omega_mat
        else:
            Sigma_mat = np.linalg.pinv(Omega_mat)
            l_hat = X @ Ef @ Sigma_mat
            mu = mu_constant
            sigma_sq = Sigma_mat / tau
        t += 1

        if start_from_f:
            F, L = L,F 
    return L, F


# -------------------------
# utils functions for ebmf
# -------------------------

def get_cond_logl(EL, EL2, EF, EF2, A, tau):
    # Reference:
    # https://github.com/stephenslab/flashr/blob/master/R/get_functions.R
    n, d = A.shape
    LF = np.outer(EL, EF)
    R2 = np.sum((A - LF)**2 + np.outer(EL2, EF2) - np.outer(EL**2, EF**2))
    cond_logl = - 0.5 * (tau * R2 + n*d * np.log(2 * np.pi / tau))
    return cond_logl

def NM_posterior_e_loglik(x, mu, sigma2, Et, Et2):
    # Reference:
    # https://github.com/stephenslab/flashr/blob/master/R/objective.R
    e_loglik = -0.5 * np.sum(np.log(2*np.pi*sigma2) + (1/sigma2) * (mu**2 * Et2 - 2 * x * (mu * Et) + x**2))
    return e_loglik

def marginal_lik_F_func(pars, y_obs, sigma_y, mu_y, update_family):
    if update_family == 'nonparametric':
        log_lik = nonpar_e_loglik(pars, y_obs, sigma_y, mu_y)
    elif update_family == 'point-normal':
        log_lik = point_normal_e_loglik(pars, y_obs, sigma_y, mu_y)
    return log_lik

def nonpar_e_loglik(pars, y_obs, sigma_y, mu_y):
    log_lik = lambda pars: np.sum([np.log(np.inner(pars[0], _gaussian_pdf(yi, mu_y*pars[1], sigma_y))) for yi in y_obs])
    return log_lik(pars)

def point_normal_e_loglik(pars, y_obs, sigma_y, mu_y):
    # Reference:
    # https://github.com/stephenslab/flashr/blob/master/R/objective.R
    log_lik = np.sum(np.log((1 - pars[0]) * _gaussian_pdf(y_obs, 0, sigma_y) +
                    pars[0] * _gaussian_pdf(y_obs, 0, np.sqrt(sigma_y ** 2 + mu_y ** 2 * pars[1] ** 2))))
    return log_lik

if __name__ == '__main__':

    from ebpca.pca import get_pca
    from simulation.helpers import fill_alignment
    prior = 'Point_normal'
    s_star = 1.3
    i = 0
    rank = 1
    iters = 4

    prior_prefix = 'univariate/' + prior
    data_prefix = '../simulation/output/%s/data/s_%.1f' % (prior_prefix, s_star)
    u_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, i), allow_pickle=False)
    v_star = np.load('%s_copy_%i_v_star.npy' % (data_prefix, i), allow_pickle=False)
    X = np.load('%s_copy_%i.npy' % (data_prefix, i), allow_pickle=False)

    # prepare the PCA pack
    pcapack = get_pca(X, rank)

    print('\n No scaling \n')
    ldenoiser = NonparEB(optimizer="Mosek", to_save=False)
    fdenoiser = NonparEB(optimizer="Mosek", to_save=False)
    U_est, V_est, obj = ebmf(pcapack, ldenoiser, fdenoiser, iters=iters,
                             ebpca_scaling=False, update_family='nonparametric', tol=1e-1)
    # print alignment
    print(fill_alignment(U_est, u_star, iters))
    print(fill_alignment(V_est, v_star, iters))

    print('\n With scaling \n')
    U_est, V_est, obj = ebmf(pcapack, ldenoiser, fdenoiser, iters=iters,
                             ebpca_scaling=True, update_family='nonparametric', tol=1e-1)
    # print alignment
    print(fill_alignment(U_est, u_star, iters))
    print(fill_alignment(V_est, v_star, iters))

    # exchange U and V to make EBMF denoise V first
    pcapack_t = pcapack
    pcapack_t = pcapack_t._replace(U=pcapack.V)
    pcapack_t = pcapack_t._replace(V=pcapack.U)
    pcapack_t = pcapack_t._replace(X=np.transpose(pcapack.X))

    # print mu, sigma2 along iterations in EBMF
    U_ebmf, V_ebmf, _ = ebmf(pcapack_t, ldenoiser=fdenoiser, fdenoiser=ldenoiser,
                             iters=iters, ebpca_scaling=True, tau_by_row=False)

    # print alignment
    print(fill_alignment(U_est, u_star, iters))
    print(fill_alignment(V_est, v_star, iters))