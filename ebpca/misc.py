'''
==========
EBMF method
==========
This module includes an implementation of the method in https://arxiv.org/abs/1802.06931 ,
  which estimates PCs with a Variational Bayes method.
We include it here to compare with EB-PCA method.

The specific function ebmf() here runs for designated number of iterations.
Note that we also implement evaluation of objective function,
  which can be used to determined if ebmf is converged or not within maximum iterations.

Input:
    pcapack: output from pca.get_pca
    ldenoiser: a PointNormalEB or NonparEB denoiser
    fdenoiser: a PointNormalEB or NonparEB denoiser
    update_family: use Point Normal model or nonparametric model to evaluate objective function,
                    corresponding to the choice of denoiser
    iters: maximum number of iterations
    tol: 1e-2, the same as the default setting in flashr
    ebpca_scaling: whether or not apply scaling to make observations on the same scale with EB-PCA
                   for plotting purpose (Figure 2) only;
                   Note that re-scaling doesn't affect the estimates of EBMF, as EBMF model does not
                   impose any assumption on the scaling of initialization right / left PC

Remarks:
    1. This implementation support nonparametric priors besides parametric priors.
    2. A test showing that this implementation is the same as flashr R package is in #???

Reference:
    https://arxiv.org/abs/1802.06931
    https://github.com/stephenslab/flashr

Typical usage example:
    L, F = ebmf(pcapack, update_family = 'nonparametric', iters=50)
'''

import numpy as np
from ebpca.empbayes import _gaussian_pdf, NonparEB

def ebmf(pcapack, ldenoiser = NonparEB(), fdenoiser = NonparEB(),
         update_family = 'nonparametric', iters = 50, tol=1e-1, ebpca_scaling=True):

    X = pcapack.X
    u, v = pcapack.U, pcapack.V
    # get dimension
    (n, d) = X.shape

    if ebpca_scaling:
        print('Apply rescaling to match the scale with EB-PCA in marginal plots')
        # get signal
        signals = pcapack.signals
        # apply the same scaling in EB-PCA
        u = u / np.sqrt((u ** 2).sum(axis=0)) * np.sqrt(n)
        v = v / np.sqrt((v ** 2).sum(axis=0)) * np.sqrt(d)
        mu_constant = np.float(signals / n)
    else:
        mu_constant = 1

    # re-label u, v with l, f
    l_hat = u
    f_hat = v

    # initialize placeholder for l, f update results
    L = l_hat[:,:, np.newaxis]
    F = f_hat[:,:, np.newaxis]

    # use the same scaling as EBMF
    # initialize parameter tau
    tau = n
    # initialize the first update with svd
    # first denoise the loadings
    l_hat = X.dot(f_hat) / np.sum(f_hat ** 2)
    mu = np.diag([mu_constant])
    sigma_sq = np.diag([1 / (np.sum(f_hat ** 2) * tau)])
    if ebpca_scaling:
        l_hat = l_hat * np.sum(f_hat ** 2)
        mu = mu * np.sum(f_hat ** 2)
        sigma_sq = sigma_sq * np.sum(f_hat ** 2)**2

    obj_funcs = []
    t = 0
    new_flag = False
    while t < iters:
        old_flag = new_flag
        print("at ebmf iter {}".format(t))
        # Denoise l_hat to get l
        ldenoiser.fit(l_hat, mu, sigma_sq, figname='_u_iter%02d.png' % (t))
        El = ldenoiser.denoise(l_hat, mu, sigma_sq)
        Varl = ldenoiser.ddenoise(l_hat, mu, sigma_sq) * (sigma_sq / mu)
        El2 = El**2 + Varl.reshape(-1,1) #[:,:,0]
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
        fdenoiser.fit(f_hat, mu_bar, sigma_bar_sq, figname='_v_iter%02d.png' % (t))
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
            mu = mu_bar * np.sum(Ef2)
            sigma_sq = sigma_sq * np.sum(Ef2)**2
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
