import numpy as np
from ebpca.empbayes import PointNormalEB, _gaussian_pdf

# TODO
# 1. Make it compatible with nonparametric denoiser.
#    In particular, pay attention to the assumption of prior 2nd moment = 1
# 2. Implement objective function based convergence
def ebmf(X, u, v, iters = 5, par_update=True, ldenoiser = PointNormalEB(), fdenoiser = PointNormalEB(), tol = 1e-2):
    # get dimension
    (n, d) = X.shape

    # re-label u, v with l, f
    l_hat = u
    f_hat = v

    # initialize placeholder for l, f update results
    L = np.reshape(l_hat, (-1,1))
    F = np.reshape(f_hat, (-1,1))

    # use the same scaling as EBMF
    # initialize parameter tau
    tau = n
    # initialize the first update with svd
    mu = 1
    sigma_sq = 1 / (np.sum(f_hat**2) * tau)

    # first denoise the loadings
    l_hat = X.dot(f_hat) / np.sum(f_hat**2)
    obj_funcs = []
    t = 0
    flag = False
    while t < iters and (not flag):
        print("at ebmf iter {}".format(t))
        # denoise l_hat to get l
        ldenoiser.estimate_prior(l_hat, mu, np.sqrt(sigma_sq))
        El = ldenoiser.denoise(l_hat, mu, np.sqrt(sigma_sq))
        Varl = ldenoiser.ddenoise(l_hat, mu, np.sqrt(sigma_sq)) * (sigma_sq / mu)
        El2 = El**2 + Varl
        L = np.hstack((L, np.reshape(El,(-1,1))))
        # Evaluate log likelihood
        [pi, sigma_x] = ldenoiser.get_estimate()
        print('l prior: %.4f, %.4f' % (pi, sigma_x))
        KL_l = marginal_lik_F_func([pi, sigma_x],
                                   l_hat, np.sqrt(sigma_sq), mu, 'point-normal') - \
               NM_posterior_e_loglik(l_hat, mu, sigma_sq, El, El2)
        # update the estimate of the factor
        f_hat = X.T.dot(El) / np.sum(El2)
        mu_bar = 1
        sigma_bar_sq = 1 / (np.sum(El2) * tau)
        fdenoiser.estimate_prior(f_hat, mu_bar, np.sqrt(sigma_bar_sq))
        Ef = fdenoiser.denoise(f_hat, mu_bar, np.sqrt(sigma_bar_sq))
        Varf = fdenoiser.ddenoise(f_hat, mu_bar, np.sqrt(sigma_bar_sq)) * (sigma_bar_sq / mu_bar)
        Ef2 = Ef**2 + Varf
        F = np.hstack((F, np.reshape(Ef, (-1,1))))
        # Evaluate log likelihood
        [pi, sigma_x] = fdenoiser.get_estimate()
        print('f prior: %.4f, %.4f' % (pi, sigma_x))
        KL_f = marginal_lik_F_func([pi, sigma_x],
                                   f_hat, np.sqrt(sigma_bar_sq), mu_bar, 'point-normal') - \
               NM_posterior_e_loglik(f_hat, mu_bar, sigma_bar_sq, Ef, Ef2)
        # update the estimate of the loading
        l_hat = X.dot(Ef) / np.sum(Ef2)
        mu = 1
        sigma_sq = 1 / (np.sum(Ef2) * tau)
        # evaluate objective function
        obj_func = get_cond_logl(El, El2, Ef, Ef2, X, tau) + KL_l + KL_f
        obj_funcs.append(obj_func)
        print('Objective F function: {:.5f}'.format(obj_func))
        t += 1
        if t == 1 or t == 2:
            flag = False
        # else:
            # use accuracy as convergence threshold
        #     flag = abs(obj_funcs[-1] - obj_funcs[-2]) < tol

    if not flag:
        print('EBMF failed to converge within {} iterations.'.format(iters))

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