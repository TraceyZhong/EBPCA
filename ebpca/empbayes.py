'''
==========
Empirical Bayes Methods
==========
This module supports empirical Bayes estimation for various prior.
Typical usage example:
'''

from abc import ABC, abstractmethod

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numba import jit
import mosek
import mosek.fusion as fusion
import sys
import time


class _BaseEmpiricalBayes(ABC):
    '''Use empirical Bayes to estimate the prior and denoise observations.
    
    Given the prior family of X, mu, cov and Y s.t. Y ~ mu X + sqrt(cov) Z, estimate X.
    Attributes
    -----
    to_plot: 
    to_save:
    fig_prefix:
    Methods
    -----
    fit: (data, mu, cov) -> dist parameters
    denoise(f, prior_par) -> denoised data
    ddenoise(f, prior_par)-> derivated of the denoising functions evaluated at data 
    '''
    def __init__(self, to_save = True, to_show = False, fig_prefix = "figures/", tol = 1e-3):
        self.to_save = to_save
        self.to_show = to_show
        self.fig_prefix = fig_prefix
        self.tol = tol
        self.pi = None
        self.Z = None

    def fit(self, f, mu, cov, **kwargs):
        figname = kwargs.get("figname", "")
        self._check_init(f, mu)
        # if given true prior density,
        # plot marginal distribution computed with true prior
        self.Z = kwargs.get("Z_star", None)
        self.pi = kwargs.get("pi_star", None)
        if self.Z is not None:
            print('plot marginal with true prior')
            self.check_margin(f, mu, cov, figname + '_true_prior')
        # initialize denoiser parameters based on observed data
        self._check_init(f, mu)
        # estimate prior with NPMLE
        self.estimate_prior(f, mu, cov)
        # visualize marginal distribution based on estimated prior
        self.check_margin(f, mu, cov, figname)
        # TODO
        # add check prior
        # self.check_prior(figname)

    def _check_init(self, f, mu):
        # check initialization
        # TODO
        # check if AMP returns mu as a np.array
        self.rank = len(f.shape)
        self.nsample = len(f)
        self.nsupp = self.nsample
        self.pi = np.full((self.nsupp,), 1/self.nsupp)
        if self.rank == 1:
            self.Z = f / mu
        else:
            self.Z = f.dot(np.linalg.pinv(mu).T)

    def estimate_prior(self, f, mu, cov):
        if self.rank == 1:
            covInv = 1 / cov
        else:
            covInv = np.linalg.inv(cov)
        start = time.time()
        self.pi = _mosek_npmle(f, self.Z, mu, covInv, self.tol)
        end = time.time()
        print('mosek elapsed %.2f s' % (end - start))
        print('pi est max:{}, min:{}'.format(np.max(self.pi), np.min(self.pi)))
        return self.pi

    def get_margin_pdf(self, x, mu, cov, dim):
        if self.rank > 1:
            loc = np.array([mu.dot(z) for z in self.Z])[:, dim]
            scalesq = cov[dim, dim]
        else:
            loc = self.Z * mu
            scalesq = cov
        return np.sum(self.pi / np.sqrt(2 * np.pi * scalesq) * np.exp(-(x - loc) ** 2 / (2 * scalesq)))

    def check_margin(self, f, mu, cov, figname):
        fig, axes = plt.subplots(nrows=self.rank, ncols=1, figsize=(7, self.rank * 3))
        plt.subplots_adjust(hspace = 0.25)
        for dim in range(self.rank):
            if self.rank == 1:
                self.plot_margin_uni(axes, f, dim, mu, cov)
                axes.set_title("PC %i, mu=%.2f, cov=%.2f" % (dim, mu, cov))
            else:
                self.plot_margin_uni(axes[dim], f[:, dim], dim, mu, cov)
                axes[dim].set_title("PC %i, mu=%.2f, cov=%.2f" % \
                                    (dim, mu[dim, dim], cov[dim, dim]))
        # plt.suptitle('marginal distribution')
        if self.to_show:
            plt.show()
        if self.to_save:
            fig.savefig(self.fig_prefix + figname + '_marginal.png')
        plt.close()

    def plot_margin_uni(self, ax, f, dim, mu, cov):
        # span the grid to be plotted
        xmin = np.quantile(f, 0.05, axis=0)
        xmax = np.quantile(f, 0.95, axis=0)
        xgrid = np.linspace(xmin - abs(xmin) / 3, xmax + abs(xmax) / 3, num=100)
        # evaluate marginal pdf
        pdf = [self.get_margin_pdf(x, mu, cov, dim) for x in xgrid]
        ax.hist(f, bins=40, alpha=0.5, density=True, color="skyblue", label="empirical dist")
        ax.plot(xgrid, pdf, color="grey", linestyle="dashed", label="theoretical density")
        ax.legend()

    @abstractmethod
    def denoise(self, f, mu, cov):
        pass

    @abstractmethod
    def ddenoise(self, f, mu, cov):
        pass

class NonparEB(_BaseEmpiricalBayes):
    '''setting prior support points and estimating prior weights using NPMLE in mosek
    '''
    # TODO
    # change all sigma to cov
    def __init__(self, to_save = True, to_show = False, fig_prefix = "univar", tol = 1e-3):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix, tol)

    def denoise(self, f, mu, cov):
        if ((self.Z is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        return vf_nonpar(f, mu, np.sqrt(cov), (self.pi, self.Z))

    def ddenoise(self, f, mu, cov):
        if ((self.Z is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        return vdf_nonpar(f, mu, np.sqrt(cov), (self.pi, self.Z))

class NonparEBHD(_BaseEmpiricalBayes):

    def __init__(self, to_save=True, to_show=False, fig_prefix="multivar", tol=1e-3, **kwargs):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix, tol)

    def denoise(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        P = get_P(f, self.Z, mu, covInv, self.pi)
        return P @ self.Z

    def ddenoise(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        P = get_P(f, self.Z, mu, covInv, self.pi)
        ZouterMZ = np.einsum("ijk, kl -> ijl", matrix_outer(self.Z, self.Z.dot(mu.T)), covInv)
        E1 = np.einsum("ij, jkl -> ikl", P, ZouterMZ)
        E2a = P @ self.Z  # shape (I * rank)
        E2 = np.einsum("ijk, kl -> ijl", matrix_outer(E2a, E2a.dot(mu.T)), covInv)  # shape (I * rank)
        return E1 - E2


class PointNormalEB(_BaseEmpiricalBayes):

    def __init__(self, em_iter = 1000, to_save = True, to_show = False, fig_prefix = "pointnormaleb"):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)
        self.pi = 0.5
        self.mu_x = 0
        self.sigma_x = 1
        self.em_iter = em_iter
        self.tol = 1e-6

    def estimate_prior(self, f, mu, sigma):
        sigma_y = sigma
        mu_y = mu
        itr = 0
        stagnant = False
        while (itr <= self.em_iter) and not stagnant:
            p_y_point = (1-self.pi) * _gaussian_pdf(f, 0, sigma_y)
            sigma_y_tilde = np.sqrt(self.sigma_x**2 * mu_y**2 + sigma_y**2)
            p_y_normal = self.pi * _gaussian_pdf(f, 0, sigma_y_tilde)
            # M setp
            w_tilde = p_y_normal / (p_y_normal + p_y_point)
            new_pi = np.mean(w_tilde)
            sigma_x_tmp = np.inner(w_tilde, f**2) / np.sum(w_tilde)
            if sigma_x_tmp > sigma_y**2:
                new_sigma_x = np.sqrt(sigma_x_tmp - sigma_y**2) / mu_y
            else:
                new_sigma_x = 0
                # ('Squared sigma_x is estimated to be 0')
            itr += 1
            if (abs(new_pi - self.pi)/ max(new_pi, self.pi) < self.tol) and (abs(new_sigma_x - self.sigma_x)/ max(new_sigma_x, self.sigma_x)< self.tol) :
                print("stagnant at {}".format(itr))
                stagnant = True
            self.pi = new_pi
            self.sigma_x = new_sigma_x


    def denoise(self, f, mu, sigma):
        mu_y = mu
        sigma_y = sigma
        mu_y_tilde = PointNormalEB._eval_mu_y_tilde(self.mu_x, mu_y)
        sigma_y_tilde = PointNormalEB._eval_sigma_y_tilde(mu_y, self.sigma_x, sigma_y)
        mu_x_tilde = PointNormalEB._eval_mu_x_tilde(f, self.mu_x, self.sigma_x, mu_y, sigma_y)
        py = (1 - self.pi) * _gaussian_pdf(f, 0, sigma_y) + self.pi * _gaussian_pdf(f, mu_y_tilde, sigma_y_tilde)

        return (self.pi * _gaussian_pdf(f, mu_y_tilde, sigma_y_tilde) * mu_x_tilde / py)


    def ddenoise(self, f, mu, sigma):
        mu_y = mu
        sigma_y = sigma
        mu_y_tilde = PointNormalEB._eval_mu_y_tilde(self.mu_x, mu_y)
        sigma_y_tilde = PointNormalEB._eval_sigma_y_tilde(mu_y, self.sigma_x, sigma_y)
        mu_x_tilde = PointNormalEB._eval_mu_x_tilde(f, self.mu_x, self.sigma_x, mu_y, sigma_y)

        py = (1 - self.pi) * _gaussian_pdf(f, 0, sigma_y) + self.pi * _gaussian_pdf(f, mu_y_tilde, sigma_y_tilde)

        # phi(y; mu_y_tilde, sigma_y_tilde)
        phi = _gaussian_pdf(f, mu_y_tilde, sigma_y_tilde)

        # derivative of p(y)
        d_py = (1 - self.pi) * _gaussian_pdf(f, 0, sigma_y) * (- f / sigma_y**2) + self.pi * phi * (- (f - mu_y_tilde) / sigma_y_tilde**2)

        # derivative of phi(y; mu_y_tilde, sigma_y_tilde) * mu_x_tilde
        d_tmp = (phi * (- (f - mu_y_tilde) / sigma_y_tilde**2)) * mu_x_tilde + phi * (mu_y * self.sigma_x**2) / (mu_y**2 * self.sigma_x**2 + sigma_y**2)

        return self.pi * (- phi * mu_x_tilde / py**2 * d_py + 1 / py * d_tmp)


    def get_margin_pdf(self, mu, sigma, x):
        mu_y = mu
        sigma_y = sigma
        mu_y_tilde = PointNormalEB._eval_mu_y_tilde(self.mu_x, mu_y)
        sigma_y_tilde = PointNormalEB._eval_sigma_y_tilde(mu_y, self.sigma_x, sigma_y)
        py = (1 - self.pi) * _gaussian_pdf(x, 0, sigma_y) + self.pi * _gaussian_pdf(x, mu_y_tilde, sigma_y_tilde)
        return py

    @staticmethod
    def _eval_mu_y_tilde(mu_x, mu_y):
        return mu_x * mu_y

    @staticmethod
    def _eval_mu_x_tilde( y, mu_x, sigma_x, mu_y, sigma_y):
        return (y * mu_y * sigma_x**2 + mu_x * sigma_y**2) / (mu_y**2 * sigma_x**2 + sigma_y**2)

    @staticmethod
    def _eval_sigma_y_tilde(mu_y, sigma_x, sigma_y):
        return np.sqrt(sigma_y**2 + mu_y**2 * sigma_x**2)

    @staticmethod
    def _eval_sigma_x_tilde(mu_y, sigma_x, sigma_y):
        return sigma_x * sigma_y * np.sqrt(1 / (mu_y**2 * sigma_x**2 + sigma_y**2))


def vf_nonpar(y, mu, sigma, prior_pars):
    """
    vectorized denoiser f under the discrete prior (x,pi)
    Computes the Bayes estimator (posterior mean) of y0 according to the model below
    input:
        y: a (n,) np ndarray; y|x_i \sim N(\gamma y0, \gamma)
        x: {x_1, \dots, x_n_np}, the support set for the nonparametric prior
        pi: probabilites for each point in the support set
        mu,sigma: parameters in y_obs|U \sim N(mu U, sigma**2) (similar for V)
    return:
        denoised y, as an updated estimate of y0
        an (n,) np ndarray
    """
    pi, x = prior_pars
    def f(y, x, pi,  mu, sigma):
        phi = np.exp(-(y - mu * x)**2 / (2*sigma**2))
        return np.inner(x*pi, phi) / np.inner(pi, phi)

    return np.asarray([f(yi, x, pi, mu, sigma) for yi in y])

def vdf_nonpar(y, mu, sigma, prior_pars):
    '''
    derivative of the denoiser under the discrete prior (x,pi)
    Computes the derivative of the Bayes estimator (posterior mean) of y0 according to the model in vf()
    input:
        y: a (n,) np ndarray; y|x_i \sim N(\gamma y0, \gamma)
        x: {x_1, \dots, x_n_np}, the support set for the nonparametric prior
        pi: probabilites for each point in the support set
        mu,sigma: parameters in y_obs|U \sim N(mu U, sigma**2) (similar for V)
    return:
        the derivate of posterior mean
        an (n,) np ndarray
    '''
    pi, x = prior_pars
    def df(y, x, pi, mu, sigma):
        phi = np.exp(-(y - mu * x)**2 / (2*sigma**2))
        E1 = np.inner(x*pi, phi) / np.inner(pi, phi)
        E2 = mu*np.inner(x**2*pi, phi)/(sigma**2) / np.inner(pi, phi)
        return (E2 - E1**2*mu/(sigma**2))
    
    return np.array([df(yi, x, pi, mu, sigma) for yi in y])


def _gaussian_pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi)) * (1 / sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))

## --- high dim funcs --- ##

@jit(nopython = True)
def my_dot(mat, vec):
    nrow, ncol = mat.shape
    res = np.array([0.0] * nrow)
    for i in range(nrow):
        for j in range(ncol):
            res[i] += mat[i, j] * vec[j]
    return res

@jit(nopython=True)
def _npmle_em(W, em_iter, nsupp):
    pi = np.array([1/nsupp] * nsupp)
    for _ in range(em_iter):
        denom = my_dot(W, pi)
        for j in range(nsupp):
            pi[j] = pi[j]*np.mean(W[:,j]/denom)
    return pi

def get_W(f, z, mu, covInv):
    '''
    Compute conditional likelihood
    '''
    if len(f.shape) > 1:
        W = get_W_multivar(f, z, mu, covInv)
    else:
        W = get_W_univar(f, z, mu, covInv)
    return W

def get_W_univar(f, z, mu, covInv):
    '''
    Compute conditional likelihood for univariate problem
    '''
    nsample = f.shape[0]
    nsupp = z.shape[0]
    W = np.empty(shape=(nsample, nsupp), )
    for i in range(nsample):
        for j in range(nsupp):
            vec = f[i] - mu * z[j]
            res = np.exp(-np.sum(covInv * vec ** 2) / 2)
            W[i, j] = res
    return W

# W[i,j] = f(x_i | z_j)
@jit(nopython = True)
def get_W_multivar(f, z, mu, covInv):
    '''
    Compute conditional likelihood for multivariate problem
    numba is used to speed up the computation
    '''
    nsample = f.shape[0]
    nsupp = z.shape[0]
    W = np.empty(shape = (nsample, nsupp),)
    for i in range(nsample):
        for j in range(nsupp):
            vec = f[i] - my_dot(mu, z[j])
            res = np.exp(-np.sum(my_dot(covInv, vec) * vec)/2)
            W[i,j] = res
    return W

# P[i,j] = P(Z_j | X_i)
def get_P(f,z,mu,covInv, pi):
    W = get_W(f,z,mu,covInv)
    denom = W.dot(pi) # denom[i] = \sum_j pi[j] * W[i,j]
    num = W * pi # W*pi[i,j] = pi[j] * W[i,j]
    return num / denom[:, np.newaxis]


# W*pi[i,j] = pi[j] * W[i,j] 
# sum_j pi[j] W[i,j] = np.sum(W*pi, axis = 1) 
@jit(nopython = True)
def negloglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    return -np.sum(np.log( np.sum(W * pi, axis = 1)))

@jit(nopython = True)
def loglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    return np.sum(np.log( np.sum(W * pi, axis = 1)))

# @jit(nopython = True)
def dnegloglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    res = np.sum( W / np.sum(W*pi, axis = 1)[:,np.newaxis], axis = 0)
    if np.isnan(res).any():
        raise ValueError("dloglik has nan value")
    return -res
    # -np.maximum(np.minimum(res, MAX_FLOAT), MIN_FLOAT)


# @jit(nopython = True)
def ddnegloglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    normedW = W / np.sum(W*pi, axis = 1)[:,np.newaxis]
    # return my_m_dot(np.transpose(normedW), normedW)
    res = normedW.T @ normedW
    if np.isnan(res).any():
        raise ValueError("dloglik has nan value")
    return res
    # np.maximum(np.minimum(res, MAX_FLOAT), MIN_FLOAT)

matrix_outer = lambda A, B: np.einsum("bi,bo->bio", A, B)

def span_grid(xx, yy):
    xv, yv = np.meshgrid(xx, yy, sparse = False)
    return np.dstack([xv, yv]).reshape((len(xx)*len(yy),2))

def _mosek_npmle(f, Z, mu, covInv, tol):
    A = get_W(f, Z, mu, covInv)
    n, m = A.shape

    # ignorant: use observed points as support points
    # objective function: the primal in Section 4.2,
    # https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.869224
    M = fusion.Model('NPMLE')
    # https://docs.mosek.com/9.2/pythonapi/solver-parameters.html
    M.getTask().putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
    print('mosek tolerance: %f' % M.getTask().getdouparam(mosek.dparam.intpnt_co_tol_rel_gap) )
    logg = M.variable(n)
    g = M.variable('g', n, fusion.Domain.greaterThan(0.))  # w = exp(v)
    f = M.variable('f', m, fusion.Domain.greaterThan(0.))
    ones = np.repeat(1.0, n)
    ones_m = np.repeat(1.0, m)
    M.constraint(fusion.Expr.sub(fusion.Expr.dot(ones_m, f), 1), fusion.Domain.equalsTo(0.0))
    M.constraint(fusion.Expr.sub(fusion.Expr.mul(A, f), g), fusion.Domain.equalsTo(0.0, n))  # , Expr.constTerm(m, n * 1.0)
    M.constraint(fusion.Expr.hstack(g, fusion.Expr.constTerm(n, 1.0), logg), fusion.Domain.inPExpCone())

    # M.setLogHandler(sys.stdout)
    M.objective(fusion.ObjectiveSense.Maximize, fusion.Expr.dot(ones, logg))
    M.solve()

    symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
    print("   Termination code: {0} {1}".format(symname, desc))

    pi = f.level()

    # normalize the negative values due to numerical issues
    print('Minimal pi value: {:.2f}'.format(np.min(pi)))
    print('Sum of estimated pi: {:.2f}'.format(np.sum(pi)))
    # address negative values due to numerical instability
    pi[pi < 0] = 0
    pi = pi / np.sum(pi)

    return pi
