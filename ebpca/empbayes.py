'''
==========
Empirical Bayes Methods
==========
This module supports empirical Bayes estimation for various prior.
'''

import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from numba import jit
import mosek
import mosek.fusion as fusion
from scipy import optimize

class _BaseEmpiricalBayes(ABC):
    
    def __init__(self, to_save = False, to_show = False, fig_prefix = ""):
        self.to_save = to_save
        self.to_show = to_show
        self.fig_prefix = "figures/"+fig_prefix
        self.rank = 0
        self.iter = 1
        self.plot_scaled = True

    @abstractmethod
    def estimate_prior(self,f, mu, cov):
        pass

    @abstractmethod
    def denoise(self, f, mu, cov):
        pass 

    @abstractmethod
    def ddenoise(self, f, mu, cov):
        pass 
    
    @abstractmethod
    def get_margin_pdf(self, x, mu, cov, dim):
        pass 

    @abstractmethod
    def check_prior(self, figname):
        pass

    def fit(self, f, mu, cov, **kwargs):
        self.estimate_prior(f,mu,cov)
        # if np.all(self.pi == 0):
        #     return 'error'
        figname = kwargs.get("figname", "")
        if (self.to_show or self.to_save):
            self.check_margin(f,mu,cov,figname)
            # self.check_prior(figname)
        self.iter += 1
        return 'normal'

    def check_margin(self, fs, mu, cov, figname):
        plt.rcParams.update({'font.size': 16})
        if self.rank == 1:
            fig, ax = plt.subplots(nrows=self.rank, ncols=1, figsize=(7, 3))
            axes = [ax]

        if self.rank > 1:
            fig, axes = plt.subplots(ncols=1, nrows=self.rank+1, figsize = (7, 3*self.rank+2), \
                gridspec_kw={'height_ratios': [3]*self.rank +[2] }, constrained_layout=True)
            # last row shows M and Sigma
            ax = axes[-1]
            ax.axis("off")
            writeMat(ax, mu, "M", vCenter = 0.25)
            writeMat(ax, cov, r"$\Sigma$", vCenter = 0.75)

        
        for dim in range(self.rank):
            self.plot_each_margin(axes[dim], fs[:, dim], dim, mu, cov)
            if self.iter > 1:
                axes[dim].get_legend().remove()
            if self.rank > 1:
                axes[dim].set_title("Iteration %i, %s%i, mu=%.2f, cov=%.2f" % \
                                    (self.iter, self.PCname, dim + 1, mu[dim, dim], cov[dim, dim]))
            else:
                axes[dim].set_title("Iteration %i, %s, SNR=%.2f" % \
                                    (self.iter, self.PCname, (mu[dim, dim])**2/ (cov[dim, dim])))
        if self.to_show:
            plt.show()
        if self.to_save:
            fig.savefig(self.fig_prefix +figname)
        plt.close()

    def plot_each_margin(self, ax, f, dim, mu, cov):
        # span the grid to be plotted
        xmin = np.quantile(f, 0.0, axis=0)
        xmax = np.quantile(f, 1.0, axis=0)
        xgrid = np.linspace(xmin - abs(xmin) / 3, xmax + abs(xmax) / 3, num=100)
        # evaluate marginal pdf
        pdf = [self.get_margin_pdf(x, mu, cov, dim) for x in xgrid]
        ax.hist(f, bins=40, alpha=0.5, density=True, color="skyblue", label="empirical dist")
        ax.plot(xgrid, pdf, color="grey", linestyle="dashed", label="theoretical density")
        ax.legend()

class NonparEB(_BaseEmpiricalBayes):
    
    def __init__(self, optimizer = "EM", PCname = 'U', ftol = 1e-8, nsupp_ratio = 1, em_iter = 100, maxiter = 100, to_save = False, to_show = False, fig_prefix = "nonpareb", **kwargs):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)
        # check if parameters are valid
        if optimizer in ["EM", "Mosek"]:
            self.optimizer = optimizer
        else:
            raise ValueError("Supported Optimizers are EM or Mosek.")
        self.nsample = None
        self.nsupp = None
        self.em_iter = em_iter
        self.ftol = ftol
        self.maxiter = maxiter
        self.nsupp_ratio = nsupp_ratio
        self.pi = None
        self.Z = None
        self.P = None
        self.PCname = PCname

    def _check_init(self, f, mu, cov):
        self.rank = len(mu)
        self.nsample = len(f)
        self.nsupp = int(self.nsupp_ratio * self.nsample)
        self.pi = np.full((self.nsupp,),1/self.nsupp)
        if self.nsupp_ratio == 1:
            self.Z = f.dot(np.linalg.pinv(mu).T)
        else:
            self.Z = f[np.random.choice(f.shape[0], self.nsupp, replace=False), :].dot(np.linalg.pinv(mu).T)

    def check_prior(self, figname):
        # can only be used when two dimension
        if self.rank == 2:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7, 5))
            ax.scatter(self.Z[:,0], self.Z[:,1], s = self.pi*len(self.pi),marker = ".")
            ax.set_title("check prior, {}".format(figname))
            if self.to_show:
                plt.show()
            if self.to_save:
                fig.savefig(self.fig_prefix + "_prior" +figname)
            plt.close()         
    
    def estimate_prior(self,f, mu, cov):
        # check initialization  
        self._check_init(f,mu,cov)
        covInv = np.linalg.inv(cov)
        if self.optimizer == "EM":
            self.pi, W = npmle_em_hd2(f, self.Z, mu, covInv, self.em_iter)
        if self.optimizer == "Mosek":
            self.pi, W = mosek_npmle(f, self.Z, mu, covInv, self.ftol)
            # use EM as fallback if encountered error in MOSEK
            if np.all(self.pi == 0):
                self.pi, W = npmle_em_hd2(f, self.Z, mu, covInv, self.em_iter)
        self.P = get_P_from_W(W, self.pi)
        del W

    def get_margin_pdf(self, x, mu, cov, dim):
        if self.plot_scaled:
            x = x / (mu[dim, dim])
            loc = self.Z[:, dim]
            scalesq = cov[dim, dim] / ((mu[dim, dim])**2)
        else:
            loc = self.Z.dot(mu.T)[:, dim]
            scalesq = cov[dim, dim]
        return np.sum(self.pi / np.sqrt(2 * np.pi * scalesq) * np.exp(-(x - loc) ** 2 / (2 * scalesq)))
    
    def denoise(self, f, mu, cov):
        if self.P is None:
            covInv = np.linalg.inv(cov)
            self.P = get_P(f,self.Z, mu, covInv, self.pi)
        return self.P @ self.Z 

    def ddenoise(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        if self.P is None:
            self.P = get_P(f, self.Z, mu, covInv, self.pi)
        ZouterMZ = np.einsum("ijk, kl -> ijl" ,matrix_outer(self.Z, self.Z.dot(mu.T)), covInv) 
        E1 = np.einsum("ij, jkl -> ikl", self.P, ZouterMZ)
        E2a = self.P @ self.Z # shape (I * rank)
        E2 = np.einsum("ijk, kl -> ijl" ,matrix_outer(E2a, E2a.dot(mu.T)), covInv)  # shape (I * rank)
        del self.P
        self.P = None
        return E1 - E2

    def get_estimate(self):
        return self.pi, self.Z

class NonparEBChecker(NonparEB):
    def __init__(self, truePriorLoc, truePriorWeight, optimizer = "EM",
                 PCname = 'U', histcol = 'skyblue', xRange = None, yRange=None,
                 ftol = 1e-6, nsupp_ratio = 1, em_iter = 10, maxiter = 100, to_save = False, to_show = False, fig_prefix = "nonparebck", **kwargs):
        NonparEB.__init__(self, optimizer, PCname, ftol, nsupp_ratio, em_iter, maxiter, to_save, to_show, fig_prefix, **kwargs)
        self.trueZ = truePriorLoc
        self.truePi = truePriorWeight
        self.histcol = histcol
        self.redirectd = False
        self.xRange = xRange
        self.yRange = yRange

    def get_margin_pdf_from_true_prior(self, x, mu, cov, dim):
        if self.plot_scaled:
            x = x / mu[dim, dim]
            loc = self.trueZ[:, dim]
            scalesq = (cov[dim, dim]) / ((mu[dim, dim]) ** 2)
        else:
            loc = self.trueZ.dot(mu.T)[:, dim]
            scalesq = cov[dim, dim]
        return np.sum(self.truePi / np.sqrt(2 * np.pi * scalesq) * np.exp(-(x - loc) ** 2 / (2 * scalesq)))

    def plot_each_margin(self, ax, f, dim, mu, cov):
        xmin = np.quantile(f, 0.0, axis=0)
        xmax = np.quantile(f, 1.0, axis=0)
        xgrid = np.linspace(xmin - abs(xmin) / 3, xmax + abs(xmax) / 3, num=1000)
        # evaluate marginal pdf
        pdf = [self.get_margin_pdf(x, mu, cov, dim) for x in xgrid]
        # print("why is this not correct")
        # print(self.trueZ.dot(mu.T)[:, dim])
        # print(self.truePi)
        truePdf = [self.get_margin_pdf_from_true_prior(x, mu, cov, dim) for x in xgrid]
        # plot scaled marginal:
        if self.plot_scaled:
            f = f / (mu[dim, dim])
            xgrid = xgrid / (mu[dim, dim])
        ax.hist(f, bins=40, alpha=0.4, density=True, color=self.histcol, label="Empirical obs")
        # ax.plot(xgrid, pdf, color="grey", linestyle="dashed", linewidth=2, label="fitted model")
        ax.plot(xgrid, truePdf, color="grey", linestyle="solid", linewidth=1, label="Theoretical density")
        if self.xRange is not None:
            ax.set_xlim(self.xRange[0], self.xRange[1])
            ax.set_ylim(self.yRange[0], self.yRange[1])
        ax.legend()
            
class NonparBayes(NonparEB):
    def __init__(self, truePriorLoc, truePriorWeight, optimizer = "EM", PCname = 'U', ftol = 1e-6, nsupp_ratio = 1, em_iter = 10, maxiter = 100, to_save = False, to_show = False, fig_prefix = "nonparebck", **kwargs):
        NonparEB.__init__(self, optimizer, PCname, ftol, nsupp_ratio, em_iter, maxiter, to_save, to_show, fig_prefix, **kwargs)
        self.Z = truePriorLoc
        self.pi = truePriorWeight
        self.rank = truePriorLoc.shape[1]
    
    def estimate_prior(self, f,mu,cov):
        pass

class PointNormalEB(_BaseEmpiricalBayes):

    def __init__(self, to_save = True, to_show = False, fig_prefix = "pointnormaleb"):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)
        self.rank = 1 # currently the point normal denoiser only supports univariate denoising
        self.pi = 0.5
        self.mu_x = 0
        self.sigma_x = 1
        self.tol = 1e-6

    def estimate_prior(self, f, mu, sigma_sq):
        sigma = np.sqrt(sigma_sq)
        # solve for point normal parameters with constrained optimization
        neg_log_lik = lambda pars: \
            -np.sum([np.logaddexp(np.log(1 - pars[0]) + _log_gaussian_pdf(yi, 0, sigma),
                                  np.log(pars[0]) + _log_gaussian_pdf(yi, 0, np.sqrt(sigma ** 2 + mu ** 2 * pars[1] ** 2)))
                     for yi in f])
        # constrain
        bnds = [(1e-6, 1 - 1e-6), (0.001e-6, None)]
        # initial parameters
        init_parameters = np.asarray([0.5, 1])
        # Minimizing neg_log_lik
        results = optimize.minimize(neg_log_lik, x0=init_parameters,
                                    method='L-BFGS-B', bounds=bnds,  # SLSQP
                                    options={'ftol': 1e-6, 'disp': False, 'maxiter': 100})
        self.pi, self.sigma_x = results.x

    def get_estimate(self):
        return self.pi, self.sigma_x

    def denoise(self, f, mu, sigma_sq):
        sigma = np.sqrt(sigma_sq)
        mu_y = mu
        sigma_y = sigma
        mu_y_tilde = PointNormalEB._eval_mu_y_tilde(self.mu_x, mu_y)
        sigma_y_tilde = PointNormalEB._eval_sigma_y_tilde(mu_y, self.sigma_x, sigma_y)
        mu_x_tilde = PointNormalEB._eval_mu_x_tilde(f, self.mu_x, self.sigma_x, mu_y, sigma_y)
        py = (1 - self.pi) * _gaussian_pdf(f, 0, sigma_y) + self.pi * _gaussian_pdf(f, mu_y_tilde, sigma_y_tilde)

        return (self.pi * _gaussian_pdf(f, mu_y_tilde, sigma_y_tilde) * mu_x_tilde / py)


    def ddenoise(self, f, mu, sigma_sq):
        sigma = np.sqrt(sigma_sq)
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

        return (self.pi * (- phi * mu_x_tilde / py**2 * d_py + 1 / py * d_tmp))[:, np.newaxis]


    def get_margin_pdf(self, x, mu, sigma_sq, dim = 0):
        sigma = np.sqrt(sigma_sq)
        mu_y = mu
        sigma_y = sigma
        mu_y_tilde = PointNormalEB._eval_mu_y_tilde(self.mu_x, mu_y)
        sigma_y_tilde = PointNormalEB._eval_sigma_y_tilde(mu_y, self.sigma_x, sigma_y)
        py = (1 - self.pi) * _gaussian_pdf(x, 0, sigma_y) + self.pi * _gaussian_pdf(x, mu_y_tilde, sigma_y_tilde)
        py = py.reshape(-1)
        return py

    def check_prior(self, figname):
        pass

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


class PointNormalBayes(PointNormalEB):
    def __init__(self, truePi, trueSigma_x, to_save = True, to_show = False, fig_prefix = "pointnormalbayes", **kwargs):
        PointNormalEB.__init__(self, to_save, to_show, "pointnormaleb", **kwargs)
        self.pi = truePi
        self.sigma_x = trueSigma_x
        self.rank = 1 # currently the point normal bayes denoiser only supports univariate denoising

    def estimate_prior(self, f, mu, cov):
        pass

class TwoPointsBayes(_BaseEmpiricalBayes):
    def __init__(self, to_save = True, to_show = False, fig_prefix = "twopointsbayes"):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)
        self.rank = 1 # currently the point normal denoiser only supports univariate denoising

    def estimate_prior(self, f, mu, cov):
        pass

    def denoise(self, f,  mu, cov):
        return np.tanh(f * mu / cov)

    def ddenoise(self, f, mu, cov):
        dd = (1 - np.tanh(f * mu / cov)**2) * mu / cov
        return dd[:,np.newaxis]

    def get_margin_pdf(self, x,  mu, cov, dim = 0):
        def two_point_normal_pdf(x, mu, sigmasq):
            pdf_plus = 1 / np.sqrt(2 * np.pi * sigmasq) * np.exp(-(x - mu) ** 2 / (2 * sigmasq))
            pdf_minus = 1 / np.sqrt(2 * np.pi * sigmasq) * np.exp(-(x + mu) ** 2 / (2 * sigmasq))
            return 0.5 * pdf_plus + 0.5 * pdf_minus

        ygrid = two_point_normal_pdf(x, mu, cov)
        return ygrid

    def check_prior(self, figname):
        pass


def jit_npmle_em_hd(f, Z, mu, covInv, em_iter, nsample, nsupp):
    # pi = np.full((nsupp,), 1/nsupp, dtype= float)
    pi = np.array([1/nsupp] * nsupp)
    
    W = get_W(f, Z, mu, covInv)

    return em(W,pi,nsupp,em_iter)



@jit(nopython=True)
def em(W, pi, nsupp,em_iter):
    for _ in range(em_iter):
        denom = my_dot(W, pi) # denom[i] = \sum_j pi[j]*W[i,j]

        for j in range(nsupp):
            pi[j] = pi[j]*np.mean(W[:,j]/denom)
    
    return pi



@jit(nopython=True)
def _npmle_em_hd(f, Z, mu, covInv, em_iter, nsample, nsupp, ndim):
    # pi = np.full((nsupp,), 1/nsupp, dtype= float)
    pi = np.array([1/nsupp] * nsupp)
    
    W = get_W(f, Z, mu, covInv)

    for _ in range(em_iter):
        denom = my_dot(W, pi) # denom[i] = \sum_j pi[j]*W[i,j]

        for j in range(nsupp):
            pi[j] = pi[j]*np.mean(W[:,j]/denom)
    
    return pi, W

def npmle_em_hd(f, Z, mu, covInv, em_iter):
    # I dont need this n_dim
    nsupp = Z.shape[0]
    pi = np.array([1/nsupp] * nsupp)
    W = get_W(f, Z, mu, covInv)

    for _ in range(em_iter):
        denom = W.dot(pi)[:,np.newaxis] # denom[i] = \sum_j pi[j]*W[i,j]
        pi = pi * np.mean(W/denom, axis = 0)
    
    return pi, W

def npmle_em_hd2(f, Z, mu, covInv, em_iter):
    # I dont need this n_dim
    nsupp = Z.shape[0]
    pi = np.array([1/nsupp] * nsupp)
    W = get_W(f, Z, mu, covInv)

    # for _ in range(em_iter):
    #     denom = W.dot(pi)[:,np.newaxis] # denom[i] = \sum_j pi[j]*W[i,j]
    #     pi = pi * np.mean(W/denom, axis = 0)
    #     # pi = pi * np.mean(np.divide(W, denom), axis = 0)
    Wt = np.array(W.T, order = 'C')
    for _ in range(em_iter):
        denom = W.dot(pi)# [:,np.newaxis] # denom[i] = \sum_j pi[j]*W[i,j]
        pi = pi * np.mean(Wt/denom, axis = 1)
    
    return pi, W



def _gaussian_pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi)) * (1 / sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def _log_gaussian_pdf(x, mu, sigma):
    return - np.log(sigma) - 0.5 * np.log(2 * np.pi) - (x - mu) ** 2 / (2 * sigma ** 2)

## --- high dim funcs --- ##

@jit(nopython = True)
def my_dot(mat, vec):
    nrow, ncol = mat.shape
    res = np.array([0.0] * nrow)
    for i in range(nrow):
        for j in range(ncol):
            res[i] += mat[i, j] * vec[j]
    return res

# W[i,j] = f(x_i | z_j)
@jit(nopython = True)
def jit_get_W(f, z, mu, covInv):
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

# consider another get W using broadcast
def get_W(f, z, mu, covInv):
    fsq = (np.einsum("ik,ik -> i", f @ covInv, f) / 2)[:,np.newaxis]
    mz = z.dot(mu.T)
    zsq = np.einsum("ik, ik->i", mz @ covInv, mz) / 2
    fz = f @ covInv @ mz.T
    del mz
    return np.exp(- fsq + fz - zsq)
    

# P[i,j] = P(Z_j | X_i)
def get_P(f,z,mu,covInv,pi):
    W = get_W(f,z,mu,covInv)
    denom = W.dot(pi) # denom[i] = \sum_j pi[j] * W[i,j]
    num = W * pi # W*pi[i,j] = pi[j] * W[i,j]
    return num / denom[:, np.newaxis]

def get_P_from_W(W, pi):
    denom = W.dot(pi) # denom[i] = \sum_j pi[j] * W[i,j]
    num = W * pi # W*pi[i,j] = pi[j] * W[i,j]
    return num / denom[:, np.newaxis]    

matrix_outer = lambda A, B: np.einsum("bi,bo->bio", A, B)

def span_grid(xx, yy):
    xv, yv = np.meshgrid(xx, yy, sparse = False)
    return np.dstack([xv, yv]).reshape((len(xx)*len(yy),2))

def mosek_npmle(f, Z, mu, covInv, tol=1e-8):
    A = get_W(f, Z, mu, covInv)
    n, m = A.shape

    # objective function: the primal in Section 4.2,
    # https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.869224
    with fusion.Model('NPMLE') as M:
        # set tolerance parameter
        # https://docs.mosek.com/9.2/pythonapi/solver-parameters.html
        M.getTask().putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
        # print('mosek tolerance: %f' % M.getTask().getdouparam(mosek.dparam.intpnt_co_tol_rel_gap) )
        logg = M.variable(n)
        g = M.variable('g', n, fusion.Domain.greaterThan(0.))  # w = exp(v)
        f = M.variable('f', m, fusion.Domain.greaterThan(0.))
        ones = np.repeat(1.0, n)
        ones_m = np.repeat(1.0, m)
        M.constraint(fusion.Expr.sub(fusion.Expr.dot(ones_m, f), 1), fusion.Domain.equalsTo(0.0))
        M.constraint(fusion.Expr.sub(fusion.Expr.mul(A, f), g), fusion.Domain.equalsTo(0.0, n))
        M.constraint(fusion.Expr.hstack(g, fusion.Expr.constTerm(n, 1.0), logg), fusion.Domain.inPExpCone())

        # uncomment to enable detailed log
        # M.setLogHandler(sys.stdout)

        # default value if MOSEK gives an error
        pi = np.repeat(0, m)

        M.objective(fusion.ObjectiveSense.Maximize, fusion.Expr.dot(ones, logg))

        # response handling for Mosek solutions
        # modified from https://docs.mosek.com/9.2/pythonfusion/errors-exceptions.html
        try:
            M.solve()
            # https://docs.mosek.com/9.2/pythonfusion/enum_index.html#accsolutionstatus
            M.acceptedSolutionStatus(fusion.AccSolutionStatus.Optimal) # Anything
           # print(" Accepted solution setting:", M.getAcceptedSolutionStatus())
            pi = f.level()
            if not np.all(pi==0):
                # address negative values due to numerical instability
                pi[pi < 0] = 0
                # normalize the negative values due to numerical issues
                pi = pi / np.sum(pi)

        except fusion.OptimizeError as e:
            print(" Optimization failed. Error: {0}".format(e))

        except fusion.SolutionError as e:
            # The solution with at least the expected status was not available.
            # We try to diagnoze why.
            print("  Error messages from MOSEK: \n  Requested NPMLE solution was not available.")
            prosta = M.getProblemStatus()

            if prosta == fusion.ProblemStatus.DualInfeasible:
                print("  Dual infeasibility certificate found.")

            elif prosta == fusion.ProblemStatus.PrimalInfeasible:
                print("  Primal infeasibility certificate found.")

            elif prosta == fusion.ProblemStatus.Unknown:
                # The solutions status is unknown. The termination code
                # indicates why the optimizer terminated prematurely.
                print("  The NPMLE solution status is unknown.")
                symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
                print("  Termination code: {0} {1}".format(symname, desc))

                print('  This warning message is likely caused by numerical errors.',
                    '\n  For details see "MSK_RES_TRM_STALL" (10006) at \n  https://docs.mosek.com/9.2/rmosek/response-codes.html')
                # Please note that if a linear optimization problem is solved using the interior-point optimizer with
                # basis identification turned on, the returned basic solution likely to have high accuracy,
                # even though the optimizer stalled.

            else:
                print("  Another unexpected problem status {0} is obtained.".format(prosta))

        except Exception as e:
            print("  Unexpected error: {0}".format(e))

        # try:
        #     pi = f.level()
        # except Exception as e:
            # print("XZ: f level doesn't have a solution.")
            # use EM as fallback
            
        #     print(pi)
        
        return pi, A

def genMatLoc(mat, hCenter, vCenter):
    # vertical location is col
    # horizontal location is row
    nrow = mat.shape[0]
    ncol = mat.shape[1]
    hLoc = np.array([0.1]*nrow) * np.arange(nrow) 
    hLoc = hLoc - hLoc.mean() + hCenter
    vLoc = np.array([0.1]*ncol) * np.arange(ncol) 
    vLoc = vLoc - vLoc.mean() + vCenter    
    return 1- hLoc, vLoc

def writeMat(ax, mat, symbol, hCenter = 0.5, vCenter=0.5):
    hL, vL= genMatLoc(mat, hCenter, vCenter)
    # write symbol
    ax.text(min(vL) - 0.05, hCenter, symbol + " =", horizontalalignment = "right")
    # plot the bracket 
    # left bracket
    # xloc = min(vL) - 0.035
    # ax.axvline(x = xloc, ymin = min(hL), ymax = max(hL), c="black", lw = 1)
    # ax.axhline(y = min(hL), xmin = xloc, xmax = xloc + 0.01, c="black", lw = 1)
    # ax.axhline(y = max(hL), xmin = xloc, xmax = xloc + 0.01, c="black", lw = 1)
    # # right bracket 
    # xloc = max(vL) + 0.05
    # ax.axvline(x = xloc, ymin = min(hL), ymax = max(hL), c="black", lw = 1)
    # ax.axhline(y = min(hL), xmin = xloc, xmax = xloc - 0.01, c="black", lw = 1)
    # ax.axhline(y = max(hL), xmin = xloc, xmax = xloc - 0.01, c="black", lw = 1)
    # write the matrix
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(vL[j], hL[i], "%.2f" % mat[i,j], verticalalignment = "center")
