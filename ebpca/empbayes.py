'''
==========
Empirical Bayes Methods
==========
This module supports empirical Bayes estimation for various prior.
Typical usage example:
'''
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import matplotlib as mpl
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

        # self.warm_start = False

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
    def get_margin_pdf(self, mu, cov, dim, x):
        pass 

    @abstractmethod
    def check_prior(self, figname):
        pass

    def fit(self, f, mu, cov, **kwargs):
        self.estimate_prior(f,mu,cov)
        figname = kwargs.get("figname", "")
        if (self.to_show or self.to_save):
            self.check_margin(f,mu,cov,figname)
            # self.check_prior(figname)

    def check_margin(self, fs, mu, cov, figname):
        # calm down, think waht is the dimension of s
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
            axes[dim].set_title("PC %i, mu=%.2f, cov=%.2f" % \
                                (dim + 1, mu[dim, dim], cov[dim, dim]))
        if self.to_show:
            plt.show()
        if self.to_save:
            fig.savefig(self.fig_prefix +figname)
        plt.close()

    def plot_each_margin(self, ax, f, dim, mu, cov):
        # span the grid to be plotted
        xmin = np.quantile(f, 0.05, axis=0)
        xmax = np.quantile(f, 0.95, axis=0)
        xgrid = np.linspace(xmin - abs(xmin) / 3, xmax + abs(xmax) / 3, num=100)
        # evaluate marginal pdf
        pdf = [self.get_margin_pdf(x, mu, cov, dim) for x in xgrid]
        ax.hist(f, bins=40, alpha=0.5, density=True, color="skyblue", label="empirical dist")
        ax.plot(xgrid, pdf, color="grey", linestyle="dashed", label="theoretical density")
        ax.legend()

class NonparEB(_BaseEmpiricalBayes):
    
    def __init__(self, optimizer = "EM", ftol = 1e-6, nsupp_ratio = 1, em_iter = 10, maxiter = 100, to_save = False, to_show = False, fig_prefix = "nonpareb", **kwargs):
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
        # print("Start Estimating the prior")
        if self.optimizer == "EM":
            self.pi = _npmle_em_hd(f, self.Z, mu, covInv, self.em_iter, self.nsample, self.nsupp, self.rank)
        if self.optimizer == "Mosek":
            self.pi = _mosek_npmle(f, self.Z, mu, covInv, self.ftol)

    def get_margin_pdf(self, x, mu, cov, dim):
        loc = self.Z.dot(mu.T)[:, dim]
        # loc = np.array([mu.dot(z) for z in self.Z])[:, dim]
        scalesq = cov[dim, dim]
        return np.sum(self.pi / np.sqrt(2 * np.pi * scalesq) * np.exp(-(x - loc) ** 2 / (2 * scalesq)))

    def denoise(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        P = get_P(f,self.Z, mu, covInv, self.pi)
        return P @ self.Z 

    def ddenoise(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        P = get_P(f, self.Z, mu, covInv, self.pi)
        ZouterMZ = np.einsum("ijk, kl -> ijl" ,matrix_outer(self.Z, self.Z.dot(mu.T)), covInv) 
        E1 = np.einsum("ij, jkl -> ikl", P, ZouterMZ)
        E2a = P @ self.Z # shape (I * rank)
        E2 = np.einsum("ijk, kl -> ijl" ,matrix_outer(E2a, E2a.dot(mu.T)), covInv)  # shape (I * rank)

        return E1 - E2

class NonparEBChecker(NonparEB):
    def __init__(self, truePriorLoc, truePriorWeight, optimizer = "EM", ftol = 1e-6, nsupp_ratio = 1, em_iter = 10, maxiter = 100, to_save = False, to_show = False, fig_prefix = "nonparebck", **kwargs):
        NonparEB.__init__(self, optimizer, ftol, nsupp_ratio, em_iter, maxiter, to_save, to_show, fig_prefix, **kwargs)
        self.trueZ = truePriorLoc
        self.truePi = truePriorWeight 
        self.redirectd = False

    def get_margin_pdf_from_true_prior(self, x, mu, cov, dim):
        loc = self.trueZ.dot(mu.T)[:, dim]
        scalesq = cov[dim, dim]
        return np.sum(self.truePi / np.sqrt(2 * np.pi * scalesq) * np.exp(-(x - loc) ** 2 / (2 * scalesq)))

    def plot_each_margin(self, ax, f, dim, mu, cov):
        xmin = np.quantile(f, 0.05, axis=0)
        xmax = np.quantile(f, 0.95, axis=0)
        xgrid = np.linspace(xmin - abs(xmin) / 3, xmax + abs(xmax) / 3, num=100)
        # evaluate marginal pdf
        pdf = [self.get_margin_pdf(x, mu, cov, dim) for x in xgrid]
        # print("why is this not correct")
        # print(self.trueZ.dot(mu.T)[:, dim])
        # print(self.truePi)
        truePdf = [self.get_margin_pdf_from_true_prior(x, mu, cov, dim) for x in xgrid]
        ax.hist(f, bins=40, alpha=0.5, density=True, color="skyblue", label="empirical dist")
        ax.plot(xgrid, pdf, color="grey", linestyle="dashed", label="theoretical density")
        ax.plot(xgrid, truePdf, color="red", linestyle="dashed", label="reference density")
        ax.legend()
            
class NonparBayes(NonparEB):
    def __init__(self, truePriorLoc, truePriorWeight, optimizer = "EM", ftol = 1e-6, nsupp_ratio = 1, em_iter = 10, maxiter = 100, to_save = False, to_show = False, fig_prefix = "nonparebck", **kwargs):
        NonparEB.__init__(self, optimizer, ftol, nsupp_ratio, em_iter, maxiter, to_save, to_show, fig_prefix, **kwargs)
        self.Z = truePriorLoc
        self.pi = truePriorWeight
        self.rank = truePriorLoc.shape[1]
    
    def estimate_prior(self, f,mu,cov):
        pass

class PointNormalEB(_BaseEmpiricalBayes):

    def __init__(self, to_save = True, to_show = False, fig_prefix = "pointnormaleb"):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)
        self.pi = 0.5
        self.mu_x = 0
        self.sigma_x = 1
        self.tol = 1e-6

    def estimate_prior(self, f, mu, sigma):
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

@jit(nopython=True)
def _npmle_em_hd(f, Z, mu, covInv, em_iter, nsample, nsupp, ndim):
    # pi = np.full((nsupp,), 1/nsupp, dtype= float)
    pi = np.array([1/nsupp] * nsupp)
    
    W = get_my_W(f, Z, mu, covInv)

    for _ in range(em_iter):
        denom = my_dot(W, pi) # denom[i] = \sum_j pi[j]*W[i,j]

        for j in range(nsupp):
            pi[j] = pi[j]*np.mean(W[:,j]/denom)

    return pi

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

def get_W(f, z, mu, covInv):
    '''
    Compute conditional likelihood
    '''
    if len(f.shape) > 1:
        W = get_W_multivar(f, z, mu, covInv)
    else:
        W = get_W_univar(f, z, mu, covInv)
    return W

@jit(nopython = True)
def get_my_W(f, z, mu, covInv):
    nsample = f.shape[0]
    nsupp = z.shape[0]
    W = np.empty(shape = (nsample, nsupp),)
    for i in range(nsample):
        for j in range(nsupp):
            vec = f[i] - my_dot(mu, z[j])
            res = np.exp(-np.sum(my_dot(covInv, vec) * vec)/2)
            W[i,j] = res
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

matrix_outer = lambda A, B: np.einsum("bi,bo->bio", A, B)

def span_grid(xx, yy):
    xv, yv = np.meshgrid(xx, yy, sparse = False)
    return np.dstack([xv, yv]).reshape((len(xx)*len(yy),2))

def _mosek_npmle(f, Z, mu, covInv, tol):
    A = get_W(f, Z, mu, covInv)
    n, m = A.shape

    # objective function: the primal in Section 4.2,
    # https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.869224
    M = fusion.Model('NPMLE')
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

    pi = np.repeat(0, m)

    M.objective(fusion.ObjectiveSense.Maximize, fusion.Expr.dot(ones, logg))

    # response handling for Mosek solutions
    # modified from https://docs.mosek.com/9.2/pythonfusion/errors-exceptions.html
    try:
        M.solve()
        # Set solution status to 'Feasible' to accept sub-optimal solutions
        # to circumvent numerical errors
        M.acceptedSolutionStatus(fusion.AccSolutionStatus.Feasible) #Anything
        if M.getProblemStatus() == fusion.ProblemStatus.Unknown:
            # print(M.getDualSolutionStatus())
            print("The MOSEK solution status is unknown.")
            symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
            print("   Termination code: {0} {1}".format(symname, desc))
            print('   This warning message is likely caused by numerical errors. \n',
                  '   For details see "MSK_RES_TRM_STALL" (10006) at \n    https://docs.mosek.com/9.2/rmosek/response-codes.html \n',
                  '   EB-PCA proceeds with sub-optimal but feasible solution. ')
            # Please note that if a linear optimization problem is solved using the interior-point optimizer with
            # basis identification turned on, the returned basic solution likely to have high accuracy,
            # even though the optimizer stalled.
        pi = f.level()

    except fusion.OptimizeError as e:
        print("Optimization failed. Error: {0}".format(e))

    except fusion.SolutionError as e:
        # The solution with at least the expected status was not available.
        # We try to diagnoze why.
        print("Requested solution was not available.")
        prosta = M.getProblemStatus()

        if prosta == fusion.ProblemStatus.DualInfeasible:
            print("Dual infeasibility certificate found.")

        elif prosta == fusion.ProblemStatus.PrimalInfeasible:
            print("Primal infeasibility certificate found.")

        elif prosta == fusion.ProblemStatus.Unknown:
            # The solutions status is unknown. The termination code
            # indicates why the optimizer terminated prematurely.
            print("The solution status is unknown.")
            symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
            print("   Termination code: {0} {1}".format(symname, desc))

            pi = f.level()

        else:
            print("Another unexpected problem status {0} is obtained.".format(prosta))

    except Exception as e:
        print("Unexpected error: {0}".format(e))

    # print('Minimal pi value: {:.2f}'.format(np.min(pi)))
    # print('Sum of estimated pi: {:.2f}'.format(np.sum(pi)))
    # address negative values due to numerical instability
    pi[pi < 0] = 0
    # normalize the negative values due to numerical issues
    pi = pi / np.sum(pi)

    return pi

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