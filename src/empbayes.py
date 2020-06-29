'''
==========
Empirical Bayes Methods
==========
This module supports NPMLE
'''

_WITH_MOSEK = True

from abc import ABC, abstractmethod

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

try:
    import mosek
    import mosek.fusion as fusion
except ImportError:
    _WITH_MOSEK = False 

if not _WITH_MOSEK:
    print("Mosek is not supported in the current environment.")
    

class _BaseEmpiricalBayes(ABC):
    '''
    Methods:
    ------
    fit: estimate the prior distribution
    denoise: denoise the posterior observations
    ddenoise: get the derivative of the denoising fuction at the posterior observations

    Attributes:
    -----
    to_save: boolean
        Save the marginal fit plot
    to_show: boolean
        Show the marginal fit plot
    '''
    
    def __init__(self, to_save = False, to_show = False, fig_prefix = ""):
        self.to_save = to_save
        self.to_show = to_show
        self.fig_prefix = "figures/"+fig_prefix
        self.rank = 0

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

    def fit(self, f, mu, cov, figname = ""):
        self.estimate_prior(f,mu,cov)
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
    '''NPMLE based empirical Bayes

    Methods
    -------
    See base class.
    
    Attributes
    -------
    optimizer: string
        "EM" or "Mosek".
    nsupp_ratio: float in [0,1]
        Ratio of number of support points over number of observations. Reduce this
        value will increase the speed of computation, but yields less accurate NPMLE.
    max_nsupp: int or None
        Max number of support points. If None, there is no such limit.
    ftol: float
        ftol for Mosek solver.    
    em_iter: int
        Number of iterations for EM to solve NPMLE.
    '''

    def __init__(self, optimizer = "EM", max_nsupp = 2000 , \
        ftol = 1e-8, nsupp_ratio = 1, em_iter = 500, \
        to_save = False, to_show = False, fig_prefix = "nonpareb"):
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
        self.nsupp_ratio = nsupp_ratio
        self.max_nsupp = max_nsupp
        self.pi = None
        self.Z = None

    def _check_init(self, f, mu, cov):
        self.rank = len(mu)
        self.nsample = len(f)
        self.nsupp = int(self.nsupp_ratio * self.nsample)
        if self.max_nsupp is not None:
            self.nsupp = max(self.nsupp, int(self.max_nsupp))
        self.pi = np.full((self.nsupp,),1/self.nsupp)
        if self.nsupp_ratio >= 1:
            self.Z = f.dot(np.linalg.pinv(mu).T)
        else:
            self.Z = f[np.random.choice(f.shape[0], self.nsupp, replace=False), :].dot(np.linalg.pinv(mu).T)

    def check_prior(self, figname):
        # can only be used when two dimension
        if self.rank == 2:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7, 5))
            ax.scatter(self.Z[:,0], self.Z[:,1], s = self.pi*len(self.pi), marker = ".")
            ax.set_title("check prior, {}".format(figname))
            if self.to_show:
                plt.show()
            if self.to_save:
                fig.savefig(self.fig_prefix + "prior" +figname)
            plt.close()         
    
    def estimate_prior(self,f, mu, cov):
        # check initialization  
        self._check_init(f,mu,cov)
        covInv = np.linalg.inv(cov)
        if self.optimizer == "EM":
            self.pi = _npmle_em_hd(f, self.Z, mu, covInv, self.em_iter)
        if self.optimizer == "Mosek":
            if _WITH_MOSEK:
                try:
                    self.pi = _mosek_npmle(f, self.Z, mu, covInv, self.ftol)
                except Exception as err:
                    print("Error occured with Mosek. Use EM instead. Error Message\n{}".format(err))
                    self.pi = _npmle_em_hd(f, self.Z, mu, covInv, self.em_iter)
            else:
                print("Mosek in not supported. Use EM instead.")
                self.pi = _npmle_em_hd(f, self.Z, mu, covInv, self.em_iter)

    def get_margin_pdf(self, x, mu, cov, dim):
        loc = self.Z.dot(mu.T)[:, dim]
        scalesq = cov[dim, dim]
        return np.sum(self.pi / np.sqrt(2 * np.pi * scalesq) * np.exp(-(x - loc) ** 2 / (2 * scalesq)))
    
    def denoise(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        P = _get_P(f,self.Z, mu, covInv, self.pi)
        return P @ self.Z 

    def ddenoise(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        P = _get_P(f, self.Z, mu, covInv, self.pi)
        ZouterMZ = np.einsum("ijk, kl -> ijl" ,matrix_outer(self.Z, self.Z.dot(mu.T)), covInv) 
        E1 = np.einsum("ij, jkl -> ikl", P, ZouterMZ)
        E2a = P @ self.Z # shape (I * rank)
        E2 = np.einsum("ijk, kl -> ijl" ,matrix_outer(E2a, E2a.dot(mu.T)), covInv)  # shape (I * rank)
        return E1 - E2

    def pos2m(self, f, mu, cov):
        covInv = np.linalg.inv(cov)
        P = _get_P(f, self.Z, mu, covInv, self.pi)
        ZouterMZ = matrix_outer(self.Z, self.Z)
        E1 = np.einsum("ij, jkl -> ikl", P, ZouterMZ)
        return E1

class NonparEBChecker(NonparEB):
    def __init__(self, truePriorLoc, truePriorWeight = None,\
        optimizer = "EM", max_nsupp = 2000 , \
        ftol = 1e-8, nsupp_ratio = 1, em_iter = 100, \
        to_save = False, to_show = False, fig_prefix = "nonparebck"):
        NonparEB.__init__(self, optimizer, max_nsupp, ftol, nsupp_ratio, em_iter, to_save, to_show, fig_prefix)
        assert len(truePriorLoc.shape) == 2
        n, k = truePriorLoc.shape
        self.trueZ = truePriorLoc
        self.rank = k
        if truePriorWeight is None:
            self.truePi = np.full((n,),1/n)
        else:
            assert len(truePriorWeight.shape) == 1
            self.truePi = truePriorWeight 

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
        truePdf = [self.get_margin_pdf_from_true_prior(x, mu, cov, dim) for x in xgrid]
        ax.hist(f, bins=40, alpha=0.5, density=True, color="skyblue", label="empirical dist")
        ax.plot(xgrid, pdf, color="grey", linestyle="dashed", label="theoretical density")
        ax.plot(xgrid, truePdf, color="red", linestyle="dashed", label="reference density")
        ax.legend()
            
class NonparBayes(NonparEB):
    def __init__(self, truePriorLoc, truePriorWeight = None, to_save = False, to_show = False, fig_prefix = "nonparbayes"):
        NonparEB.__init__(self, to_save = to_save, to_show = to_show, fig_prefix = fig_prefix)
        assert len(truePriorLoc.shape) == 2
        n, k = truePriorLoc.shape
        self.Z = truePriorLoc
        self.rank = k
        if truePriorWeight is None:
            self.pi = np.full((n,),1/n)
        else:
            assert len(truePriorWeight.shape) == 1
            self.pi = truePriorWeight
    
    def estimate_prior(self, f,mu,cov):
        pass


def _npmle_em_hd(f, Z, mu, covInv, em_iter):
    # I dont need this n_dim
    nsupp = Z.shape[0]
    pi = np.array([1/nsupp] * nsupp)
    W = _get_W(f, Z, mu, covInv)

    Wt = np.array(W.T, order = 'C')
    for _ in range(em_iter):
        denom = W.dot(pi)# [:,np.newaxis] # denom[i] = \sum_j pi[j]*W[i,j]
        pi = pi * np.mean(Wt/denom, axis = 1)
    
    return pi

# consider another get W using broadcast
# W[i,j] = f(x_i | z_j)
def _get_W(f, z, mu, covInv):
    fsq = (np.einsum("ik,ik -> i", f @ covInv, f) / 2)[:,np.newaxis]
    mz = z.dot(mu.T)
    zsq = np.einsum("ik, ik->i", mz @ covInv, mz) / 2
    fz = f @ covInv @ mz.T
    del mz
    return np.exp(- fsq + fz - zsq)

# P[i,j] = P(Z_j | X_i)
def _get_P(f,z,mu,covInv,pi):
    W = _get_W(f,z,mu,covInv)
    denom = W.dot(pi) # denom[i] = \sum_j pi[j] * W[i,j]
    num = W * pi # W*pi[i,j] = pi[j] * W[i,j]
    return num / denom[:, np.newaxis]

def _get_P_from_W(W, pi):
    denom = W.dot(pi) # denom[i] = \sum_j pi[j] * W[i,j]
    num = W * pi # W*pi[i,j] = pi[j] * W[i,j]
    return num / denom[:, np.newaxis]    


def _mosek_npmle(f, Z, mu, covInv, tol=1e-8):
    A = _get_W(f, Z, mu, covInv)
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

        M.objective(fusion.ObjectiveSense.Maximize, fusion.Expr.dot(ones, logg))

        # response handling for Mosek solutions
        # modified from https://docs.mosek.com/9.2/pythonfusion/errors-exceptions.html
        try:
            M.solve()
            # https://docs.mosek.com/9.2/pythonfusion/enum_index.html#accsolutionstatus
            M.acceptedSolutionStatus(fusion.AccSolutionStatus.Optimal) # Anything Optimal
            # print(" Accepted solution setting:", M.getAcceptedSolutionStatus())
            pi = f.level()
            # print(pi[:5])
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

        pi = f.level()
        
        return pi

def _gaussian_pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi)) * (1 / sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def _log_gaussian_pdf(x, mu, sigma):
    return - np.log(sigma) - 0.5 * np.log(2 * np.pi) - (x - mu) ** 2 / (2 * sigma ** 2)

matrix_outer = lambda A, B: np.einsum("bi,bo->bio", A, B)

def span_grid(xx, yy):
    xv, yv = np.meshgrid(xx, yy, sparse = False)
    return np.dstack([xv, yv]).reshape((len(xx)*len(yy),2))

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
    # write the matrix
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(vL[j], hL[i], "%.2f" % mat[i,j], verticalalignment = "center")
