'''
==========
Empirical Bayes Methods
==========
This module supports empirical Bayes estimation for various prior.

Typical usage example:
'''

from abc import ABC, abstractmethod

import numpy as np 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from numba import jit


class _BaseEmpiricalBayes(ABC):
    '''Use empirical Bayes to estimate the prior and denoise observations.
    
    Given the prior family of X, mu, sigm and Y s.t. Y ~ mu X + sigma Z, estimate X.

    Attributes
    -----
    to_plot: 
    to_save:
    fig_prefix:


    Methods
    -----
    fit: (data, mu, sigma) -> dist parameters
    denoise(f, prior_par) -> denoised data
    ddenoise(f, prior_par)-> derivated of the denoising functions evaluated at data 
    '''
    def __init__(self, to_save = True, to_show = False, fig_prefix = "figures/"):
        self.to_save = to_save
        self.to_show = to_show
        self.fig_prefix = "figures/"+fig_prefix
        self.warm_start = False

    @abstractmethod
    def estimate_prior(self,f, mu, sigma):
        pass

    @abstractmethod
    def denoise(self, f, mu, sigma):
        pass 

    @abstractmethod
    def ddenoise(self, f, mu, sigma):
        pass 
    
    @abstractmethod
    def get_margin_pdf(self, mu, sigma, x):
        pass 
    
    def fit(self, f, mu, sigma, **kwargs):
        self.estimate_prior(f,mu,sigma)
        figname = kwargs.get("figname", "")
        self.check_margin(f,mu,sigma,figname)

    def check_margin(self, f, mu, sigma, figname):
        xmin = np.quantile(f,0.05); xmax = np.quantile(f,0.95)
        xgrid = np.linspace(xmin - abs(xmin)/3, xmax + abs(xmax)/3)
        
        pdf = [self.get_margin_pdf(mu, sigma, x) for x in xgrid]

        fig, ax = plt.subplots()
        ax.hist(f, bins = 40, alpha = 0.5, density = True, color = "skyblue", label = "empirical dist")
        ax.plot(xgrid, pdf, color = "grey", linestyle = "dashed", label = "theoretical density")
        ax.legend()
        ax.set_title("marginal for non-parametric prior \n mu={mu}, sigma={sigma}".format(mu=mu,sigma = sigma))
        if self.to_show:
            plt.show()
        if self.to_save:
            fig.savefig(self.fig_prefix + figname)
        plt.close()

class _BaseEmpiricalBayesHD(ABC):
    '''Use empirical Bayes to estimate the prior and denoise observations.
    
    Given the prior family of X, mu, sigm and Y s.t. Y ~ mu X + sigma Z, estimate X.

    Attributes
    -----
    to_plot: 
    to_save:
    fig_prefix:


    Methods
    -----
    fit: (data, mu, sigma) -> dist parameters
    denoise(f, prior_par) -> denoised data
    ddenoise(f, prior_par)-> derivated of the denoising functions evaluated at data 
    '''
    def __init__(self, to_save = True, to_show = False, fig_prefix = "figures/"):
        self.to_save = to_save
        self.to_show = to_show
        self.fig_prefix = "figures/"+fig_prefix
        self.warm_start = False
        self.dim = 1

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
    
    def fit(self, f, mu, cov, **kwargs):
        self.estimate_prior(f,mu,cov)
        figname = kwargs.get("figname", "")
        self.check_margin(f,mu,cov,figname)

    def check_margin(self, fs, mu, cov, figname):

        fig, axes = plt.subplots( nrows = self.dim, ncols = 1, figsize = (self.dim * 7, 5))

        for dim in range(self.dim):
            f = fs[:,dim]    
            xmin = np.quantile(f,0.05, axis = 0); xmax = np.quantile(f,0.95, axis = 0)
            xgrid = np.linspace(xmin - abs(xmin)/3, xmax + abs(xmax)/3, num = 100)
        
            pdf = [self.get_margin_pdf(mu, cov, dim,x) for x in xgrid]

            ax = axes[dim]
            ax.hist(f, bins = 40, alpha = 0.5, density = True, color = "skyblue", label = "empirical dist")
            ax.plot(xgrid, pdf, color = "grey", linestyle = "dashed", label = "theoretical density")
            ax.legend()
            ax.set_title("marginal for non-parametric prior dim{dim} \n mu={mu}, cov={cov}".format(dim = dim, mu=mu,cov = cov))
        
        if self.to_show:
            plt.show()
        if self.to_save:
            fig.savefig(self.fig_prefix + figname)
        plt.close()


class NonparEB(_BaseEmpiricalBayes):
    '''setting prior support points and estimating prior weights using NPMLE.
    '''

    def __init__(self, em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nonpareb",  **kwargs):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)
        self.x_supp = None
        self.pi = None 
        self.n_supp = 0
        self.em_iter = em_iter
        
    def estimate_prior(self, f, mu, sigma):
        self.n_supp = min(int(np.sqrt(len(f/mu)) * (np.max(f/mu) - np.min(f/mu))**2/4), 1000)
        self.x_supp = np.linspace(np.min(f/mu), np.max(f/mu), self.n_supp)
        self.x_supp = np.concatenate([self.x_supp, [0]])
        self.n_supp += 1
        self.pi = np.repeat(1/self.n_supp, self.n_supp)
        self.pi = npmle_em(y_obs = f, mu = mu, sigma = sigma, x_grid = self.x_supp, n_np = self.n_supp, dim =  len(f), n_iter = self.em_iter, pi_t = self.pi)
               
    def denoise(self, f, mu, sigma):
        if ((self.x_supp is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        return vf_nonpar(f, mu, sigma, (self.pi, self.x_supp))

    def ddenoise(self, f, mu, sigma):
        if ((self.x_supp is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        return vdf_nonpar(f, mu, sigma, (self.pi, self.x_supp))   
    
    def get_margin_pdf(self, mu, sigma, x):
        if ((self.x_supp is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        
        return np.sum(self.pi /np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-self.x_supp*mu)**2/(2*sigma**2)))


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

class NonparEBHD(_BaseEmpiricalBayesHD):

    def __init__(self, em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nonparebhd",  **kwargs):
        _BaseEmpiricalBayesHD.__init__(self, to_save, to_show, fig_prefix)
        self.nsample = None
        self.nsupp = None
        self.em_iter = em_iter
        self.pi = None
        self.init = False 
        self.Z = None

    def _check_init(self,f, mu, cov):
        # check initialization
        self.dim = len(mu)
        self.nsample = len(f)
        self.nsupp = len(f)
        self.pi = np.full((self.nsupp,),1/self.nsupp)
        
        self.Z = f.dot(np.linalg.pinv(mu).T)
        
        self.init = True
    
    def estimate_prior(self,f, mu, cov):
        # check initialization
        
        self._check_init(f,mu,cov)
        covInv = np.linalg.inv(cov)
        self.pi = _npmle_em_hd(f, self.Z, mu, covInv, self.em_iter, self.nsample, self.nsupp, self.dim)

    # @jit(nopython = True)
    def _npmle_em(self, f, mu ,cov):
        
        def _get_phi(f, z, mu, covInv):
            return np.exp(-(covInv.dot(f - mu.dot(z)).dot(f - mu.dot(z)))/2)
        
        covInv = np.linalg.inv(cov)
        for _ in range(self.em_iter):
            # W_ij = f(xi|zj)
            W = np.empty(shape = (self.nsample, self.nsupp), order = 'F')
            for i in range(self.nsample):
                for j in range(self.nsupp):
                    W[i,j] = _get_phi(f[i], self.Z[j], mu,covInv)
            
            denom = W.dot(self.pi)
            self.pi = self.pi * (W / denom[:, np.newaxis]).mean(axis = 0)

    def denoise(self, f, mu, cov):
        # check initialization
        self._check_init(f,mu,cov)

        def _get_phi(f, z, mu, covInv):
            return np.exp(-(covInv.dot(f - mu.dot(z)).dot(f - mu.dot(z)))/2)
        
        covInv = np.linalg.inv(cov)
        res = np.empty(shape = (self.nsample, self.dim))
        for i in range(self.nsample):
            phi = np.array([_get_phi(f[i], self.Z[j], mu, covInv) for j in range(self.nsupp)])
            num = np.array([self.pi[j] * phi[j]*self.Z[j] for j in range(self.nsupp)]).sum(axis = 0)
            denum = np.inner(self.pi , phi)
            # print("num {}, denum {}".format(num, denum))
            res[i] = num/denum
        return res
       

    def ddenoise(self, f, mu, cov):

        self._check_init(f,mu,cov)
        
        covInv = np.linalg.inv(cov)
        def _get_phi(f, z, mu, covInv):
            return np.exp(-(covInv.dot(f - mu.dot(z)).dot(f - mu.dot(z)))/2)
        
        res = np.empty(shape = (self.nsample, self.dim, self.dim))

        for i in range(self.nsample):   
            phi = np.array([_get_phi(f[i], self.Z[j], mu, covInv) for j in range(self.nsupp)])
            E1 = [np.outer(self.Z[j], mu.dot(self.Z[j])) * self.pi[j] * phi[j] for j in range(self.nsupp)]
            E1 = np.array(E1).sum(axis = 0).dot(covInv)
            E2 = np.array([self.Z[j] * self.pi[j] * phi[j] for j in range(self.nsupp)]).sum(axis = 0)
            E2 = np.outer(E2, mu.dot(E2)).dot(covInv)
            denom = np.inner(self.pi , phi)
            res[i,:] = E1 / denom - E2/ denom**2
        
        return res
        

    def get_margin_pdf(self, mu, cov, dim, x):
        locs =  np.array([mu.dot(z) for z in self.Z])[:,dim] # mu.dot(z), the dimth element
        scalesq = cov[dim,dim]
        return np.sum(self.pi /np.sqrt(2*np.pi*scalesq) * np.exp(-(x - locs)**2/(2*scalesq)) )

        
        


        

 

class TestEB(_BaseEmpiricalBayes):

    def __init__(self, to_save = True, to_show = False, fig_prefix = "testeb"):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)

    def estimate_prior(self,f,mu,sigma):
        pass
    
    def denoise(self, f, mu, sigma, prior_par = None):
        return np.tanh(f)

    def ddenoise(self, f,mu, sigma, prior_par = None):
        return 1 - np.tanh(f)**2

    def get_margin_pdf(self, mu, sigma, x):
        sigmasq = sigma**2
        pdf_plus = 1/np.sqrt(2*np.pi*sigmasq) * np.exp(-(x-mu)**2/(2*sigmasq))
        pdf_minus = 1/np.sqrt(2*np.pi*sigmasq) * np.exp(-(x+mu)**2/(2*sigmasq))
        return 0.5*pdf_plus+0.5*pdf_minus
        

@jit(nopython=True)
def npmle_em(y_obs, mu, sigma, x_grid, n_np, dim, n_iter, pi_t):
    
    """
    Input:
        y_obs: denoised estimator for U/V
        mu,sigma: parameters in y_obs|U \sim N(mu U, sigma**2) (similar for V)
        n_np: the number of support points
        x_grid: a set of data-driven support points
        n_iter: the number of EM iterations
    
    Solve nonparametric MLE with EM algorithm
    Maximize the marginal likelihood to estimate pi via EM algorithm
    
    Output:
        pi_t: estimated probabilities on nonparametric support points
    """
    
    # initialize with uniform distribution
    phi_matrix = np.zeros((dim, n_np))
    for j in range(dim):
        phi_matrix[j,:] = np.exp(-(y_obs[j] - mu * x_grid)**2 / (2*sigma**2))
    
    for _ in range(n_iter):
        
        weights = np.zeros((dim, n_np))
        
        for j in range(len(y_obs)):
            tmp = pi_t * phi_matrix[j,:]
            weights[j,:] = tmp / np.sum(tmp)
            
        pi_t = np.sum(weights, axis=0) / dim
        
    return pi_t

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

@jit(nopython = True)
def my_dot(mat, vec, ndim):
    res = np.array([0.0]*ndim)
    for i in range(ndim):
        for j in range(ndim):
            res[i] += mat[i,j] * vec[j]
    return res

@jit(nopython=True)
def my_inner(veca, vecb, ndim):
    res = 0
    for i in range(ndim):
        res += veca[i] * vecb[i]
    return res

@jit(nopython=True)
def _npmle_em_hd(f, Z, mu, covInv, em_iter, nsample, nsupp, ndim):
    # pi = np.full((nsupp,), 1/nsupp, dtype= float)
    pi = np.array([1/nsupp] * nsupp)
    
    for _ in range(em_iter):
        # W_ij = f(xi|zj)
        W = np.empty(shape = (nsample, nsupp),)
        for i in range(nsample):
            for j in range(nsupp):
                vec = f[i] - my_dot(mu, Z[j], ndim)
                res = np.exp(-np.sum(my_dot(covInv, vec, ndim) * vec)/2)
                W[i,j] = res
        
        denom = my_dot(W, pi, nsupp) # denom[i] = \sum_j pi[j]*W[i,j]
        # print("denom shape {}".format(denom.shape))

        for j in range(nsupp):
            pi[j] = pi[j]*np.mean(W[:,j]/denom)
        
        # pi = np.array([pi[j]*np.mean(W[:,j]/denom) for j in range(nsupp)])
        # pi = np.mean(pi * (W / denom[:, np.newaxis]), axis = 0) # use normal representation
    return pi