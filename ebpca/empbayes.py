'''
==========
Empirical Bayes Methods
==========
This module supports empirical Bayes estimation for various prior.

Typical usage example:
'''

from abc import ABC, abstractmethod

import numpy as np 
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
    def fit(self, f, mu, sigma):
        pass 

    @abstractmethod
    def denoise(self, f, mu, sigma):
        pass 

    @abstractmethod
    def ddenoise(self, f, mu, sigma):
        pass 
    
    @abstractmethod
    def get_marginal_dist(self, mu, sigma):
        pass 
    
    def check_margin(self, f, mu, sigma, figname):

            xgrid, pdf = self.get_marginal_dist(mu, sigma)
            
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


class NonparEB(_BaseEmpiricalBayes):
    '''setting prior support points and estimating prior weights using NPMLE.
    '''

    def __init__(self, em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nonpareb",  **kwargs):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)
        self.x_supp = None
        self.pi = None 
        self.n_supp = 0
        self.em_iter = em_iter
        

    def fit(self, f, mu, sigma, **kwargs):
        # from section 6.2 in https://projecteuclid.org/euclid.aos/1245332828
        # upper bound this number, otherwise it takes forever to run
        self.n_supp = min(int(np.sqrt(len(f/mu)) * (np.max(f/mu) - np.min(f/mu))**2/4), 1000)
        self.x_supp = np.linspace(np.min(f/mu), np.max(f/mu), self.n_supp)
        self.x_supp = np.concatenate([self.x_supp, [0]])
        self.n_supp += 1
        self.pi = np.repeat(1/self.n_supp, self.n_supp)
        self.pi = npmle_em(y_obs = f, mu = mu, sigma = sigma, x_grid = self.x_supp, n_np = self.n_supp, dim =  len(f), n_iter = self.em_iter, pi_t = self.pi)
        figname = kwargs.get("figname", None)
        self.check_margin(f, mu, sigma, figname = figname)

    def denoise(self, f, mu, sigma):
        if ((self.x_supp is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        return vf_nonpar(f, mu, sigma, (self.pi, self.x_supp))

    def ddenoise(self, f, mu, sigma):
        if ((self.x_supp is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        return vdf_nonpar(f, mu, sigma, (self.pi, self.x_supp))   

    def get_marginal_dist(self, mu, sigma):
        if ((self.x_supp is None) or (self.pi is None)):
            raise ValueError("denoise before fit is done.")
        
        bound = np.max(self.x_supp.max()) + 3 * np.sqrt(sigma**2)
        xgrid = np.arange(-bound, bound, 0.01)
        pdf_func = np.vectorize(lambda x: np.sum(self.pi /np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-self.x_supp*mu)**2/(2*sigma**2))))
        pdf = pdf_func(xgrid)

        return xgrid, pdf
        
    

class TestEB(_BaseEmpiricalBayes):

    def __init__(self, to_save = True, to_show = False, fig_prefix = "testeb"):
        _BaseEmpiricalBayes.__init__(self, to_save, to_show, fig_prefix)

    def fit(self, f, mu,sigma, **kwargs):
        figname = kwargs.get("figname", None)
        self.plot_marginal_dist(f, mu, sigma, figname = figname)
    
    def denoise(self, f, mu, sigma, prior_par = None):
        return np.tanh(f)

    def ddenoise(self, f,mu, sigma, prior_par = None):
        return 1 - np.tanh(f)**2

    def plot_marginal_dist(self, f, mu, sigma, figname):
        sigmasq = sigma**2
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,5))
        ax.hist(f, bins = 40, density = True)
        xgrid = np.linspace(min(f)*1.1,max(f)*1.1,1000)
        ygrid = two_point_normal_pdf(xgrid, mu, sigmasq)
        ax.plot(xgrid, ygrid, color = 'orange')
        ax.vlines(x = mu, ymin = 0, ymax = max(ygrid), color = 'orange')
        ax.vlines(x = -mu, ymin = 0, ymax = max(ygrid), color = 'orange')
        ax.set_title("{} mu {} sigma {}".format(figname, mu, sigma))
        fig.savefig(self.fig_prefix + figname)

def two_point_normal_pdf(x,mu,sigmasq):
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
        E2 = np.inner(x**2*pi, phi) / np.inner(pi, phi)
        return (E2 - E1**2)
    
    return np.array([df(yi, x, pi, mu, sigma) for yi in y])