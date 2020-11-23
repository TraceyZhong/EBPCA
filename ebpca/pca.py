'''
==========
PCA analysis on the original data
==========
This module consists of:
    1. Standardize and impute the original data
    2. study singular values and singular vectors
    3. estimate low rank signals and the norm of the leading singular vectors projected to the signal space

Input:
    X: ndarray of shape(n_samples, n_features)

Remarks:
    1. We standardize along the feature axis 

Reference:
    http://www.cmapx.polytechnique.fr/~benaych/YJMVA3366.pdf

'''
import os
from collections import namedtuple

import numpy as np
import scipy.stats
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

PcaPack = namedtuple("PCAPack", ["X", "U", "V", "mu", "K", \
    "n_samples", "n_features", "signals", "sample_aligns", "feature_aligns"])



## --- get pca pack and check goe spectra --- ##

def get_pca_u(X, K = 0):
    if K == 0:
        raise(ValueError("# PC can not be zero."))
    n_samples, n_features = X.shape
    U, Lambdas, Vh = np.linalg.svd(X, full_matrices = False)
    U = U[:,:K]
    Lambda = Lambdas[:K]
    Vh = Vh[:K,:]
    sol = signal_solver_gaussian_u(Lambda, None, n_samples, n_features)
    pca_pack = PcaPack(X = X, U = U, V = Vh.transpose(), mu = Lambdas[K:], \
        n_samples = n_samples, n_features = n_features, \
        K = K, signals = sol["signal"], sample_aligns= sol["sample_align"], \
            feature_aligns= sol["feature_align"])
    return pca_pack

def get_pca_v(X, K = 0):
    if K == 0:
        raise(ValueError("# PC can not be zero."))
    n_samples, n_features = X.shape
    U, Lambdas, Vh = np.linalg.svd(X, full_matrices = False)
    U = U[:,:K]
    Lambda = Lambdas[:K]
    Vh = Vh[:K,:]
    sol = signal_solver_gaussian_v(Lambda, None, n_samples, n_features)
    pca_pack = PcaPack(X = X, U = U, V = Vh.transpose(), mu = Lambdas[K:], \
        n_samples = n_samples, n_features = n_features, \
        K = K, signals = sol["signal"], sample_aligns= sol["sample_align"], \
            feature_aligns= sol["feature_align"])
    return pca_pack

def get_pca(X, K=0, s = None):
    if K == 0:
        raise(ValueError("# PC can not be zero."))
    n_samples, n_features = X.shape
    U, Lambdas, Vh = np.linalg.svd(X, full_matrices = False)
    U = U[:,:K]
    Lambda = Lambdas[:K]
    Vh = Vh[:K,:]
    # solve init parameters
    aspect_ratio = n_features/ n_samples
    print("s should be at least {:.4f} to satisfy the super critical condition.".format(1/aspect_ratio**(1/4)))
    
    if s is None:
        singval_threshold = 1 + np.sqrt(aspect_ratio)
        print("singval should be at least {:.4f} to satisfy the super critical condition.".format(singval_threshold))
        if min(Lambda) < singval_threshold:
            raise(ValueError("Signal doesn't seperate from the bulk."))
        greek_lambda = Lambda / np.sqrt(aspect_ratio)
        s = np.sqrt((greek_lambda**2 * aspect_ratio - 1 - aspect_ratio + \
            np.sqrt((greek_lambda**2*aspect_ratio - 1 - aspect_ratio)**2 - 4*aspect_ratio) \
                ) / (2*aspect_ratio))
        print("Estimation of s is {:.4f}.".format(s))
    
    else:
        s = np.array(s).reshape(1)
    
    sample_align = np.sqrt(1- (1 + s**2)/(s**2*(aspect_ratio*s**2 + 1)))
    feature_align = np.sqrt(1- (1 + aspect_ratio*s**2) /(aspect_ratio*s**2*(s**2 + 1)))
    pca_pack = PcaPack(X = X, U = U, V = Vh.transpose(), mu = Lambdas[K:], \
        n_samples = n_samples, n_features = n_features, \
        K = K, signals = s, sample_aligns= sample_align, \
            feature_aligns= feature_align)
    return pca_pack

def get_bayes_pca(X, s, K=0):
    if K == 0:
        raise(ValueError("# PC can not be zero."))
    n_samples, n_features = X.shape
    U, Lambdas, Vh = np.linalg.svd(X, full_matrices = False)
    U = U[:,:K]
    Vh = Vh[:K,:]
    # solve init parameters
    s = np.array(s).reshape(1)
    aspect_ratio = n_features/ n_samples
    sample_align = np.sqrt(1- (1 + s**2)/(s**2*(aspect_ratio*s**2 + 1)))
    feature_align = np.sqrt(1- (1 + aspect_ratio*s**2) /(aspect_ratio*s**2*(s**2 + 1)))
    pca_pack = PcaPack(X = X, U = U, V = Vh.transpose(), mu = Lambdas[K:], \
        n_samples = n_samples, n_features = n_features, \
        K = K, signals = s, sample_aligns= sample_align, \
            feature_aligns= feature_align)
    return pca_pack

def check_residual_spectrum_u(pca_pack, to_show = False, to_save = False):
    '''we require the noise variance to be 1/n_features
    mu must be sorted in descending order
    '''
    mu = pca_pack.mu
    n_samples = pca_pack.n_samples
    n_features = pca_pack.n_features

    shorter_side = min(n_samples, n_features)
    mu = np.pad(mu, (0,n_samples - len(mu)))[:n_samples]
    
    fig, ax = plt.subplots()
    ax.hist(mu[:shorter_side], density = True, bins = 50, label = "sample singular values")
    x = np.linspace(0.01, mu.max(), num = 50)
    if n_samples > n_features:
        scaler = n_samples / n_features
    else:
        scaler = 1
    ax.plot(x, scaler*np.array(sqrtMPlaw(x, n_samples, n_features)), label = "MP law prediction of spectral distribution")
    ax.legend()
    ax.set_title("Residual Spectrum")
    if to_save:
        fig.savefig("./figures/residual_check.pdf")
    if to_show:
        plt.show()

## --- solve signal --- ##

def _cauchy_transformation(z, mu, n_samples, n_features):
    prod = np.mean(z / (z**2 - mu[:n_samples]**2))
    prod = np.mean(z/(z**2 - mu[:n_features]**2))*prod
    return prod

def _d_cauchy_transformation(z, mu, n_samples, n_features):
    a = np.mean(z / (z**2 - mu[:n_samples]**2))
    b = np.mean(z/(z**2 - mu[:n_features]**2))
    da = -np.mean((z**2 + mu[:n_samples]**2)/ (z**2 - mu[:n_samples]**2)**2)
    db = -np.mean( (z**2 + mu[:n_features]**2)/ (z**2 - mu[:n_features]**2)**2)
    return da*b + db*a 

def _phase_transition_threshold(mu, n_samples, n_features, tol = 0.01):
    supp_max_plus = mu.max() + tol 
    mu = np.pad(mu, (0,(max(n_features, n_samples) - len(mu))), constant_values = (0,0))
    ct = _cauchy_transformation(supp_max_plus, mu, n_samples, n_features)
    return 1/np.sqrt(ct)

def signal_solver_gaussian_original(singval, mu = None, n_samples = 0, n_features = 0, **kwargs):
    '''we require the noise variance to be 1/n_features
    '''
    aspect_ratio = n_samples/n_features
    singval_limit = lambda d, y, s: (d*d + 1) * (d*d + y)/ (d*d) - s*s
    theta = fsolve(singval_limit, x0 = singval, args = (aspect_ratio, singval))

    sample_align = np.sqrt((theta**4 - aspect_ratio)/ (theta**2 + aspect_ratio)) / theta
    feature_align = np.sqrt((theta**4 - aspect_ratio)/ (theta**2 + 1)) / theta

    alpha = theta / np.sqrt(aspect_ratio)

    return {"alpha": alpha, "sample_align":  sample_align, "feature_align": feature_align}

def signal_solver_gaussian_u(singval, mu = None, n_samples = 0, n_features = 0, **kwargs):
    '''we require the noise variance to be 1/n_features
    '''
    aspect_ratio = n_samples/n_features
    print("s should be at least {} to satisfy the super critical condition.".format(1/aspect_ratio**(1/4)))
    print("singval should be at least {} to satisfy the super critical condition.".format(1 + np.sqrt(aspect_ratio)))
    greek_lambda = singval / np.sqrt(aspect_ratio)
    s = np.sqrt((greek_lambda**2 * aspect_ratio - 1 - aspect_ratio + \
        np.sqrt((greek_lambda**2*aspect_ratio - 1 - aspect_ratio)**2 - 4*aspect_ratio) \
            ) / (2*aspect_ratio))
    theta = s * np.sqrt(aspect_ratio)
    sample_align = np.sqrt((theta**4 - aspect_ratio)/ (theta**2 + aspect_ratio)) / theta
    feature_align = np.sqrt((theta**4 - aspect_ratio)/ (theta**2 + 1)) / theta

    return {"signal": s, "sample_align":  sample_align, "feature_align": feature_align}

def signal_solver_gaussian_v(singval, mu = None, n_samples = 0, n_features = 0, **kwargs):
    '''we require the noise variance to be 1/n_samples
    '''
    aspect_ratio = n_features/n_samples
    print("s should be at least {} to satisfy the super critical condition.".format(1/aspect_ratio**(1/4)))
    print("singval should be at least {} to satisfy the super critical condition.".format(1 + np.sqrt(aspect_ratio)))
    greek_lambda = singval / np.sqrt(aspect_ratio)
    s = np.sqrt((greek_lambda**2 * aspect_ratio - 1 - aspect_ratio + \
        np.sqrt((greek_lambda**2*aspect_ratio - 1 - aspect_ratio)**2 - 4*aspect_ratio) \
            ) / (2*aspect_ratio))
    print("Estimation of s is {}.".format(s))
    # theta = s * np.sqrt(aspect_ratio)
    # sample_align = np.sqrt((theta**4 - aspect_ratio)/ (theta**2 + aspect_ratio)) / theta
    # feature_align = np.sqrt((theta**4 - aspect_ratio)/ (theta**2 + 1)) / theta
    sample_align = np.sqrt(1- (1 + s**2)/(s**2*(aspect_ratio*s**2 + 1)))
    feature_align = np.sqrt(1- (1 + aspect_ratio*s**2) /(aspect_ratio*s**2*(s**2 + 1)))

    return {"signal": s, "sample_align":  sample_align, "feature_align": feature_align}

def sqrtmplaw(x, n_samples = 0, n_features = 0):
    '''we require the noise variance to be 1/n_features
    '''
    aspect_ratio = n_samples/n_features
    lambda_plus = (1+np.sqrt(aspect_ratio))**2
    lambda_minus = (1 - np.sqrt(aspect_ratio))**2
    if x**2 < lambda_minus or x**2 > lambda_plus:
        return 0
    else: 
        return 1/(np.pi * aspect_ratio * x) * np.sqrt((lambda_plus - x**2)*(x**2 - lambda_minus))

def sqrtMPlaw(arr, n, p):
    return [sqrtmplaw(x, n, p) for x in arr]

def signal_solver(singval, mu, n_samples, n_features, rank = 0, supp_max = None, tol = 0.01):
    '''solve for singal values and estimate alignments
    Sample Usage
    -----
    estimates = ebpca.pca.signal_solver(singval, mu, n_samples, n_features)
    
    Inputs
    -----
    signvar: double
        the outlying singular value
    mu: ndarray of shape (n_samples, ) or (n_features, )
        the distribution of singular values of the noise matrix
    n_samples, n_features: int
    supp_max: the sup of supp of mu

    spike must be tol away form the bulk. <=> singval > supp_max + tol

    Output
    -----
    A dictionary of {alpha, sample_align, feature_align}
    '''
    # remove the leading k singular value
    mu = mu[rank:]

    if supp_max is None:
        supp_max = mu.max()
    if singval <= (supp_max+tol):
        raise ValueError("signal must seperate from singval supp of noise matrix")
    mu = np.pad(mu, (0,(max(n_features, n_samples) - len(mu))), constant_values = (0,0))

    # signal strength, can convert to alpha later
    theta = np.sqrt(1/_cauchy_transformation(singval, mu, n_samples, n_features))

    dCauchy = _d_cauchy_transformation(singval, mu, n_samples, n_features)
    # alignment in the sample direction
    phi = np.mean(singval/(singval**2 - mu[:n_samples]**2))
    sample_align = np.sqrt(abs(2 * phi/dCauchy))/theta 

    # alignment in the feature direction
    phi = np.mean(singval/(singval**2 - mu[:n_features]**2))
    feature_align = np.sqrt(abs(2 * phi/dCauchy))/theta 

    alpha = theta / np.sqrt(n_samples/n_features)

    return {"alpha": alpha, "sample_align":  sample_align, "feature_align": feature_align}


## --- plot PC --- ##

def plot_pc(samples,label,nPCs=10,u_ref=None):
    u,s,vh = np.linalg.svd(samples,full_matrices=False)
    plt.hist(s[1:],bins=50)
    plt.ylim([0,100])
    plt.title('Singular values, leading removed')
    plt.savefig('figures/singvals_%s.png' % label)
    plt.close()
    for i in range(nPCs):
        fig, (ax2,ax3) = plt.subplots(nrows = 1, ncols = 2, figsize=(5,4))
        if u_ref is not None and sum(u[:,i]*u_ref[:,i]) < 0:
            u[:,i] = -u[:,i]
        # ax1.imshow(u[:,i].reshape((64,64)),cmap='gray')
        ax2.hist(u[:,i],bins=50)
        scipy.stats.probplot(u[:,i],plot=ax3)
        ax2.set_title('PC %d' % i)
        ax3.set_title('PC %d' % i)
        fig.savefig('figures/PC_%d_%s.png' % (i,label))
        plt.close()
    return u,s,vh




    



