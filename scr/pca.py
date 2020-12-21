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
    1. We standardize along the sample axis 
'''
import os
from collections import namedtuple

import numpy as np
import scipy.stats
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

PcaPack = namedtuple("PCAPack", ["X", "U", "V", "mu", "K", \
    "n_samples", "n_features", "signals", "sample_aligns", "feature_aligns"])

## --- get pca pack and check residual spectra --- ##

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
        # print("singval should be at least {:.4f} to satisfy the super critical condition.".format(singval_threshold))
        if min(Lambda) < singval_threshold:
            raise(ValueError("Signal doesn't seperate from the bulk."))
        greek_lambda = Lambda / np.sqrt(aspect_ratio)
        s = np.sqrt((greek_lambda**2 * aspect_ratio - 1 - aspect_ratio + \
            np.sqrt((greek_lambda**2*aspect_ratio - 1 - aspect_ratio)**2 - 4*aspect_ratio) \
                ) / (2*aspect_ratio))
        print("Estimation of s is {}.".format(s))
    
    else:
        s = np.array(s).reshape(-1)
    
    sample_align = np.sqrt(1- (1 + s**2)/(s**2*(aspect_ratio*s**2 + 1)))
    feature_align = np.sqrt(1- (1 + aspect_ratio*s**2) /(aspect_ratio*s**2*(s**2 + 1)))
    pca_pack = PcaPack(X = X, U = U, V = Vh.transpose(), mu = Lambdas[K:], \
        n_samples = n_samples, n_features = n_features, \
        K = K, signals = s, sample_aligns= sample_align, \
            feature_aligns= feature_align)
    return pca_pack

def check_residual_spectrum(pca_pack, to_show = False, to_save = False):
    '''we require the noise variance to be 1/n_samples
    mu must be sorted in descending order
    '''
    mu = pca_pack.mu
    n_samples = pca_pack.n_samples
    n_features = pca_pack.n_features

    shorter_side = min(n_samples, n_features)
    
    fig, ax = plt.subplots()
    ax.hist(mu[:shorter_side], density = True, bins = 50, label = "Sample singular values")
    x = np.linspace(mu.min()-0.1, mu.max(), num = 50)
    aspect_ratio = n_features / n_samples
    if aspect_ratio > 1:
        scaler = aspect_ratio
    else:
        scaler = 1
    ax.plot(x, scaler*np.array(sqrtMPlaw(x, n_samples, n_features)), label = "MP law prediction of spectral distribution")
    ax.legend()
    ax.set_title("Singular values")
    if to_save:
        fig.savefig("./figures/residual_check.pdf")
    if to_show:
        plt.show()

def sqrtmplaw(x, n_samples = 0, n_features = 0):
    '''we require the noise variance to be 1/n_samples
    '''
    aspect_ratio = n_features/n_samples
    lambda_plus = (1+np.sqrt(aspect_ratio))**2
    lambda_minus = (1 - np.sqrt(aspect_ratio))**2
    if x**2 < lambda_minus or x**2 > lambda_plus:
        return 0
    else: 
        return 1/(np.pi * aspect_ratio * x) * np.sqrt((lambda_plus - x**2)*(x**2 - lambda_minus))

def sqrtMPlaw(arr, n, p):
    return [sqrtmplaw(x, n, p) for x in arr]
