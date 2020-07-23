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

Typical usage example:

'''
import os

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn import preprocessing 

## --- standarize data --- ##

def transform(X, data_label = None):
    '''
    Input:
    -----
    X: ndrray of shape (n_samples, n_features)

    Output:
    -----
    data: ndrray of shape (n_samples, n_features)
        imputed and standardized X
    mu: ndrray of shape (n_samples, ) or (n_features, )
        singular values of data.
    n_samples, n_features: int
    '''
    
    # impute data
    col_mean = np.nanmean(X, axis = 0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_mean, idx[1])

    if np.isnan(X).any():
        raise ValueError("data contain a column with no valid value")

    # standarize data along feature space # this seems not correct.
    X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

    # engenvalues 
    _, s, _ = np.linalg.svd(X,full_matrices=False)
    
    # plot pc 
    plot_pc(X, data_label)

    return {"data": X, "mu": s, "n_samples": X.shape[0], "n_features": X.shape[1]}

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
    A dictionary of {alpha, align_sample, align_feature}
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
    align_sample = np.sqrt(abs(2 * phi/dCauchy))/theta 

    # alignment in the feature direction
    phi = np.mean(singval/(singval**2 - mu[:n_features]**2))
    align_feature = np.sqrt(abs(2 * phi/dCauchy))/theta 

    alpha = theta / np.sqrt(n_samples/n_features)

    return {"alpha": alpha, "align_sample":  align_sample, "align_feature": align_feature}


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




    



