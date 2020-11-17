# A collection of helper functions for rank one and rank two simulations
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from tutorial import get_alignment, redirect_pc

def approx_prior(Ustar, pca_U):
    Ustar = redirect_pc(Ustar[:, np.newaxis], pca_U)
    truePriorLoc = Ustar
    truePriorWeight = np.full((len(truePriorLoc),), 1 / len(truePriorLoc))
    return truePriorLoc, truePriorWeight

def fill_alignment(U_est, u_star, iters):
    '''
    create an alignment vector with expected length (=iters)
    dealing with early stopping mechanism in amp, caused by numerical error and MOSEK
    '''
    alignment = []
    for i in range(iters):
        if i < U_est.shape[2]:
            alignment.append(get_alignment(U_est[:, :, i], u_star))
        else:
            alignment.append(np.nan)
    return alignment

# --------------------
# rank one simulations
# --------------------

# functions for simulating priors and data under the signal-plus-noise model
def simulate_prior(prior, n=2000, seed=1):
    '''
    simulate 3 distributions with 2nd moment=1
    Uniform: represents continuous prior
    Two_points: represents degenerative prior / cluster structure
    Point_normal: represents sparse prior
    '''
    np.random.seed(seed)
    if prior == 'Uniform':
        theta = np.random.uniform(-2, 1, size = n)
    if prior == 'Two_points':
        theta = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
    if prior == 'Point_normal':
        point_obs = np.repeat(0, n)
        assignment = np.random.binomial(n=1, p=0.5, size=n)
        normal_obs = np.random.normal(loc=0, scale=np.sqrt(2), size=n)
        theta = point_obs * (1 - assignment) + normal_obs * assignment
    return theta

def simulate_rank1_model(u, v, s):
    n = u.shape[0]
    d = v.shape[0]
    W = np.random.normal(0, np.sqrt(1/n), n*d).reshape((n, d))
    A = s * 1 / n * np.outer(u, v) + W
    return A
