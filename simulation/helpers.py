# A collection of helper functions for rank one and rank two simulations
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from tutorial import get_alignment, redirect_pc, normalize_pc

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

def get_joint_alignment(mar):
    joint = [np.sqrt(np.mean((np.array(mar[j])**2), axis=0)) for j in range(len(mar))]
    return joint
# --------------------
# rank one simulations
# --------------------

# functions for simulating priors and data under the signal-plus-noise model
def simulate_prior(prior, n=2000, seed=1, rank=1):
    '''
    simulate univariate or bivariate distributions with marginal 2nd moment=1
    Univariate distributions:
      Uniform: represents continuous prior
      Two_points: represents degenerative prior / cluster structure
      Point_normal: represents sparse prior
    Bivariate distributions:
      Uniform_circle: represents continuous bivariate prior
      Three_points: represent bivariate cluster structure
    '''
    np.random.seed(seed)
    if rank == 1:
        if prior == 'Uniform':
            theta = np.random.uniform(-2, 1, size=n)
        if prior == 'Two_points':
            theta = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
        if prior == 'Point_normal':
            point_obs = np.repeat(0, n)
            assignment = np.random.binomial(n=1, p=0.5, size=n)
            normal_obs = np.random.normal(loc=0, scale=np.sqrt(2), size=n)
            theta = point_obs * (1 - assignment) + normal_obs * assignment
    elif rank == 2:
        if prior == 'Uniform_circle':
            theta_theta = np.random.uniform(0, 2 * np.pi, n)
            theta = np.array([np.cos(theta_theta), np.sin(theta_theta)]).T
        elif prior == 'Three_points':
            Three_points = [[-1, 1], [0, -1], [1, 1]]
            theta_choice = np.random.choice([i for i in range(3)],
                                             size=n, replace=True, p=np.array([1 / 4, 1 / 2, 1 / 4]))
            theta = np.array([Three_points[i] for i in theta_choice])
        theta = normalize_pc(theta)
    return theta

def signal_plus_noise_model(u, v, s, rank=1):
    n = u.shape[0]
    d = v.shape[0]
    W = np.random.normal(0, np.sqrt(1/n), n*d).reshape((n, d))
    if rank == 1:
        A = s * 1 / n * np.outer(u, v) + W
    elif rank == 2:
        A = 1 / n * u @ s @ v.T + W
    return A
