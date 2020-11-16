# A collection of helper functions for rank one and rank two simulations
import numpy as np

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
