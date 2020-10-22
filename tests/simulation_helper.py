import numpy as np
# helper functions for simulation

def get_alignment(theta, theta_hat):
    prod = np.inner(theta, theta_hat) / \
           (np.sqrt(np.sum(theta ** 2)) * np.sqrt(np.sum(theta_hat ** 2)))
    return np.abs(prod)

def normalize_2nd_moment(u):
    return u / np.sqrt(np.sum(u**2)) * np.sqrt(n)

def simulate_prior(prior, n=2000, seed=1):
    '''
    simulate 3 distributions with 2nd moment=1
    Uniform: continuous prior
    Two_points: degenerative prior
    Point_normal: sparse prior
    '''
    np.random.seed(seed)
    if prior == 'Uniform':
        theta = np.random.uniform(-np.sqrt(3), np.sqrt(3), size = n)
    if prior == 'Two_points':
        theta = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
    if prior == 'Point_normal':
        point_obs = np.repeat(0, n)
        assignment = np.random.binomial(n=1, p=0.1, size=n)
        normal_obs = np.random.normal(loc=0, scale=np.sqrt(10), size=n)
        theta = point_obs * (1 - assignment) + normal_obs * assignment
    return theta

def pdf_prior(prior, theta):
    x = theta
    if prior == 'Uniform':
        pi = np.repeat(1 / (2 * np.sqrt(3)), x.shape[0])
    if prior == 'Two_points':
        pi = np.repeat(0.5, x.shape[0])
    if prior == 'Point_normal':
        pi = 0.1 * (1 / (np.sqrt(2 * np.pi) * np.sqrt(10)) * np.exp(- x ** 2 / (2 * 10))) \
             + 0.9 * np.array([int(x0 == 0) for x0 in x])
    pi = pi / np.sum(pi)
    return x, pi

def convolve_noise(theta, mu = 0.5, s = 1):
    n = len(theta)
    return mu * theta + np.random.normal(loc=0, scale=s, size=n)

def simulate_rank1_model(u, v, s):
    n = u.shape[0]
    d = v.shape[0]
    W = np.random.normal(0, np.sqrt(1/n), n*d).reshape((n, d))
    A = s * 1 / n * np.outer(u, v) + W
    return A

