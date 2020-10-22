import numpy as np
from scipy.linalg import svd
import os
import sys
sys.path.extend(['/gpfs/ysm/project/zf59/cs/empiricalbayespca/generalAMP'])

from ebpca.empbayes import NonparEB
from ebpca.amp import ebamp_gaussian
from ebpca.pca import signal_solver_gaussian

# rank-1 simulation

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

if __name__ == '__main__':
    
    # test AMP
    # sanity check: AMP works and the marginals are correct
    figprefix = 'figures/simulation/univariate/'
    if not os.path.exists(figprefix):
        os.mkdir(figprefix)
    n = 1000
    alpha = 1.2
    d = int(n * alpha)
    s_star = 1.6
    iters = 10

    for prior in ['Uniform', 'Two_points', 'Point_normal']:
        if not os.path.exists(figprefix + prior):
            os.mkdir(figprefix + prior)
        # simulate data by the AMP model
        u_star = simulate_prior(prior, n)
        v_star = simulate_prior(prior, d)
        A = simulate_rank1_model(u_star, v_star, s_star)
        u, s, vh = svd(A, full_matrices=False)
        u = u[:, 0]
        vh = vh[0, :]

        # estimate parameters for AMP
        init_pars = signal_solver_gaussian(singval=s, n_samples=d, n_features=n)

        # initiate AMP
        udenoiser = NonparEB(to_save=True, to_show=False,
                             fig_prefix=figprefix + '%s/' % prior)
        vdenoiser = NonparEB(to_save=True, to_show=False,
                             fig_prefix=figprefix + '%s/' % prior)
        # run AMP
        print(init_pars["sample_align"][0])
        print(init_pars["alpha"][0])
        U_est, V_est = ebamp_gaussian(A, u, vh, init_pars, iters=iters,
                                      udenoiser=udenoiser, vdenoiser=vdenoiser,
                                      figprefix='%i_' % s_star)

        print([get_alignment(U_est[:, i], u_star) for i in range(U_est.shape[1])])

    # test denoiser
    n = 1000
    mu = np.array([0.5])
    sigma = np.array([2])
    denoiser = NonparEB(to_save=True, to_show=False,
                        fig_prefix="figures/simulation/univariate/")

    # for prior in ['Uniform', 'Two_points', 'Point_normal']:
    #     # sanity check for NPMLE
    #     theta = simulate_prior(prior, n)
    #     x = convolve_noise(theta, mu, sigma)
    #     Z_star, pi_star = pdf_prior(prior, x)
    #     denoiser.fit(x, mu, sigma ** 2, Z_star=Z_star, pi_star=pi_star, figname='%s' % prior)


