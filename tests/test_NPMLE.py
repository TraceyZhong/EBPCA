# -------------------------------------------------------
# Compare mosek, cvxpy and EM for solving NPMLE
# on a compound dicision problem
#
# Sep 17, 2020
# Chang Su, c.su@yale.edu
# -------------------------------------------------------

import mosek
from mosek.fusion import *
import cvxpy as cp
import numpy as np
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from ebpca.empbayes import _mosek_npmle, _npmle_em, get_W, NonparEB, NonparEBHD

# -------------------------------------------------------
# Implementation of NPMLE in three different software /
# algorithms:
# mosek, cvxpy (both for convex optimization)
# and EM
# -------------------------------------------------------

def test_cvxpy(A,  n=2000, solver='ECOS'):
    '''
    Use cvxpy to solve for the primal function of NPMLE
    See section 4.2 in https://www.tandfonline.com/doi/pdf/10.1080/01621459.2013.869224
    '''
    m = A.shape[1]
    g = cp.Variable(n)
    f = cp.Variable(m)
    constraints = [g >= 0, f >= 0, cp.sum(f) == 1, A @ f == g]
    objective_fn = cp.sum(cp.log(g))
    problem = cp.Problem(cp.Maximize(objective_fn), constraints)
    print("Problem is DCP:", problem.is_dcp())
    result = problem.solve(solver=solver, verbose = False) # ECOS,SCS
    print(problem.status)
    # print(result)
    print("cvxpy elapsed: {:.2f}".format(problem.solver_stats.solve_time))
    return f.value

# -------------------------------------------------------
# functions for simulating data and generating input
# to solvers
# -------------------------------------------------------

def gen_data(zTruthVar, n=2000, rank=2, muTruth=np.array([[2, 0], [0, 2]]), covTruth=np.array([[1, 0], [0, 1]])):
    '''
    Simulate data according to f|Z \sim N(mu Z, cov), Z \sim g
    where g is a mixture of 3 Gaussian (2dim) or a degenerative two-point prior (1dim)
    '''
    np.random.seed(1)
    ustar1 = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
    if rank == 2:
        ustar2 = np.repeat(np.array([1, -1, 0, 0]), int(n / 4))
        zTruth = np.vstack((ustar1, ustar2)).T + np.random.normal(scale=zTruthVar / 10, size=n * rank).reshape(
            (n, rank))
        zTruth = zTruth / np.sqrt(np.diag(zTruth.T @ zTruth)) * np.sqrt(n)
        np.random.seed(3333)
        f = zTruth.dot(muTruth.T) + np.random.multivariate_normal([0, 0], covTruth, size=n)
    if rank == 1:
        zTruth = ustar1
        f = zTruth * muTruth + np.random.normal(0, np.sqrt(covTruth), size=n)
    return f, zTruth

def gen_input(zTruthVar, n):
    '''
    Generate input to mosek and cvxpy solvers for NPMLE
    '''
    f, mu, cov, Z = gen_data(zTruthVar, n)
    if len(mu) > 1:
        covInv = np.linalg.inv(cov)
    else:
        covInv = 1/cov
    zEst = f
    A = get_W(f, zEst, mu, covInv)
    return A

# -------------------------------------------------------
# functions for evaluating the performance of
# NPMLE solutions on 2-dim data
# -------------------------------------------------------


def plot_data(zTruth, f, zTruthVar, figPrefix):
    '''
    Visualize simulated data
    '''
    fig, axes = plt.subplots(ncols=2, figsize=(2 * 6, 5))
    axes[0].set(xlim=(-2, 2), ylim=(-2, 2))
    axes[0].scatter(zTruth[:, 0], zTruth[:, 1])
    axes[0].set_title("Ground Truth Z")
    # axes[1].set(xlim=(-5, 5), ylim=(-5, 5))
    axes[1].scatter(f[:, 0], f[:, 1])
    axes[1].set_title("noisy F")
    plt.savefig(figPrefix + 'test_data_{}.png'.format(zTruthVar))
    plt.close()


def plot_pi(Z, zEst, pi, zTruthVar, figPrefix):
    '''
    Visualize prior estimates
    '''
    cmap = mpl.cm.Blues
    fig, axes = plt.subplots(ncols=2, figsize=(2 * 6, 5))
    axes[0].set(xlim=(-2, 2), ylim=(-2, 2))
    axes[0].scatter(Z[:, 0], Z[:, 1])
    axes[0].set_title("Ground Truth Z")
    axes[1].scatter(zEst[:, 0], zEst[:, 1], c=cmap(pi / np.max(pi) * 1000), alpha=0.5)
    axes[1].set_title("estimated Z")
    plt.savefig(figPrefix + 'piEst_{}.png'.format(zTruthVar))
    plt.close()

def plot_f(fObs, f2, fname, figPrefix):
    '''
    Visualize observed data
    '''
    fig, axes = plt.subplots(ncols=2, figsize=(2 * 6, 5))
    axes[0].set(xlim=(-5, 5), ylim=(-5, 5))
    axes[0].scatter(fObs[:, 0], fObs[:, 1])
    axes[0].set_title("Observed f")
    axes[1].set(xlim=(-5, 5), ylim=(-5, 5))
    axes[1].scatter(f2[:, 0], f2[:, 1])
    axes[1].set_title(fname)
    plt.savefig(figPrefix + '{}_{}.png'.format(fname, zTruthVar))
    plt.close()

def plot_priorEst_fDeno(zEst, pi, fDeno, figPrefix):
    '''
    Plot estimated prior versus denoised data
    '''
    cmap = mpl.cm.Blues
    fig, axes = plt.subplots(ncols=2, figsize=(2 * 6, 5))
    axes[0].set(xlim=(-2, 2), ylim=(-2, 2))
    axes[0].scatter(zEst[:, 0], zEst[:, 1], c=cmap(pi / np.max(pi) * 1000), alpha=0.5)
    axes[0].set_title("Estimated prior")
    # axes[1].set(xlim=(-5, 5), ylim=(-5, 5))
    axes[1].scatter(fDeno[:, 0], fDeno[:, 1])
    axes[1].set_title("Denoised")
    plt.savefig(figPrefix + 'prior_fDeno_{}.png'.format(zTruthVar))
    plt.close()

def reconvolute(Z, pi, muTruth, covTruth):
    '''
    Reconvolute noises to the estimated prior.
    Show that NPMLE is 'correct' in the sense of
    fitting marginal likelihood
    '''
    # draw n samples to simulate data from the estimated prior
    n = len(pi)
    np.random.seed(333)
    zSample = np.random.choice([i for i in range(n)], size=n, p=pi, replace=True)
    zEst = Z[zSample, :]
    fNew = zEst.dot(muTruth.T) + np.random.multivariate_normal([0, 0], covTruth, size=n)
    return fNew

def denoise(Z, pi, W):
    '''
    Denoise (compute posterior mean of) observed data
    based on estimated prior
    '''
    denom = W.dot(pi)
    num = W * pi
    P = num / denom[:, np.newaxis]
    return P @ Z

# test mosek in solving NPMLE
if __name__ == '__main__':
    figPrefix = 'figures/NPMLE'
    if not os.path.exists(figPrefix):
        os.mkdir(figPrefix)
        os.mkdir(figPrefix + '/univariate')
        os.mkdir(figPrefix + '/bivariate')

    for rank in [1]:
        print(rank)
        # set parameters
        zTruthVar = 2
        n = 1000
        if rank == 1:
            muTruth = np.array([2])
            covTruth = np.array([0.8])
            covInvTruth = 1 / covTruth
        elif rank == 2:
            muTruth = np.array([[2, 0], [0, 2]])  # np.array([2])
            covTruth = np.array([[1, 0], [0, 1]])  # np.array([0.8])
            covInvTruth = np.linalg.inv(covTruth)  # 1 / covTruth

        # simulate data
        f, zTruth = gen_data(zTruthVar, n, rank, muTruth, covTruth)
        if rank == 2:
            zEst = f.dot(np.linalg.pinv(muTruth).T)
        else:
            zEst = f / muTruth
        A = get_W(f, zEst, muTruth, covInvTruth)

        # run mosek, cvxpy and EM
        # to demonstrate that
        # 1) two convex op implementation gives the same results
        # 2) EM hasn't converged. By increasing the iterations the solution will be more similar to convex op

        mosek_res = []
        mosek_elapsed = []
        for tol in [1e-6, 1e-3, 1e-8]:
            # initiate mosek solver
            if rank == 1:
                denoiser = NonparEB(to_show=True, fig_prefix=figPrefix + '/univariate/', tol=tol)
            else:
                denoiser = NonparEBHD(to_show=True, fig_prefix=figPrefix + '/bivariate/', tol=tol)

            denoiser._check_init(f, muTruth)
            a = time.time()
            pi_mosek = denoiser.estimate_prior(f, muTruth, covTruth)
            b = time.time()
            denoiser.check_margin(f, muTruth, covTruth, 'tol' + str(tol))

            mosek_elapsed.append(b - a)
            mosek_res.append(pi_mosek)

        print('mosek elapsed: ', mosek_elapsed)
        print('mosek pi est under different tol:', \
              [np.round(sum((mosek_res[i] - mosek_res[-1])**2), 4) for i in range(2)])

        pi_cvx = test_cvxpy(A, n)
        print('cvxpy elapsed: {:.2f} s'.format(time.time() - b))  # much slower
        print('diff bt cvxpy and mosek estimates: %.4f' % sum((pi_cvx - pi_mosek) ** 2))

        em_elapsed = []
        for em_iter in [200, 2000]:
            a = time.time()
            pi_npmle = _npmle_em(A, em_iter, nsupp=f.shape[0])
            b = time.time()
            em_elapsed.append(b-a)
            print('diff bt EM (%i) and mosek estimates: %.4f' % (em_iter, sum((pi_cvx - pi_npmle) ** 2)))

        print('em elapsed ', em_elapsed)

        # visualize mosek estimates
        if rank == 1:
            figPrefix_rank = figPrefix + '/univariate/' + 'tol' + str(tol)
        elif rank == 2:
            figPrefix_rank = figPrefix + '/bivariate/' + 'tol' + str(tol)
            plot_data(zTruth, f, zTruthVar, figPrefix_rank)
            plot_pi(zTruth, zEst, pi_mosek, zTruthVar, figPrefix_rank)
            fDeno = denoise(zEst, pi_mosek, A)
            fRecon = reconvolute(zEst, pi_mosek, muTruth, covTruth)
            plot_f(f, fDeno, 'Denoised', figPrefix_rank)
            plot_f(f, fRecon, 'Reconvoluted', figPrefix_rank)
            plot_priorEst_fDeno(zEst, pi_mosek, fDeno, figPrefix_rank)

