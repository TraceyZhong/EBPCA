# A collection of helper functions for rank one and rank two simulations
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from tutorial import get_alignment, redirect_pc, normalize_pc

# --------------------
# rank one simulations
# --------------------

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
            theta = np.random.uniform(0, np.sqrt(3), size=n)
        if prior == 'Two_points':
            theta = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
        if prior == 'Point_normal_0.1':
            point_obs = np.repeat(0, n)
            assignment = np.random.binomial(n=1, p=0.1, size=n)
            normal_obs = np.random.normal(loc=0, scale=np.sqrt(10), size=n)
            theta = point_obs * (1 - assignment) + normal_obs * assignment
        if prior == 'Point_normal_0.5':
            point_obs = np.repeat(0, n)
            assignment = np.random.binomial(n=1, p=0.5, size=n)
            normal_obs = np.random.normal(loc=0, scale=np.sqrt(2), size=n)
            theta = point_obs * (1 - assignment) + normal_obs * assignment
        if prior == 'Exponential':
            theta = np.random.exponential(1 / np.sqrt(2), size=n)
        if prior == 'Exponential_centered':
            theta = np.random.exponential(1 / np.sqrt(2), size=n)
            theta = theta - np.mean(theta)
            theta = theta / np.sqrt(np.mean(theta**2))
        if prior == 'Poisson':
            theta = np.random.poisson(1, size=n)
            theta = theta / np.sqrt(np.mean(theta ** 2))
        if prior == 'Poisson_centered':
            theta = np.random.poisson(1, size=n)
            theta = theta - np.mean(theta)
            theta = theta / np.sqrt(np.mean(theta ** 2))
        if prior == 'Beta':
            theta = np.random.beta(2, 5, size=n)
            theta = theta / np.sqrt(np.mean(theta ** 2))
        if prior == 'Beta_centered':
            theta = np.random.beta(2, 5, size=n)
            theta = theta - np.mean(theta)
            theta = theta / np.sqrt(np.mean(theta ** 2))
        if prior == 'Uniform_centered':
            theta = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=n)
        if prior == 'Normal':
            theta = np.random.normal(loc=0, scale=np.sqrt(1), size=n)
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

def signal_plus_noise_model(u, v, s, seed, rank = 1):
    np.random.seed(seed)
    n = u.shape[0]
    d = v.shape[0]
    W = np.random.normal(0, np.sqrt(1/n), n*d).reshape((n, d))
    if rank == 1:
        A = s * 1 / n * np.outer(u, v) + W
    elif rank == 2:
        A = 1 / n * u @ s @ v.T + W
    return A

# --------------------
# rank two simulations
# --------------------

def get_marginal_alignment(est, star):
    """
    Evaluate marginal alignments pf estimated PC
    """
    rank = est.shape[1]
    if len(est.shape) > 2:
        # Evaluate alignments for sequences of estimates
        iters = est.shape[2]
        return [fill_alignment(est[:, [j], :], star[:, [j]], iters) for j in range(rank)]
    else:
        # Evaluate alignments for one dePC
        return [get_alignment(est[:, [j]], star[:, [j]]) for j in range(rank)]

def get_space_distance(U,V):
    assert U.shape == V.shape
    Qu, _ = np.linalg.qr(U, mode = "reduced")
    Qv, _ = np.linalg.qr(V, mode = "reduced")
    _, s, _ = np.linalg.svd(Qu.T.dot(Qv))
    print(s)
    return np.sqrt(1 - np.min(s)**2)

def get_joint_error(U_est, Ustar, iterates=True):
    pass

def get_joint_alignment(mar, iterates=True):
    """
    Evaluate the joint alignment
    iterates: if to evaluate the marginal alignments from a multiple run experiment
    where the mar is of dimension n_rep * rank * iters
    """
    if iterates:
        joint = np.sqrt(np.mean(np.power(mar, 2), axis = 1))
    else:
        joint = np.sqrt(np.mean(np.power(mar, 2)))
    return joint

def regress_out_top(X, i):
    """
    Regress out the top i PCs
    """
    print('Regress out PC %i' % i)
    U, Lambdas, Vh = np.linalg.svd(X, full_matrices=False)
    X = X - U[:, :i].dot(np.diag(Lambdas[:i])).dot(Vh[:i, :])
    return X

def align_pc(pc, reference):
    '''sample usage see examples/showcase.py
    we use it to convert the pc estimates s.t. it has the same sign and direction as the ground truth pc
    '''
    pc = redirect_pc(pc, reference)
    scale_star = np.sqrt((reference ** 2).sum(axis=0))
    pc = normalize_pc(pc) / np.sqrt(len(pc)) * scale_star
    return pc

def get_error(align):
    return np.sqrt(1 - align**2)