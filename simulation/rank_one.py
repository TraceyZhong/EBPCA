import numpy as np
import os

import sys
sys.path.extend(['../../generalAMP'])
from ebpca.empbayes import NonparEB as NonparEB
from ebpca.amp import ebamp_gaussian as ebamp_gaussian
from ebpca.preprocessing import normalize_obs
from ebpca.pca import get_pca
from tutorial import get_alignment

# ----------------------
# Setup for simulation
# ----------------------

# take named argument from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prior", type=str, help="enter univariate prior", 
                    default='Uniform', const='Uniform', nargs='?')
parser.add_argument("--n_rep", type=int, help="enter number of independent data to be simulated",
                    default=1, const=1, nargs='?')
parser.add_argument("--s_star", type=float, help="enter signal strength", 
                    default=1.4, const=1.4, nargs='?')
parser.add_argument("--iters", type=int, help="enter EB-PCA iterations", 
                    default=5, const=5, nargs='?')
args = parser.parse_args()

prior = args.prior
n_rep = args.n_rep
s_star = args.s_star
iters = args.iters

print('\nRunning EB-PCA rank one simulations with %i replications, prior=%s, signal strength=%.1f, iterations=%i'\
      % (n_rep, prior, s_star, iters))

# create directory to save alignemnt, simulated data and figures
prior_prefix = 'univariate/' + prior
if not os.path.exists('output/' + prior_prefix):
    os.makedirs('output/%s/alignments' % prior_prefix)
    os.mkdir('output/%s/data' % prior_prefix)
data_prefix = 'output/%s/data/s_%.1f' % (prior_prefix, s_star)

# --------------
# util functions
# --------------

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
        theta = np.random.uniform(-np.sqrt(3), np.sqrt(3), size = n) # -np.sqrt(3), np.sqrt(3)
    if prior == 'Two_points':
        theta = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
    if prior == 'Point_normal':
        point_obs = np.repeat(0, n)
        assignment = np.random.binomial(n=1, p=0.1, size=n)
        normal_obs = np.random.normal(loc=0, scale=np.sqrt(10), size=n)
        theta = point_obs * (1 - assignment) + normal_obs * assignment
    return theta

def simulate_rank1_model(u, v, s):
    n = u.shape[0]
    d = v.shape[0]
    W = np.random.normal(0, np.sqrt(1/n), n*d).reshape((n, d))
    A = s * 1 / n * np.outer(u, v) + W
    return A

# -----------------
# rank-1 simulation
# -----------------

# set parameters for simulation
n = 2000
gamma = 2
d = int(n * gamma)
rank = 1
alignment = []

print('fixed parameters: n=%i, gamma=%.1f\n' % (n, gamma))

# generate simulation data
np.random.seed(1)
seeds = [np.random.randint(0, 10000, n_rep) for i in range(2)] # set seed for each dataset

for i in range(n_rep):
    if not os.path.exists('%s_copy_%i.npy' % (data_prefix, i)):
        # simulate data based on the chosen prior
        u_star = simulate_prior(prior, n, seed=seeds[0][i])
        v_star = simulate_prior(prior, d, seed=seeds[1][i])
        Y = simulate_rank1_model(u_star, v_star, s_star)
        # normalize the observational matrix
        X = normalize_obs(Y, rank)
        np.save('%s_copy_%i_u_star.npy' % (data_prefix, i), u_star, allow_pickle=False)
        np.save('%s_copy_%i_v_star.npy' % (data_prefix, i), v_star, allow_pickle=False)
        np.save('%s_copy_%i.npy' % (data_prefix, i), X, allow_pickle=False)

# run EB-PCA
u_alignment = []
v_alignment = []
for i in range(n_rep):
    print('Replication %i' % i)
    # load simulation data
    u_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, i), allow_pickle=False)
    v_star = np.load('%s_copy_%i_v_star.npy' % (data_prefix, i), allow_pickle=False)
    X = np.load('%s_copy_%i.npy' % (data_prefix, i), allow_pickle=False)
    # prepare the PCA pack
    pcapack = get_pca(X, rank)
    # initiate denoiser
    udenoiser = NonparEB(optimizer="Mosek", ftol=1e-3, to_save=False,
                         nsupp_ratio=np.sqrt(n)/n)
    vdenoiser = NonparEB(optimizer="Mosek", ftol=1e-3, to_save=False,
                         nsupp_ratio=np.sqrt(d)/d)
    # run AMP
    U_est, V_est = ebamp_gaussian(pcapack, iters=iters,
                                  udenoiser=udenoiser, vdenoiser=vdenoiser)
    # evaluate alignment
    u_alignment.append([get_alignment(U_est[:, :, j], u_star) for j in range(U_est.shape[2])])
    v_alignment.append([get_alignment(V_est[:, :, j], v_star) for j in range(V_est.shape[2])])

np.save('output/%s/alignments/u_s_%.1f_n_rep_%i.npy' % (prior_prefix, s_star, n_rep), u_alignment, allow_pickle=False)
np.save('output/%s/alignments/v_s_%.1f_n_rep_%i.npy' % (prior_prefix, s_star, n_rep), v_alignment, allow_pickle=False)

print('\n Simulation finished. \n')


