import numpy as np
import os

import sys
sys.path.extend(['../generalAMP'])
from ebpca.empbayes import NonparEB as NonparEB
from ebpca.amp import ebamp_gaussian_active as ebamp_gaussian
from ebpca.preprocessing import normalize_obs
from ebpca.pca import get_pca, check_residual_spectrum
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
                    default=10, const=10, nargs='?')
args = parser.parse_args()

prior = args.prior
n_rep = args.n_rep
s_star = args.s_star
iters = args.iters

print('\n Running EB-PCA %i simulations with prior=%s, signal strength=%.1f, iterations=%i \n'\
      % (n_rep, prior, s_star, iters))

# create directory to save alignemnt, simulated data and figures
# where saving data is for reproducible purpose
if not os.path.exists('output/%s/' % prior):
    os.mkdir('output/%s/' % prior)
    os.mkdir('output/%s/alignments' % prior)
    os.mkdir('output/%s/data' % prior)

fig_prefix = 'figures/simulation/univariate/'
if not os.path.exists(fig_prefix):
   os.mkdir(fig_prefix)

# save data for reproducibility
if os.path.exists('output/%s/data/copy_%i.npy' % (prior, n_rep - 1)):
    save_data = False
else:
    save_data = True

# save plots only when replication=1
if n_rep == 1:
    to_save = True
    fig_prefix = fig_prefix + '%s/' % prior
else:
    to_save = False
    fig_prefix = '.'

# --------------
# util functions
# --------------

# functions for simulating priors and data under the EB-PCA model
def simulate_prior(prior, n=2000, seed=1):
    '''
    simulate 3 distributions with 2nd moment=1
    Uniform: represents continuous prior
    Two_points: represents degenerative prior / cluster structure
    Point_normal: represents sparse prior
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
n = 1000
alpha = 2
d = int(n * alpha)
rank = 1
alignment = []

# set seed(s) for reproducibility
np.random.seed(1)
seeds = [np.random.randint(0, 10000, n_rep) for i in range(2)]

for i in range(n_rep):
    # simulate data based on the chosen prior
    u_star = simulate_prior(prior, n, seed = seeds[0][i])
    v_star = simulate_prior(prior, d, seed = seed[1][i])
    Y = simulate_rank1_model(u_star, v_star, s_star)
    # Normalize the observational matrix.
    X = normalize_obs(Y, rank)
    # Prepare the PCA pack.
    pcapack = get_pca(X, rank)
    # save simulated data
    # to be consistent across different methods
    if save_data:
        np.save('output/%s/data/copy_%i.npy' % (prior, i), pcapack.X, allow_pickle=False)
        np.save('output/%s/data/copy_u_%i.npy' % (prior, i), pcapack.U, allow_pickle=False)
        np.save('output/%s/data/copy_vh_%i.npy' % (prior, i), pcapack.V, allow_pickle=False)

    # initiate denoiser
    udenoiser = NonparEB(optimizer="Mosek", ftol=1e-3, to_save=to_save, fig_prefix=fig_prefix)
    vdenoiser = NonparEB(optimizer="Mosek", ftol=1e-3, to_save=to_save, fig_prefix=fig_prefix)
    
    # run AMP
    U_est, V_est = ebamp_gaussian(pcapack, iters=iters,
                                  udenoiser=udenoiser, vdenoiser=vdenoiser,
                                  figprefix='s_%.1f_alpha_%.1f' % (s_star, alpha))
    
    alignment.append([get_alignment(U_est[:, i], u_star) for i in range(U_est.shape[1])])

np.save('output/%s/alignments/n_rep_%i.npy' % (prior, n_rep), alignment, allow_pickle=False)

print('\n Simulation finished. \n')


