import numpy as np
from scipy.linalg import svd
import os
from simulation_helper import *

import sys
sys.path.extend(['/gpfs/ysm/project/zf59/cs/empiricalbayespca/generalAMP'])
from ebpca.empbayes import NonparEB
from ebpca.amp import ebamp_gaussian
from ebpca.pca import signal_solver_gaussian

# take named argument from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prior", type=str, help="enter univariate prior", 
                    default='Uniform', const='Uniform', nargs='?')
parser.add_argument("--n_data", type=int, help="enter number of independent data to be simulated", 
                    default=1, const=1, nargs='?')
parser.add_argument("--s_star", type=float, help="enter signal strength", 
                    default=1.4, const=1.4, nargs='?')
parser.add_argument("--iters", type=int, help="enter EB-PCA iterations", 
                    default=10, const=10, nargs='?')
args = parser.parse_args()

prior = args.prior
n_data = args.n_data
s_star = args.s_star
iters = args.iters

print('\n Running EB-PCA %i simulations with prior=%s, signal strength=%.1f, iterations=%i \n'\
      % (n_data, prior, s_star, iters))

# create directory to save alignemnt, simulated data and figures
# where saving data is for reproducible purpose
if not os.path.exists('output/%s/' % prior):
    os.mkdir('output/%s/' % prior)
    os.mkdir('output/%s/alignments' % prior)
    os.mkdir('output/%s/data' % prior)

figprefix = 'figures/simulation/univariate/'
if not os.path.exists(figprefix):
   os.mkdir(figprefix)

n = 1000
alpha = 1.2
d = int(n * alpha)

# rank-1 simulation

alignment = []

# set seed and save data for reproducibility
np.random.seed(1)
if os.path.exists('output/%s/data/copy_%i.npy' % (prior, n_data - 1)):
    save_data = False
else:
    save_data = True

for i in range(n_data):
    # simulate data based on the chosen prior
    u_star = simulate_prior(prior, n)
    v_star = simulate_prior(prior, d)
    A = simulate_rank1_model(u_star, v_star, s_star)
    u, s, vh = svd(A, full_matrices=False)
    u = u[:, 0]
    vh = vh[0, :]
    
    # save simulated data
    # to be consistent across different methods]
    if save_data:
        np.save('output/%s/data/copy_%i.npy' % (prior, i), A, allow_pickle=False)
        np.save('output/%s/data/copy_u_%i.npy' % (prior, i), u, allow_pickle=False)
        np.save('output/%s/data/copy_vh_%i.npy' % (prior, i), vh, allow_pickle=False)
    
    # estimate parameters for AMP
    init_pars = signal_solver_gaussian(singval=s, n_samples=d, n_features=n)
    
    # initiate denoiser
    udenoiser = NonparEB(to_save=True, to_show=False, fig_prefix=figprefix + '%s/' % prior)
    vdenoiser = NonparEB(to_save=True, to_show=False, fig_prefix=figprefix + '%s/' % prior)
    
    # run AMP
    U_est, V_est = ebamp_gaussian(A, u, vh, init_pars, iters=iters,
                                  udenoiser=udenoiser, vdenoiser=vdenoiser,
                                  figprefix='s_%.1f' % s_star)
    
    alignment.append([get_alignment(U_est[:, i], u_star) for i in range(U_est.shape[1])])

np.save('output/%s/alignments/n_data_%i.npy' % (prior, n_data), alignment, allow_pickle=False)

print('\n Simulation finished. \n')


