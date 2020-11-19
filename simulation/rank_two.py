import numpy as np
import os
import time
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.empbayes import NonparEB as NonparEB
from ebpca.amp import ebamp_gaussian as ebamp_gaussian
from ebpca.preprocessing import normalize_obs
from ebpca.pca import get_pca
from simulation.helpers import simulate_prior, signal_plus_noise_model, \
    fill_alignment, get_joint_alignment, regress_out_top

# ----------------------
# Setup for simulation
# ----------------------

# take named argument from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, help="enter the version of EB-PCA to run",
                    default='joint', const='joint', nargs='?')
parser.add_argument("--prior", type=str, help="enter bivariate prior",
                    default='Uniform_circle', const='Uniform_circle', nargs='?')
parser.add_argument("--n_rep", type=int, help="enter number of independent data to be simulated",
                    default=1, const=1, nargs='?')
parser.add_argument("--iters", type=int, help="enter EB-PCA iterations",
                    default=5, const=5, nargs='?')
args = parser.parse_args()

method = args.method
prior = args.prior
n_rep = args.n_rep
iters = args.iters

print('\nRunning EB-PCA %s estimation on two PCs rank two simulations with %i replications, %s prior, %i iterations'\
      % (method, n_rep, prior, iters))

# set other fixed parameters for simulation
n = 1000
gamma = 2
d = int(n * gamma)
s_star = [1.8, 1.3]
rank = 2

print('fixed parameters: n=%i, gamma=%.1f, s=%.1f, %.1f\n' % (n, gamma, s_star[0], s_star[1]))

# create directory to save alignment, simulated data and figures
prior_prefix = 'bivariate/' + prior
if not os.path.exists('output/' + prior_prefix):
    os.makedirs('output/%s/alignments' % prior_prefix)
    os.mkdir('output/%s/data' % prior_prefix)
data_prefix = 'output/%s/data/s_%.1f_%.1f' % (prior_prefix, s_star[0], s_star[1])

# -----------------
# rank-2 simulation
# -----------------

alignment = []

# generate simulation data
np.random.seed(1)
seeds = [np.random.randint(0, 10000, n_rep) for i in range(2)] # set seed for each dataset

for i in range(n_rep):
    if not os.path.exists('%s_copy_%i.npy' % (data_prefix, i)):
        print('Simulating data for replication %i' % i)
        # simulate data based on the chosen prior
        u_star = simulate_prior(prior, n, seed=seeds[0][i], rank=rank)
        v_star = simulate_prior(prior, d, seed=seeds[1][i], rank=rank)
        Y = signal_plus_noise_model(u_star, v_star, np.diag(s_star), rank)
        # normalize the observational matrix
        X = normalize_obs(Y, rank)
        np.save('%s_copy_%i_u_star.npy' % (data_prefix, i), u_star, allow_pickle=False)
        np.save('%s_copy_%i_v_star.npy' % (data_prefix, i), v_star, allow_pickle=False)
        np.save('%s_copy_%i.npy' % (data_prefix, i), X, allow_pickle=False)

# run EB-PCA with marginal / joint estimation of bivariate priors
u_alignment = []
v_alignment = []
start_time = time.time()
for i in range(n_rep):
    print('Replication %i' % i)
    # load simulation data
    u_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, i), allow_pickle=False)
    v_star = np.load('%s_copy_%i_v_star.npy' % (data_prefix, i), allow_pickle=False)
    X = np.load('%s_copy_%i.npy' % (data_prefix, i), allow_pickle=False)
    if method == 'joint':
        # prepare the PCA pack
        pcapack = get_pca(X, rank)
        # initiate denoiser
        udenoiser = NonparEB(optimizer="Mosek", to_save=False)
        vdenoiser = NonparEB(optimizer="Mosek", to_save=False)
        # run AMP
        U_est, V_est = ebamp_gaussian(pcapack, iters=iters,
                                      udenoiser=udenoiser, vdenoiser=vdenoiser)
    elif method == 'marginal':
        U_est = np.zeros([n, rank, iters + 1])
        V_est = np.zeros([d, rank, iters + 1])
        # initiate denoiser
        for j in range(rank):
            udenoiser = NonparEB(optimizer="Mosek", to_save=False)
            vdenoiser = NonparEB(optimizer="Mosek", to_save=False)
            # regress out top PC
            X = regress_out_top(X, j)
            # normalize data
            X = normalize_obs(X, 1)
            # prepare the PCA pack
            pcapack = get_pca(X, 1)
            # run AMP
            U_mar_est, V_mar_est = ebamp_gaussian(pcapack, iters=iters,
                                                  udenoiser=udenoiser, vdenoiser=vdenoiser)
            U_est[:, j, :] = U_mar_est[:, 0, :]
            V_est[:, j, :] = V_mar_est[:, 0, :]

    # evaluate alignment
    u_alignment.append([fill_alignment(U_est[:,[j],:], u_star[:,[j]], iters) for j in range(rank)])
    v_alignment.append([fill_alignment(V_est[:,[j],:], v_star[:,[j]], iters) for j in range(rank)])

end_time = time.time()
print('Simulation takes %.2f s' % (end_time - start_time))

print('\nright PC alignments:\n\t marginal:', u_alignment)
print('\t joint alignments:', get_joint_alignment(u_alignment))
print('\nleft PC alignments:\n\t marginal:', v_alignment)
print('\t joint alignments:', get_joint_alignment(v_alignment))

np.save('output/%s/alignments/%s_u_s_%.1f_%.1f_n_rep_%i.npy' % (prior_prefix, method, s_star[0], s_star[1], n_rep),
        u_alignment, allow_pickle=False)
np.save('output/%s/alignments/%s_v_s_%.1f_%.1f_n_rep_%i.npy' % (prior_prefix, method, s_star[0], s_star[1], n_rep),
        v_alignment, allow_pickle=False)

print('\n Simulation finished. \n')