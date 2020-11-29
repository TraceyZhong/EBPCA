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
    get_marginal_alignment, get_joint_alignment, regress_out_top

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
parser.add_argument("--n_copy", type=int, help="enter which independent data to be use",
                    default=1, const=1, nargs='?')
# set a small default iteration because the signal strength is large in this example
parser.add_argument("--iters", type=int, help="enter EB-PCA iterations",
                    default=5, const=5, nargs='?')
parser.add_argument("--s_star_1", type=float, help="true signal strength 1",
                    default=4, const=4, nargs='?')
parser.add_argument("--s_star_2", type=float, help="true signal strength 2",
                    default=2, const=2, nargs='?')
args = parser.parse_args()

method = args.method
prior = args.prior
n_copy = args.n_copy
iters = args.iters

print('\nRunning EB-PCA with %s estimation for bivariate prior: #%i replication, %s prior, %i iterations'\
      % (method, n_copy, prior, iters))

# set other fixed parameters for simulation
n = 1000
gamma = 1
d = int(n * gamma)
s_star = [args.s_star_1, args.s_star_2]
rank = 2
iters = 10
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

# define a general function to run EB-PCA with marginal estimation and joint estimation
def run_rankK_EBPCA(method, X, rank, iters):
    n, d = X.shape
    if method == 'joint':
        # prepare the PCA pack
        pcapack = get_pca(X, rank)
        # initiate denoiser
        udenoiser = NonparEB(optimizer="Mosek", to_save=False)
        vdenoiser = NonparEB(optimizer="Mosek", to_save=False)
        # run AMP
        U_est, V_est, conv = ebamp_gaussian(pcapack, iters=iters,
                                            udenoiser=udenoiser, vdenoiser=vdenoiser,
                                            return_conv=True)
        print('joint convergence ', conv)
    elif method == 'marginal':
        U_est = np.empty([n, rank, iters + 1])
        V_est = np.empty([d, rank, iters + 1])
        # initiate denoiser
        for j in range(rank):
            udenoiser = NonparEB(optimizer="Mosek", to_save=False)
            vdenoiser = NonparEB(optimizer="Mosek", to_save=False)
            if j > 0:
                # regress out top PC
                X = regress_out_top(X, j)
            # normalize data
            X = normalize_obs(X, 1)
            # prepare the PCA pack
            pcapack = get_pca(X, 1)
            # else:
            #     pcapack = get_pca(X, rank)
            # run AMP
            U_mar_est, V_mar_est, conv = ebamp_gaussian(pcapack, iters=iters,
                                                        udenoiser=udenoiser, vdenoiser=vdenoiser,
                                                        return_conv = True)
            print('marginal dim %i convergence ' % (j + 1), conv)
            U_est[:, j, :] = U_mar_est[:, 0, :]
            V_est[:, j, :] = V_mar_est[:, 0, :]
    return U_est, V_est

alignment = []

# generate simulation data
np.random.seed(1)
seeds = [np.random.randint(0, 10000, 50) for i in range(3)] # set seed for each dataset

# set i to be the nominal copy number - 1
i = n_copy - 1

if not os.path.exists('%s_copy_%i.npy' % (data_prefix, i)):
    print('Simulating data for replication %i' % i)
    # simulate data based on the chosen prior
    u_star = simulate_prior(prior, n, seed=seeds[0][i], rank=rank)
    v_star = simulate_prior(prior, d, seed=seeds[1][i], rank=rank)
    Y = signal_plus_noise_model(u_star, v_star, np.diag(s_star), seed=seeds[2][i], rank=rank)
    # normalize the observational matrix
    X = normalize_obs(Y, rank)
    np.save('%s_copy_%i_u_star.npy' % (data_prefix, i), u_star, allow_pickle=False)
    np.save('%s_copy_%i_v_star.npy' % (data_prefix, i), v_star, allow_pickle=False)
    np.save('%s_copy_%i.npy' % (data_prefix, i), X, allow_pickle=False)

# run EB-PCA with marginal / joint estimation of bivariate priors
u_alignment = []
v_alignment = []
start_time = time.time()

# Run EB-PCA for one replication

print('Replication %i' % i)
# load simulation datau_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, i), allow_pickle=False)
v_star = np.load('%s_copy_%i_v_star.npy' % (data_prefix, i), allow_pickle=False)
X = np.load('%s_copy_%i.npy' % (data_prefix, i), allow_pickle=False)

# run EB-PCA with designated method for prior estimation
U_est, V_est = run_rankK_EBPCA(method, X, rank, iters)

# evaluate marginal alignment
u_alignment.append(get_marginal_alignment(U_est, u_star))
v_alignment.append(get_marginal_alignment(V_est, v_star))

if i == 0:
    # save replication 0 results for making plots
    np.save('output/%s/denoisedPC/%s_%s_s_%.1f_%.1f_n_copy_%i.npy' %
            (prior_prefix, method, "U", s_star[0], s_star[1], i),
            U_est, allow_pickle=False)
    np.save('output/%s/denoisedPC/%s_%s_s_%.1f_%.1f_n_copy_%i.npy' %
            (prior_prefix, method, "V", s_star[0], s_star[1], i),
             V_est, allow_pickle=False)

end_time = time.time()
print('Simulation takes %.2f s' % (end_time - start_time))

print('\nright PC alignments:\n\t marginal:', u_alignment)
print('\t joint', get_joint_alignment(u_alignment))
print('\nleft PC alignments:\n\t marginal:', v_alignment)
print('\t joint', get_joint_alignment(v_alignment))

np.save('output/%s/alignments/%s_u_s_%.1f_%.1f_n_copy_%i.npy' % (prior_prefix, method, s_star[0], s_star[1], n_copy),
        u_alignment, allow_pickle=False)
np.save('output/%s/alignments/%s_v_s_%.1f_%.1f_n_copy_%i.npy' % (prior_prefix, method, s_star[0], s_star[1], n_copy),
        v_alignment, allow_pickle=False)

print('\n Simulation finished. \n')