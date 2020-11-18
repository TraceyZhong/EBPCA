import numpy as np
import os
import time
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.empbayes import NonparEB as NonparEB, NonparBayes, PointNormalBayes, TwoPointsBayes
from ebpca.amp import ebamp_gaussian as ebamp_gaussian
from ebpca.preprocessing import normalize_obs
from ebpca.pca import get_pca
from ebpca.misc import ebmf
from simulation.helpers import simulate_prior, signal_plus_noise_model, fill_alignment, approx_prior

# ----------------------
# Setup for simulation
# ----------------------

# take named argument from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, help="enter the method to test",
                    default='EB-PCA', const='EB-PCA', nargs='?')
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
method = args.method

print('\nRunning %s rank one simulations with %i replications, %s prior, signal strength=%.1f, iterations=%i'\
      % (method, n_rep, prior, s_star, iters))

# create directory to save alignemnt, simulated data and figures
prior_prefix = 'univariate/' + prior
if not os.path.exists('output/' + prior_prefix):
    print('Creating directories:')
    os.makedirs('output/%s/alignments' % prior_prefix)
    os.mkdir('output/%s/data' % prior_prefix)
    os.mkdir('output/%s/denoisedPC' % prior_prefix)
data_prefix = 'output/%s/data/s_%.1f' % (prior_prefix, s_star)


# -----------------
# rank-1 simulation
# -----------------

# set fixed parameters for simulation
n = 1000
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
        print('Simulating data for replication %i' % i)
        # simulate data based on the chosen prior
        u_star = simulate_prior(prior, n, seed=seeds[0][i])
        v_star = simulate_prior(prior, d, seed=seeds[1][i])
        Y = signal_plus_noise_model(u_star, v_star, s_star)
        # normalize the observational matrix
        X = normalize_obs(Y, rank)
        np.save('%s_copy_%i_u_star.npy' % (data_prefix, i), u_star, allow_pickle=False)
        np.save('%s_copy_%i_v_star.npy' % (data_prefix, i), v_star, allow_pickle=False)
        np.save('%s_copy_%i.npy' % (data_prefix, i), X, allow_pickle=False)

# run EB-PCA / BayesAMP / EBMF
u_alignment = []
v_alignment = []
obj_funcs = []
start_time = time.time()
for i in range(n_rep):
    print('Replication %i' % i)
    # load simulation data
    u_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, i), allow_pickle=False)
    v_star = np.load('%s_copy_%i_v_star.npy' % (data_prefix, i), allow_pickle=False)
    X = np.load('%s_copy_%i.npy' % (data_prefix, i), allow_pickle=False)
    # prepare the PCA pack
    pcapack = get_pca(X, rank)
    if method == 'EB-PCA':
        # initiate denoiser
        udenoiser = NonparEB(optimizer="Mosek", to_save=False)
        vdenoiser = NonparEB(optimizer="Mosek", to_save=False)
        # run AMP
        U_est, V_est = ebamp_gaussian(pcapack, iters=iters,
                                      udenoiser=udenoiser, vdenoiser=vdenoiser)
    elif method == 'BayesAMP':
        # initiate denoiser
        if prior == 'Uniform':
            # here we put equal weights on observed PC data points to approximate the true Bayes denoiser
            [truePriorLoc, truePriorWeight] = approx_prior(u_star, pcapack.U)
            udenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)
            [truePriorLoc, truePriorWeight] = approx_prior(v_star, pcapack.V)
            vdenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)
        elif prior == 'Point_normal':
            udenoiser = PointNormalBayes(0.1, np.sqrt(10))
            vdenoiser = PointNormalBayes(0.1, np.sqrt(10))
        elif prior == 'Two_points':
            udenoiser = TwoPointsBayes()
            vdenoiser = TwoPointsBayes()
        # run AMP
        U_est, V_est = ebamp_gaussian(pcapack, iters=iters,
                                      udenoiser=udenoiser, vdenoiser=vdenoiser)
    elif method == 'EBMF':
        ldenoiser = NonparEB(optimizer="Mosek", to_save=False)
        fdenoiser = NonparEB(optimizer="Mosek", to_save=False)
        U_est, V_est, obj = ebmf(pcapack, ldenoiser, fdenoiser, iters=iters,
                                 ebpca_scaling=False, update_family='nonparametric', tol=1e-1)
        obj_funcs.append(obj)
    # evaluate alignment
    # maximal EBMF iterations: 50
    u_alignment.append(fill_alignment(U_est, u_star, iters))
    v_alignment.append(fill_alignment(V_est, v_star, iters))
    # save denoised PC along iterations
    np.save('output/%s/denoisedPC/%s_leftPC_s_%.1f_n_copy_%i.npy' % (prior_prefix, method, s_star, i),
            U_est, allow_pickle=False)
    np.save('output/%s/denoisedPC/%s_rightPC_s_%.1f_n_copy_%i.npy' % (prior_prefix, method, s_star, i),
            V_est, allow_pickle=False)

end_time = time.time()
print('Simulation takes %.2f s' % (end_time - start_time))

print('right PC alignments:', u_alignment)
print('left PC alignments:', v_alignment)

# save alignments
np.save('output/%s/alignments/%s_u_s_%.1f_n_rep_%i.npy' % (prior_prefix, method, s_star, n_rep),
        u_alignment, allow_pickle=False)
np.save('output/%s/alignments/%s_v_s_%.1f_n_rep_%i.npy' % (prior_prefix, method, s_star, n_rep),
        v_alignment, allow_pickle=False)
# save objective function values for EBMF
if method == 'EBMF':
    np.save('output/%s/alignments/%s_obj_funcs_s_%.1f_n_rep_%i.npy' % (prior_prefix, method, s_star, n_rep),
            obj_funcs, allow_pickle=False)
    print('objective function differences:', [np.ediff1d(obj) for obj in obj_funcs])

print('\n Simulation finished. \n')
