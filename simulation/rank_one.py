import numpy as np
import os
import time
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.empbayes import NonparEB, NonparBayes, PointNormalBayes, TwoPointsBayes
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
                    default=1.0, const=1.0, nargs='?')
parser.add_argument("--iters", type=int, help="enter EB-PCA iterations", 
                    default=10, const=10, nargs='?')
parser.add_argument("--gamma", type=float, help="enter d/n",
                    default=2.0, const=2.0, nargs='?')
parser.add_argument("--n", type=int, help="enter n",
                    default=1000, const=1000, nargs='?')
parser.add_argument("--nsupp_ratio_u", type=float, help="enter n",
                    default=1.0, const=1.0, nargs='?')
parser.add_argument("--nsupp_ratio_v", type=float, help="enter n",
                    default=1.0, const=1.0, nargs='?')
parser.add_argument("--saveDE", type=str, help="enter n",
                    default=False, const=False, nargs='?')
parser.add_argument("--bayesdenoiser", type=str, help="enter n",
                    default=False, const=False, nargs='?')
parser.add_argument("--useEM", type=str, help="enter n",
                    default=False, const=False, nargs='?')
args = parser.parse_args()

prior = args.prior
n_rep = args.n_rep
s_star = args.s_star
iters = args.iters
method = args.method
gamma = args.gamma
saveDE = args.saveDE
n = args.n
nsupp_ratio_u = args.nsupp_ratio_u
nsupp_ratio_v = args.nsupp_ratio_v
bayesdenoiser = args.bayesdenoiser
useEM = args.useEM

print('\nRunning %s rank one simulations with %i replications, %s prior, signal strength=%.1f, iterations=%i'\
      % (method, n_rep, prior, s_star, iters))

print('Other tuning parameters: n=%i, gamma=%.1f, nsupp_ratio_u=%.1f, nsupp_ratio_v=%.1f' % \
      (n, gamma, nsupp_ratio_u, nsupp_ratio_v))

print('UseEM: %s' % useEM)

# create directory to save alignemnt, simulated data and figures
tuning_par_prefix = '/n_%i_gamma_%.1f_nsupp_ratio_%.1f_%.1f_useEM_%s' % (n, gamma, nsupp_ratio_u, nsupp_ratio_v, useEM)
prior_prefix = 'univariate/' + prior + tuning_par_prefix
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
# n = 2000
# gamma = 2
d = int(n * gamma)
rank = 1
alignment = []

print('fixed parameters: n=%i, gamma=%.1f\n' % (n, gamma))

# generate simulation data
np.random.seed(1)
seeds = [np.random.randint(0, 10000, 50) for i in range(3)] # set seed for each dataset

for i in range(n_rep):
    if not os.path.exists('%s_copy_%i_n_%i_gamma_%.1f.npy' % (data_prefix, i, n, gamma)):
        print('Simulating data for replication %i' % i)
        # simulate data based on the chosen prior
        u_star = simulate_prior(prior, n, seed=seeds[0][i])
        v_star = simulate_prior(prior, d, seed=seeds[1][i])
        Y = signal_plus_noise_model(u_star, v_star, s_star, seed=seeds[2][i])
        # normalize the observational matrix
        X = normalize_obs(Y, rank)
        np.save('%s_copy_%i_u_star_n_%i_gamma_%.1f.npy' % (data_prefix, i, n, gamma), u_star, allow_pickle=False)
        np.save('%s_copy_%i_v_star_n_%i_gamma_%.1f.npy' % (data_prefix, i, n, gamma), v_star, allow_pickle=False)
        np.save('%s_copy_%i_n_%i_gamma_%.1f.npy' % (data_prefix, i, n, gamma), X, allow_pickle=False)

# run EB-PCA / BayesAMP / EBMF
u_alignment = []
v_alignment = []
obj_funcs = []
conv_trace = []
start_time = time.time()
for i in range(n_rep):
    print('Replication %i' % i)
    # load simulation data
    u_star = np.load('%s_copy_%i_u_star_n_%i_gamma_%.1f.npy' % (data_prefix, i, n, gamma), allow_pickle=False)
    v_star = np.load('%s_copy_%i_v_star_n_%i_gamma_%.1f.npy' % (data_prefix, i, n, gamma), allow_pickle=False)
    X = np.load('%s_copy_%i_n_%i_gamma_%.1f.npy' % (data_prefix, i, n, gamma), allow_pickle=False)
    if method == 'EB-PCA' or method == 'EBMF':
        # prepare the PCA pack
        pcapack = get_pca(X, rank)
        # initiate denoiser
        if bayesdenoiser:
            [truePriorLoc, truePriorWeight] = approx_prior(u_star, pcapack.U)
            udenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)
            [truePriorLoc, truePriorWeight] = approx_prior(v_star, pcapack.V)
            vdenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)
        elif useEM:
            udenoiser = NonparEB(em_iter = 400, to_save=False, nsupp_ratio=nsupp_ratio_u)
            vdenoiser = NonparEB(em_iter = 400, to_save=False, nsupp_ratio=nsupp_ratio_v)
        else:
            udenoiser = NonparEB(optimizer="Mosek", to_save=False, nsupp_ratio=nsupp_ratio_u)
            vdenoiser = NonparEB(optimizer="Mosek", to_save=False, nsupp_ratio=nsupp_ratio_v)
        if method == 'EB-PCA':
            # run AMP
            U_est, V_est, conv = ebamp_gaussian(pcapack, iters=iters,
                                            udenoiser=udenoiser, vdenoiser=vdenoiser,
                                            return_conv=True)
            conv_trace.append(conv)
        else:
            ldenoiser = NonparEB(optimizer="Mosek", to_save=False, nsupp_ratio=nsupp_ratio_u)
            fdenoiser = NonparEB(optimizer="Mosek", to_save=False, nsupp_ratio=nsupp_ratio_v)
            U_est, V_est, obj = ebmf(pcapack, ldenoiser, fdenoiser, iters=iters,
                                     ebpca_scaling=False, update_family='nonparametric', tol=1e-1)
            obj_funcs.append(obj)
    elif method == 'BayesAMP':
        # Oracle Bayes AMP takes true signal strength and true prior as input
        # prepare the PCA pack
        pcapack = get_pca(X, rank, s_star)
        # initiate denoiser
        if prior == 'Uniform':
            # here we put equal weights on observed PC data points to approximate the true Bayes denoiser
            [truePriorLoc, truePriorWeight] = approx_prior(u_star, pcapack.U)
            udenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)
            [truePriorLoc, truePriorWeight] = approx_prior(v_star, pcapack.V)
            vdenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)
        elif prior == 'Point_normal':
            udenoiser = PointNormalBayes(0.1, np.sqrt(10), to_save=False)
            vdenoiser = PointNormalBayes(0.1, np.sqrt(10), to_save=False)
        elif prior == 'Two_points':
            udenoiser = TwoPointsBayes(to_save=False)
            vdenoiser = TwoPointsBayes(to_save=False)
        # run AMP
        U_est, V_est, conv = ebamp_gaussian(pcapack, iters=iters,
                                      udenoiser=udenoiser, vdenoiser=vdenoiser,
                                      return_conv=True)
        conv_trace.append(conv)
    # evaluate alignment
    # maximal EBMF iterations: 50
    u_alignment.append(fill_alignment(U_est, u_star, iters))
    v_alignment.append(fill_alignment(V_est, v_star, iters))
    print(fill_alignment(U_est, u_star, iters))
    print(fill_alignment(V_est, v_star, iters))
    if saveDE:
        # save denoised PC along iterations
        print('Saving denoised PCs')
        np.save('output/%s/denoisedPC/%s_U_s_%.1f_n_copy_%i.npy' % (prior_prefix, method, s_star, i),
                U_est, allow_pickle=False)
        np.save('output/%s/denoisedPC/%s_V_s_%.1f_n_copy_%i.npy' % (prior_prefix, method, s_star, i),
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

# save convergence records
if method == 'EBMF':
    # save objective function values for EBMF
    np.save('output/%s/alignments/%s_obj_funcs_s_%.1f_n_rep_%i.npy' % (prior_prefix, method, s_star, n_rep),
            obj_funcs, allow_pickle=False)
    print('objective function differences:', [np.ediff1d(obj) for obj in obj_funcs])
else:
    # save convergence trace for EB-PCA or BayesAMP
    np.save('output/%s/alignments/conv_%s_s_%.1f_n_rep_%i.npy' % (prior_prefix, method, s_star, n_rep),
            conv_trace, allow_pickle=False)
    print(conv_trace)

print('\n Simulation finished. \n')
