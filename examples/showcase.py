'''
==================
Real data showcase
==================

This script includes the demonstration of EB-PCA method on four real datasets.
For each dataset, we will compare four sets of PCs:
    1. 'Ground truth' estimated from full data
    2. PCA estimates from a subset
    3. EB-PCA estimates with marginal estimation of bivariate priors
    3. EB-PCA estimates with joint estimation of bivariate priors
The cleaning procedure for these datasets are included in ./data/.

'''

import numpy as np
import os
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.preprocessing import normalize_obs, plot_pc
from ebpca.pca import get_pca, check_residual_spectrum
from simulation.helpers import align_pc
from simulation.rank_two import run_rankK_EBPCA
import matplotlib.pyplot as plt
from visualization import vis_2dim_subspace
from tutorial import get_alignment
import time

# load cleaned full data
def load_data(data_name):
    data_dir = 'data/' + data_name
    if data_name == '1000G':
        full_data = np.load(data_dir + '/normalized_1000G_1e5.npy')
    elif data_name == 'UKBB':
        full_data = np.load(data_dir + '/normalized_UKBB_1e5.npy')
    elif data_name == 'PBMC':
        full_data = np.load(data_dir + '/GTEx_full_norm.npy')
    elif data_name == 'GTEx':
        full_data = np.load(data_dir + '/pbmc_norm_clean.npy')
    return full_data

# Make directories to save figures
def make_dir(data_name):
    if not os.path.exists('figures/{}'.format(data_name)):
        print('Creating directories for saving {} figures and results'.format(data_name))
        os.mkdir('results/{}'.format(data_name))
        os.mkdir('figures/{}'.format(data_name))
    else:
        print('Directories for {} already exist'.format(data_name))

# Normalize data to equalize the effect of samples
def normalize_samples(data, rows_as_samples=True):
    if rows_as_samples:
        axi = 1
    else:
        axi = 0
    # input data: #samples * #features
    sample_mean = np.mean(data, axis=axi)
    sample_sd = np.std(data, axis=axi)
    if np.any(sample_sd < 1e-6):
        print('{} samples have zero sd'.format(np.sum(sample_sd == 0)))
    subset = sample_sd >= 1e-6
    if rows_as_samples:
        data = data[subset, :]
        norm = (data - sample_mean[subset, np.newaxis]) / sample_sd[subset, np.newaxis]
    else:
        data = data[:, subset]
        norm = (data - sample_mean[np.newaxis, subset]) / sample_sd[np.newaxis, subset]
    return norm

# Generate random subsets to demonstrate our methods
# Note that in the real data examples, the right PCs are of interest
# while the left PCs are not
def generate_subset(X_full_norm, n_sub, seed = 32423):
    np.random.seed(seed)
    m, n = X_full_norm.shape
    X_sub = X_full_norm[np.random.choice([i for i in range(n)], n_sub, replace=False), :]
    return X_sub

if __name__ == '__main__':
    # set parameters for different datasets

    # ---Rank---
    # Rank equals the number of PCs in the EB-PCA model
    # For each dataset, we manually inspect the singular value distribution
    # and identify the number of signal components
    real_data_rank = {'1000G': 4, 'UKBB': 2, 'PBMC': 2, }

    # ---Subset size---
    # The size of the random subsets
    subset_size = {'1000G': 1000, 'UKBB': 1000}

    # take arguments from command line
    # run single example for visualization or multiple replications to demonstrate quantitative performance
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="which dataset to show",
                        default='1000G', const='1000G', nargs='?')
    parser.add_argument("--n_rep", type=int, help="number of replications",
                        default=1, const=1, nargs='?')
    parser.add_argument("--iters", type=int, help="number of iterations",
                        default=5, const=5, nargs='?')
    args = parser.parse_args()
    data_name = args.data_name
    n_rep = args.n_rep
    iters = args.iters
    # only make plots when replication=1
    to_plot = (n_rep == 1)

    print('Analyzing dataset: %s, #replications = %i' % (data_name, n_rep))
    start_time = time.time()

    # ---------------------------------------------------
    # step 1: load data and explore ground truth
    # ---------------------------------------------------
    if not os.path.exists('results/%s/norm_data.npy' % data_name):
        print('Load raw data and apply normalization')
        # make directories to save output
        make_dir(data_name)

        # load data
        full_data = load_data(data_name)

        # transpose the data such that the data is #samples * #features
        # the right PCs (V) will be the vectors of interests
        # e.g. population identity, cell type identity
        # and the left PCs (U) are the samples from which we extract information to infer the left PCs
        full_data = np.transpose(full_data)
        print('full %s dataset have #samples=%i, #features=%i' % (data_name, full_data.shape[0], full_data.shape[1]))

        # normalize all samples to have the same variance
        # e.g. genetic variants / gene expressions to have the same variance
        # in practice we observe that this step can help with the interpretation of PCs
        # and avoids single sample dominating the PCs
        norm_data = normalize_samples(full_data, rows_as_samples=True)

        # normalize data to satisfy the EB-PCA assumption
        norm_data = normalize_obs(norm_data, real_data_rank[data_name])

        # save data
        np.save('results/%s/norm_data.npy' % data_name, norm_data)
    else:
        print('Load normalized data.')
        norm_data = np.load('results/%s/norm_data.npy' % data_name)

    if to_plot:
        # visualize spectrum of the residual matrix (noise)
        full_pcapack = get_pca(norm_data, real_data_rank[data_name])
        if not os.path.exists('figures/%s/residual_check_%s' % (data_name, data_name)):
            check_residual_spectrum(full_pcapack, to_save=True,
                                    fig_prefix=data_name, label=data_name)

        # visualize distributions of singular values and PCs
        if not os.path.exists('figures/%s/PC_0_%s' % (data_name, data_name)):
            # explore eigenvalue and eigenvector distributions
            plot_pc(norm_data, label=data_name, nPCs=real_data_rank[data_name],
                    to_show=False, to_save=True, fig_prefix='%s/' % data_name)

        # visualize the joint structure of PCs
        for i in range(int(real_data_rank[data_name] / 2) - 1):
            fig, ax = plt.subplots(figsize=(6, 4))
            vis_2dim_subspace(ax, full_pcapack.V[:, (2 * i):(2 * i + 2)], [1,1],
                              data_name, 'ground_truth_PCA',
                              data_dir='data/', to_save=True)

    # -------------------------------------------
    # step 3: Get random subset(s) from full data
    # -------------------------------------------
    # By subsetting we get noisy observations of the ground truth PCs,
    # and show EB-PCA successfully recovers the ground truth,
    # both quantitatively (alignments) and qualitatively (visualization of PC)

    # set seed
    np.random.seed(1)
    seeds = np.random.randint(0, 10000, 50)  # set seed for each dataset

    for i in range(n_rep):
        X = generate_subset(norm_data, subset_size[data_name], seed = seeds[0])

        # normalize data to satisfy the EB-PCA assumption
        X = normalize_obs(X, real_data_rank[data_name])

        # get pca estimates
        sub_pcapack = get_pca(X, real_data_rank[data_name])

        # visualize naive PCA
        if to_plot:
            # evaluate alignment
            PCA_align = [get_alignment(sub_pcapack.V[:, [j]], full_pcapack.V[:, [j]]) \
                         for j in range(real_data_rank[data_name])]
            # align the direction and scale of sample PCs with the true PC
            sub_PCs = sub_pcapack.V
            sub_PCs = align_pc(sub_PCs, full_pcapack.V)

            for i in range(int(real_data_rank[data_name] / 2) - 1):
                fig, ax = plt.subplots(figsize=(6, 4))
                vis_2dim_subspace(ax, sub_PCs[:, (2 * i):(2 * i + 2)], PCA_align[(2 * i):(2 * i + 2)],
                                  data_name, 'naive_PCA',
                                  data_dir='data/', to_save=True)

        # ----------------------------------------
        # step 4: Run EB-PCA
        # ----------------------------------------
        # run EB-PCA with joint estimation (by default)
        est_dir = 'results/%s/PC_estimates_iters_%i_n_rep_%i' % (data_name, iters, n_rep)
        if not os.path.exists(est_dir + '.npy'):
            _, V_joint = run_rankK_EBPCA('joint', X, real_data_rank[data_name], iters)
            np.save('results/%s/PC_estimates_iters_%i_n_rep_%i.npy' % (data_name, iters, n_rep), V_joint,
                    allow_pickle=False)
        else:
            V_joint = np.load(est_dir)

        # also run EB-PCA with marginal estimation if to_plot
        if to_plot:
            if not os.path.exists(est_dir + '_marginal.npy'):
                _, V_mar = run_rankK_EBPCA('mar', X, real_data_rank[data_name], iters)
                np.save(est_dir + '_marginal.npy', V_mar, allow_pickle=False)
            else:
                V_mar = np.load(est_dir + '_marginal.npy')

        # ----------------------------------------
        # step 5: Visualize estimated PC
        # ----------------------------------------
        if to_plot:
            # evaluate alignments
            mar_align = [get_alignment(V_mar[:, [j], -1], full_pcapack.V[:, [j]]) \
                         for j in range(real_data_rank[data_name])]
            joint_align = [get_alignment(V_joint[:, [j], -1], full_pcapack.V[:, [j]]) \
                           for j in range(real_data_rank[data_name])]

            # test sqrt(1-align^2)
            print('marginal alignments: ', np.sqrt(1 - np.power(mar_align, 2)))
            print('marginal alignments: ', np.sqrt(1 - np.power(joint_align, 2)))

            # align the direction and scale of sample PCs with the true PC
            V_mar_est = align_pc(V_mar[:, :, -1], full_pcapack.V)
            V_joint_est = align_pc(V_joint[:, :, -1], full_pcapack.V)

            # loop over plots
            V_est = [V_mar_est, V_joint_est]
            method_name = ['EB-PCA_marginal', 'EB-PCA_joint']
            aligns = [mar_align, joint_align]

            for i in range(2):
                for j in range(int(real_data_rank[data_name] / 2) - 1):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    vis_2dim_subspace(ax, V_est[i][:, (2 * j):(2 * j + 2)], aligns[i][(2 * i):(2 * i + 2)],
                                      data_name, method_name[i],
                                      data_dir='data/', to_save=True)

    end_time = time.time()
    print('Simulation takes %.2f s' % (end_time - start_time))