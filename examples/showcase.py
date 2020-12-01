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
from simulation.helpers import get_error
from simulation.rank_two import run_rankK_EBPCA
from tutorial import get_alignment, redirect_pc
import time
from visualization import vis_2dim_subspace

# load cleaned full data
def load_data(data_name):
    data_dir = 'data/' + data_name
    if data_name in ['1000G', 'UKBB', 'PBMC', 'GTEx']:
        if data_name == '1000G':
            full_data = np.load(data_dir + '/normalized_1000G_1e5.npy')
        elif data_name == 'UKBB':
            full_data = np.load(data_dir + '/normalized_UKBB_1e5.npy')
        elif data_name == 'PBMC':
            full_data = np.load(data_dir + '/pbmc_norm_clean.npy')
        elif data_name == 'GTEx':
            full_data = np.load(data_dir + '/GTEx_full_norm.npy')
    else:
        print('There are only 4 datasets supported: 1000G, UKBB, PBMC, GTEx')
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
    X_sub = X_full_norm[np.random.choice([i for i in range(m)], n_sub, replace=False), :]
    return X_sub

def prep_subsets(X_norm, n_sub, data_name, seeds, n_rep=50):
    for i in range(n_rep):
        X = generate_subset(X_norm, n_sub, seed=seeds[i])
        # normalize data to satisfy the EB-PCA assumption
        X = normalize_obs(X, real_data_rank[data_name])
        np.save('results/%s/subset_n_copy_%i.npy' % (data_name, i + 1), X)

def eval_align_stats(data_name, method, s_star, ind=-1):
    align_dir = 'results/%s' % data_name
    for i in range(50):
        if i == 0:
            aligns = np.load('%s/joint_alignment_n_copy_%i.npy' % (align_dir, i+1))
        else:
            sec = np.load('%s/joint_alignment_n_copy_%i.npy' % (align_dir, i+1))
            aligns = np.vstack([aligns, sec])

    print('Print alignment statistics for sample PCA:')

    print('\n Print alignment statistics for EB-PCA: \n')
    print('\t mean:', np.mean(aligns, axis=0))
    print('\t mean:', np.std(aligns, axis=0))

if __name__ == '__main__':
    # set parameters for different datasets

    # ---Rank---
    # Rank equals the number of PCs in the EB-PCA model
    # For each dataset, we manually inspect the singular value distribution
    # and identify the number of signal components
    real_data_rank = {'1000G': 4, 'UKBB': 2, 'PBMC': 3, 'GTEx': 2}

    # ---Subset size---
    # The size of the random subsets
    subset_size = {'1000G': 1000, 'UKBB': 2000, 'PBMC': 13711, 'GTEx': 2000}

    # ---iterations---
    iters_list = {'1000G': 5, 'UKBB': 5, 'PBMC': 5, 'GTEx': 5}

    # ---plot: x range---
    xRange_list = {'1000G': [[-0.045, 0.03], [-0.055, 0.07]], 'UKBB': [[-0.03, 0.13], []],
                   'PBMC': [[-0.06, 0.025], [-0.04, 0.10]], 'GTEx': [-0.13, 0.08]}

    # ---plot: y range---
    yRange_list = {'1000G': [[-0.045, 0.045], [-0.045, 0.11]], 'UKBB': [[-0.15, 0.055], []],
                   'PBMC': [[-0.045, 0.105], [-0.05, 0.12]], 'GTEx': [-0.1, 0.05]}

    # ---legend position---
    legend_pos = {'1000G': ['upper left', 'upper left'], 'UKBB': ['lower right', []],
                  'PBMC': ['upper left', 'upper left'], 'GTEx': 'upper left'}

    # ---optmizer-----
    optimizer = {'1000G': 'Mosek', 'UKBB': 'Mosek', 'PBMC': 'EM', 'GTEx': 'Mosek'}

    # ---full data PC name----
    fullPC = {'1000G': 'Ground truth PC', 'UKBB': 'Ground truth PC', 'PBMC': 'Sample PC'}

    # ---singular value dist lim----
    sv_lim = {'1000G': [0,2], 'UKBB': [0, 2.5], 'PBMC': [0,2]}

    # ---pcs----
    pcs = {'1000G': [[0,1], [2,3]], 'UKBB': [[0,1],[]], 'PBMC': [[0,1],[1,2]]}

    # take arguments from command line
    # run single example for visualization or multiple replications to demonstrate quantitative performance
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="which dataset to show",
                        default='PBMC', const='PBMC', nargs='?')
    parser.add_argument("--n_copy", type=int, help="which replication to run",
                        default=1, const=1, nargs='?')
    parser.add_argument("--to_plot", type=str, help="whether or not to plot",
                        default='yes', const='yes', nargs='?')
    args = parser.parse_args()
    data_name = args.data_name
    n_copy = args.n_copy
    # only make plots when replication=1
    to_plot = False
    if args.to_plot == 'yes':
        to_plot = (n_copy == 1)

    print('Analyzing dataset: %s, #replications = %i' % (data_name, n_copy))
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

    # save true PC
    if not os.path.exists('results/%s/ground_truth_PC.npy' % data_name):
        print('Load normalized data.')
        norm_data = np.load('results/%s/norm_data.npy' % data_name)
        full_pcapack = get_pca(norm_data, real_data_rank[data_name])
        V_star = full_pcapack.V
        np.save('results/%s/ground_truth_PC.npy' % data_name, full_pcapack.V)
    else:
        V_star = np.load('results/%s/ground_truth_PC.npy' % data_name)
        n, _ = V_star.shape

    if to_plot:
        # print('Load normalized data.')
        norm_data = np.load('results/%s/norm_data.npy' % data_name)
        full_pcapack = get_pca(norm_data, real_data_rank[data_name])
        # visualize spectrum of the residual matrix (noise)
        if not os.path.exists('figures/%s/singvals_dist_%s.png' % (data_name, data_name)):
            check_residual_spectrum(full_pcapack, xmin = sv_lim[data_name][0], xmax = sv_lim[data_name][1],
                                    to_save=True, fig_prefix=data_name, label=data_name)

        # visualize distributions of singular values and PCs
        if not os.path.exists('figures/%s/PC_0_%s.png' % (data_name, data_name)):
            # explore eigenvalue and eigenvector distributions
            plot_pc(full_pcapack.X, label=data_name, nPCs=real_data_rank[data_name],
                    to_show=False, to_save=True, fig_prefix='%s/' % data_name)

        # visualize the joint structure of PCs
        for i in range(2):
            xRange = [l * np.sqrt(n) for l in xRange_list[data_name][i]]
            yRange = [l * np.sqrt(n) for l in yRange_list[data_name][i]]
            pc1 = pcs[data_name][i][0]
            pc2 = pcs[data_name][i][1]
            if i == 0:
                add_legend = True
            else:
                add_legend = False
            vis_2dim_subspace(V_star[:, pc1 : (pc2+1)] * np.sqrt(n), [1,1], data_name, fullPC[data_name],
                              xRange=xRange, yRange=yRange,
                              data_dir='data/', to_save=True,
                              PC1=pc1 + 1, PC2=pc2 + 1,
                              legend_loc=legend_pos[data_name][i], plot_legend=add_legend)

    # -------------------------------------------
    # step 3: Get random subset(s) from full data
    # -------------------------------------------
    # By subsetting we get noisy observations of the ground truth PCs,
    # and show EB-PCA successfully recovers the ground truth,
    # both quantitatively (alignments) and qualitatively (visualization of PC)

    if data_name == 'PBMC':
        # use the full PBMC data as an example of applying EB-PCA
        X = norm_data
    else:
        # set seed
        np.random.seed(1)
        seeds = np.random.randint(0, 10000, 50)  # set seed for each dataset

        # make subsets
        if not os.path.exists('results/%s/subset_n_copy_%i.npy' % (data_name, 50)):
            print('Load normalized data.')
            norm_data = np.load('results/%s/norm_data.npy' % data_name)
            # print('Making 50 random subsets')
            # prep_subsets(norm_data, subset_size[data_name], data_name, seeds, n_rep=50)
            # remove normalized data
            del norm_data

        # load generated subsets
        X = np.load('results/%s/subset_n_copy_%i.npy' % (data_name, n_copy))

        # get pca estimates
        sub_pcapack = get_pca(X, real_data_rank[data_name])

        # visualize naive PCA
        if to_plot:
            # evaluate alignment
            PCA_align = [get_alignment(sub_pcapack.V[:, [j]], V_star[:, [j]]) \
                         for j in range(real_data_rank[data_name])]
            # align the direction and scale of sample PCs with the true PC
            sub_PCs = sub_pcapack.V
            sub_PCs = redirect_pc(sub_PCs, V_star)

            # loop over pairs of PCs
            for i in range(2):
                xRange = [l * np.sqrt(n) for l in xRange_list[data_name][i]]
                yRange = [l * np.sqrt(n) for l in yRange_list[data_name][i]]
                pc1 = pcs[data_name][i][0]
                pc2 = pcs[data_name][i][1]
                print(pcs[data_name][i])
                ax = vis_2dim_subspace(sub_PCs[:, pc1:(pc2+1)] * np.sqrt(n), PCA_align[pc1:(pc2+1)],
                                       data_name, 'Sample PC (%i subsample)' % subset_size[data_name], xRange=xRange, yRange=yRange,
                                       data_dir='data/', to_save=True, legend_loc=legend_pos[data_name][i],
                                       PC1=pc1+1, PC2=pc2[1]+1)

    # ----------------------------------------
    # step 4: Run EB-PCA
    # ----------------------------------------
    est_dir = 'results/%s/PC_estimates_iters_%i_n_copy_%i' % (data_name, iters_list[data_name], n_copy)

    if not os.path.exists(est_dir + '.npy'):
        # run EB-PCA with joint estimation (by default)
        _, V_joint, _ = run_rankK_EBPCA('joint', X, real_data_rank[data_name], iters_list[data_name],
                                        optimizer = optimizer[data_name])
        np.save(est_dir + '.npy', V_joint, allow_pickle=False)

        # evaluate alignments
        joint_align = [get_alignment(V_joint[:, [j], -1], V_star[:, [j]]) \
                       for j in range(real_data_rank[data_name])]
        np.save('results/%s/joint_alignment_n_copy_%i.npy' % (data_name, n_copy),
                joint_align, allow_pickle=False)

    # also run EB-PCA with marginal estimation for visualization purpose
    if to_plot:
        # load joint alignment
        V_joint = np.load(est_dir + '.npy')
        joint_align = np.load('results/%s/joint_alignment_n_copy_%i.npy' % (data_name, n_copy),
                              allow_pickle=False)

        # if not os.path.exists(est_dir + '_marginal.npy'):
        #     _, V_mar, _ = run_rankK_EBPCA('marginal', X, real_data_rank[data_name], iters_list[data_name])
        #     np.save(est_dir + '_marginal.npy', V_mar, allow_pickle=False)
        # else:
        #     V_mar = np.load(est_dir + '_marginal.npy')

    # ----------------------------------------
    # step 5: Visualize estimated PC
    # ----------------------------------------
    if to_plot:
        # evaluate alignments
        # mar_align = [get_alignment(V_mar[:, [j], -1], V_star[:, [j]]) \
        #              for j in range(real_data_rank[data_name])]
        joint_align = [get_alignment(V_joint[:, [j], -1], V_star[:, [j]]) \
                       for j in range(real_data_rank[data_name])]

        # test sqrt(1-align^2)
        # print('marginal sqrt(1-alignments): ', np.sqrt(1 - np.power(mar_align, 2)))
        print('marginal sqrt(1-alignments): ', [get_error(a) for a in joint_align])

        # align the direction and scale of sample PCs with the true PC
        # V_mar_est = align_pc(V_mar[:, :, -1], V_star)
        V_joint_est = redirect_pc(V_joint[:, :, -1], V_star)

        # loop over plots
        # V_est = [V_mar_est, V_joint_est]
        # method_name = ['EB-PCA_marginal', 'EB-PCA_joint']
        # aligns = [mar_align, joint_align]

        # loop over methods
        # for i in range(2):

        # loop over sets of PCs
        for i in range(2):
            xRange = [l * np.sqrt(n) for l in xRange_list[data_name][i]]
            yRange = [l * np.sqrt(n) for l in yRange_list[data_name][i]]
            pc1 = pcs[data_name][i][0]
            pc2 = pcs[data_name][i][1]
            print(pcs[data_name][i])
            ax = vis_2dim_subspace(V_joint_est[:, pc1:(pc2+1)], joint_align[pc1:(pc2+1)],
                                   data_name, 'Joint EB-PCA',
                                   xRange=xRange, yRange=yRange,
                                   data_dir='data', to_save=True,
                                   PC1=pc1+1, PC2=pc2+1, legend_loc=legend_pos[data_name][i])

    end_time = time.time()
    print('Simulation takes %.2f s' % (end_time - start_time))
