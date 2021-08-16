'''
==================
Real data showcase
==================

This script includes the demonstration of EB-PCA method on four real datasets,
as well as the comparison with MF-VB on real datasets.
For each dataset, we can compare three sets of PCs:
    1. 'Ground truth' estimated from full data
    2. PCA estimates from a subset
    3. EB-PCA / MF-VB estimates with joint estimation of multivariate priors
by both PC scatter plots and quantitative subspace alignment measures.
The cleaning procedure for these datasets are included in ./data/.

'''

import numpy as np
import os
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.preprocessing import normalize_obs, plot_pc
from ebpca.pca import get_pca, check_residual_spectrum
from simulation.helpers import get_space_distance
from simulation.rank_two import run_rankK_EBPCA
from tutorial import redirect_pc
import time
from visualization import vis_2dim_subspace, load_sample_labels
import pandas as pd
from pandas_plink import read_plink1_bin

def read_plink_genotype(dir):
    G = read_plink1_bin(dir+'.bed', dir+'.bim', dir+'.fam', verbose=True)
    genotype = G.values
    return genotype

# load cleaned full data
def load_data(data_name):
    data_dir = 'data/' + data_name
    if data_name in ['1000G', 'UKBB', 'PBMC', 'GTEx', 'Hapmap3',
                     '1000G_African', '1000G_Caucasian',
                     '1000G_East-Asian', '1000G_Hispanic',
                     '1000G_South-Asian']:
        print(data_name)
        if '1000G' in data_name:
            if data_name == '1000G':
                full_data = np.load('data/1000G' + '/normalized_1000G_1e5.npy')
            else:
                full_data = read_plink_genotype('data/1000G/Merge')
            popu_label_df = pd.read_csv('data/1000G/Popu_labels.txt', sep=' ')
            for sub_popu in ['African', 'Caucasian', 'East-Asian', 'South-Asian', 'Hispanic']:
                if sub_popu in data_name:
                    full_data = full_data[(popu_label_df['Population_broad'] == sub_popu).values, :]
            if data_name == '1000G_African':
                # remove outliers in African population
                Af_outlier = np.load('data/1000G/subset_index.npy')
                full_data = full_data[Af_outlier, :]
        elif data_name == 'Hapmap3':
            full_data = np.load(data_dir + '/hapmap3.npy')
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

def prep_subsets(X_norm, n_sub, data_name, subset_size, seeds, rank, n_rep=50):
    for i in range(n_rep):
        X = generate_subset(X_norm, n_sub, seed=seeds[i])
        # normalize data to satisfy the EB-PCA assumption
        X = normalize_obs(X, rank)
        np.save('results/%s/subset_size_%i_n_copy_%i.npy' % (data_name, subset_size, i + 1), X)

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

# if the method is MF-VB, the estimates will be scaled
# provide this function to rescale MF-VB estimates and enable uniform plotting
def match_scale(U, U_star):
    n, k = U.shape
    for i in range(k):
        tmp = U[:, i]
        U[:, i] = tmp / np.sqrt(np.sum(tmp ** 2)) * np.sqrt(np.sum(U_star[:, i] ** 2)) * np.sqrt(n)
    return U

if __name__ == '__main__':
    # set parameters for different datasets

    # ---Rank---
    # Rank equals the number of PCs in the EB-PCA model
    # For each dataset, we manually inspect the singular value distribution
    # and identify the number of signal components
    real_data_rank = {'1000G': 4, 'UKBB': 2, 'PBMC': 3, 'GTEx': 2, 'Hapmap3': 4,
                      '1000G_African': 2, '1000G_Caucasian': 2,
                      '1000G_East-Asian': 2, '1000G_Hispanic': 2,
                      '1000G_South-Asian': 2 }

    # ---Subset size---
    # The size of the random subsets
    subset_size_list = {'1000G': 1000, 'UKBB': 2000, 'PBMC': 13711, 'GTEx': 2000, 'Hapmap3': 5000,
                   '1000G_African': 10000, '1000G_Caucasian': 5000,
                   '1000G_East-Asian': 5000, '1000G_Hispanic': 5000,
                   '1000G_South-Asian': 10000
                   }

    # ---iterations---
    iters_list = {'1000G': 5, 'UKBB': 5, 'PBMC': 5, 'GTEx': 5, 'Hapmap3': 5,
                  '1000G_African': 5, '1000G_Caucasian': 5,
                  '1000G_East-Asian': 5, '1000G_Hispanic': 5,
                  '1000G_South-Asian': 5}

    # ---plot: x range---
    xRange_list = {'1000G': [[-0.045, 0.025], [-0.055, 0.065]], 'UKBB': [[-0.03, 0.13], []],
                   'PBMC': [[-0.06, 0.025], [-0.04, 0.10]], 'GTEx': [-0.13, 0.08],
                   'Hapmap3': [[-0.05, 0.04], [-0.09, 0.05]],
                   '1000G_African': [[-0.12, 0.04], []], '1000G_Caucasian': [[-0.075, 0.126], []],
                   '1000G_East-Asian': [[-0.08, 0.10], []], '1000G_Hispanic': [[-0.135, 0.105], []],
                   '1000G_South-Asian': [[-0.08, 0.13], []]
                   }

    # ---plot: y range---
    yRange_list = {'1000G': [[-0.045, 0.04], [-0.045, 0.11]], 'UKBB': [[-0.15, 0.055], []],
                   'PBMC': [[-0.045, 0.105], [-0.05, 0.12]], 'GTEx': [-0.1, 0.05],
                   'Hapmap3': [[-0.045, 0.055], [-0.12, 0.05]],
                   '1000G_African': [[-0.09, 0.08], []], '1000G_Caucasian': [[-0.115, 0.12], []],
                   '1000G_East-Asian': [[-0.12, 0.12], []], '1000G_Hispanic': [[-0.38, 0.13], []],
                   '1000G_South-Asian': [[-0.11, 0.16], []]
                   }

    # ---legend position---
    legend_pos = {'1000G': ['upper left', 'upper left'], 'UKBB': ['lower right', []],
                  'PBMC': ['upper left', 'upper left'], 'GTEx': 'upper left',
                  'Hapmap3': ['upper left', 'upper left'],
                  '1000G_African': 'upper left', '1000G_Caucasian': 'upper left',
                  '1000G_East-Asian': 'upper left', '1000G_Hispanic': ['upper left'],
                  '1000G_South-Asian': 'upper left'}

    # ---optmizer-----
    # optimizer = {'1000G': 'Mosek', 'UKBB': 'Mosek', 'PBMC': 'EM', 'GTEx': 'Mosek'}

    # ---full data PC name----
    fullPC = {'1000G': 'Ground truth PCs', 'UKBB': 'Ground truth PCs', 'PBMC': 'Sample PCs',
              'Hapmap3': 'Ground truth PCs',
              '1000G_African': 'Ground truth PCs', '1000G_Caucasian': 'Ground truth PCs',
              '1000G_East-Asian': 'Ground truth PCs', '1000G_Hispanic': 'Ground truth PCs',
              '1000G_South-Asian': 'Ground truth PCs'}

    # ---singular value dist lim----
    sv_lim = {'1000G': [0.5,1.5], 'UKBB': [0.5,1.5], 'PBMC': [0.25,2],
              'Hapmap3': [0,2],
              '1000G_African': [0.5,1.5], '1000G_Caucasian': [0.5, 1.5],
              '1000G_East-Asian': [0.5, 1.5], '1000G_Hispanic': [0.5, 1.5],
              '1000G_South-Asian': [0.5, 1.5]}

    # ---pcs----
    pcs = {'1000G': [[0,1], [2,3]], 'UKBB': [[0,1],[]], 'PBMC': [[0,1],[1,2]],
           'Hapmap3': [[0,1], [2,3]],
           '1000G_African': [[0,1], []], '1000G_Caucasian': [[0,1], []],
           '1000G_East-Asian': [[0,1], []], '1000G_Hispanic': [[0,1], []],
           '1000G_South-Asian': [[0,1], []]}

    # ---npc---
    npc = {'1000G': 2, 'UKBB': 1, 'PBMC': 2, 'Hapmap3': 2,
           '1000G_African': 1, '1000G_Caucasian': 1,
           '1000G_East-Asian': 1, '1000G_Hispanic': 1,
           '1000G_South-Asian': 1}

    # ---sample name---
    sample_name = {'1000G': 'SNPs', 'UKBB': 'SNPs', 'PBMC': 'genes', 'Hapmap3': 'SNPs',
                   '1000G_African': 'SNPs', '1000G_Caucasian': 'SNPs',
                   '1000G_East-Asian': 'SNPs', '1000G_Hispanic': 'SNPs',
                   '1000G_South-Asian': 'SNPs'}
    # ---subpopu----
    sub_popu = {'1000G_African': 'LWK', '1000G': 'Caucasian', 'Hapmap3': 'Caucasian'}
    # take arguments from command line
    # run single example for visualization or multiple replications to demonstrate quantitative performance
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="which dataset to show",
                        default='PBMC', const='PBMC', nargs='?')
    parser.add_argument("--n_copy", type=int, help="which data subset to run EB-PCA / MF-VB on",
                        default=1, const=1, nargs='?')
    parser.add_argument("--to_plot", type=str, help="whether or not to plot",
                        default='no', const='no', nargs='?')
    parser.add_argument("--subset_size", type=int, help="subset size",
                        default=1000, const=1000, nargs='?')
    parser.add_argument("--optimizer", type=str, help="EM optimizer",
                        default='Mosek', const='Mosek', nargs='?')
    parser.add_argument("--pca_method", type=str, help="which pca method to use: EB-PCA or MF-VB",
                        default='EB-PCA', const='EB-PCA', nargs='?')
    parser.add_argument("--ebpca_ini", type=str, help="whether to initiate with RMT",
                        default='yes', const='yes', nargs='?')
    args = parser.parse_args()
    data_name = args.data_name
    n_copy = args.n_copy
    # only make plots when replication=1
    to_plot = False
    if args.to_plot == 'yes':
        to_plot = (n_copy == 1)
    optimizer = args.optimizer
    subset_size = args.subset_size
    # subset_size = subset_size_list[data_name]

    pca_method = args.pca_method
    ebpca_ini = (args.ebpca_ini == 'yes')
    print('Run %s method with ebpca_ini %s' % (pca_method, ebpca_ini))
    print('on dataset: %s, subset id = %i' % (data_name, n_copy))
    print('Parameters: subset size=%i, optimizer=%s' % (subset_size, optimizer))

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
        # e.g. population stratifcation, cell type identity
        # and the left PCs (U) are the samples from which we extract information to infer the left PCs
        full_data = np.transpose(full_data)
        print('full %s dataset have #samples=%i, #features=%i' % (data_name, full_data.shape[0], full_data.shape[1]))

        # normalize all samples to have the same variance
        # e.g. genetic variants / gene expressions to have the same variance
        # in practice we observe that this step can help with the interpr etation of PCs
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
        print('Load normalized data.')
        norm_data = np.load('results/%s/norm_data.npy' % data_name)
        full_pcapack = get_pca(norm_data, real_data_rank[data_name])
        # visualize spectrum of the residual matrix (noise)
        if not os.path.exists('figures/%s/singvals_dist_%s_full.png' % (data_name, data_name)):
            check_residual_spectrum(full_pcapack, xmin = sv_lim[data_name][0], xmax = sv_lim[data_name][1],
                                    to_save=True, fig_prefix=data_name, label=data_name + '_full')

        # visualize distributions of singular values and PCs
        if not os.path.exists('figures/%s/PC_0_%s_full.png' % (data_name, data_name)):
            # explore eigenvalue and eigenvector distributions
            print('Plot eigenvalue dist:')
            plot_pc(full_pcapack.X, label=data_name + '_full', nPCs=real_data_rank[data_name],
                    to_show=False, to_save=True, fig_prefix='%s/' % data_name)
        # visualize the joint structure of PCs
        if not os.path.exists('figures/%s/Ground_truth_PCs_1000G_PC_3_4' % data_name):
            for i in range(npc[data_name]):
                xRange = [l * np.sqrt(n) for l in xRange_list[data_name][i]]
                yRange = [l * np.sqrt(n) for l in yRange_list[data_name][i]]
                pc1 = pcs[data_name][i][0]
                pc2 = pcs[data_name][i][1]
                ax = vis_2dim_subspace(V_star[:, pc1: (pc2 + 1)] * np.sqrt(n), [1, 1], data_name, fullPC[data_name],
                                       xRange=xRange, yRange=yRange,
                                       data_dir='data/', to_save=True,
                                       PC1=pc1 + 1, PC2=pc2 + 1,
                                       legend_loc=legend_pos[data_name][i])

    # -------------------------------------------
    # step 3: Get random subset(s) from full data
    # -------------------------------------------
    # By subsetting we get noisy observations of the ground truth PCs,
    # and show EB-PCA successfully recovers the ground truth,
    # both quantitatively (alignments / subspace distance) and qualitatively (visualization of PC)

    if data_name == 'PBMC':
        # use the full PBMC data as an example of applying EB-PCA
        X = norm_data
    else:
        # set seed
        np.random.seed(1)
        seeds = np.random.randint(0, 10000, 50)  # set seed for each dataset

        # make subsets
        if not os.path.exists('results/%s/subset_size_%i_n_copy_%i.npy' % (data_name, subset_size, n_copy)):#50
            print('Load normalized data.')
            norm_data = np.load('results/%s/norm_data.npy' % data_name)
            if not to_plot:
                print('Making 50 random subsets')
                prep_subsets(norm_data, subset_size, data_name,
                             seeds, real_data_rank[data_name], n_rep=50)
            else:
                print('Making 1 random subset')
                prep_subsets(norm_data, subset_size, data_name,
                             [seeds[0]], real_data_rank[data_name], n_rep=1)
            # remove normalized data
            # del norm_data

        # load one of the generated subsets
        # with id given by n_copy
        X = np.load('results/%s/subset_size_%i_n_copy_%i.npy' % (data_name, subset_size, n_copy))

        # get pca estimates
        sub_pcapack = get_pca(X, real_data_rank[data_name])

        # codes for sub-population analyses for 1000G
        # popu_label = load_sample_labels(data_name)
        # purple_cluster = popu_label == sub_popu[data_name]
        # print('========================')
        # print('\t full\t')
        # print('theo var ', (1 - full_pcapack.feature_aligns ** 2))
        # print('emp var ', [np.var(full_pcapack.V[purple_cluster, i]) * V_star.shape[0] \
        #                    for i in range(real_data_rank[data_name])])
        # print('========================\n')

        # print('========================')
        # print('\t subset\t')
        # print('theo var ', (1 - sub_pcapack.feature_aligns**2))
        # print('emp var ', [np.var(sub_pcapack.V[purple_cluster, i]) * V_star.shape[0] \
        #                    for i in range(real_data_rank[data_name])])
        # print('========================')

        # visualize naive PCA
        if to_plot:
            # evaluate error
            PCA_error = [get_space_distance(sub_pcapack.V[:, [j]], V_star[:, [j]])
                         for j in range(real_data_rank[data_name])]
            # align the direction f sample PCs with the true PC
            sub_PCs = sub_pcapack.V
            sub_PCs = redirect_pc(sub_PCs, V_star)

            if not os.path.exists('figures/%s/Sample_PCs_(%s_SNPs)_1000G_PC_1_2.png' % \
                                  (data_name, subset_size)):
                # loop over pairs of PCs
                for i in range(npc[data_name]):
                    xRange = [l * np.sqrt(n) for l in xRange_list[data_name][i]]
                    yRange = [l * np.sqrt(n) for l in yRange_list[data_name][i]]
                    pc1 = pcs[data_name][i][0]
                    pc2 = pcs[data_name][i][1]
                    ax = vis_2dim_subspace(sub_PCs[:, pc1:(pc2 + 1)] * np.sqrt(n), PCA_error[pc1:(pc2 + 1)],
                                           data_name, 'Sample PCs (%i %s)' % (subset_size, sample_name[data_name]),
                                           xRange=xRange, yRange=yRange,
                                           data_dir='data/', to_save=True, legend_loc=legend_pos[data_name][i],
                                           PC1=pc1 + 1, PC2=pc2 + 1)
            # visualize spectrum of the residual matrix (noise)
            if not os.path.exists('figures/%s/singvals_dist_%s.png' % (data_name, data_name)):
                check_residual_spectrum(sub_pcapack, xmin=sv_lim[data_name][0], xmax=sv_lim[data_name][1],
                                            to_save=True, fig_prefix=data_name, label=data_name)

            # visualize distributions of singular values and PCs
            if not os.path.exists('figures/%s/PC_0_%s.png' % (data_name, data_name)):
                # explore eigenvalue and eigenvector distributions
                print('Plot eigenvalue dist:')
                plot_pc(sub_pcapack.X, label=data_name, nPCs=real_data_rank[data_name],
                        to_show=False, to_save=True, fig_prefix='%s/' % data_name)

    # ----------------------------------------
    # step 4: Run EB-PCA / MF-VB
    # ----------------------------------------
    est_dir = 'results/%s/PC_estimates_iters_%i_size_%i_n_copy_%i' % (data_name, iters_list[data_name], subset_size, n_copy)

    if pca_method == 'MF-VB':
        method_name = '%s_RMT_%s' % (pca_method, ebpca_ini)
        res_path = est_dir + '_%s.npy' % method_name
    else:
        method_name = ''
        res_path = est_dir + '.npy'
    if not os.path.exists(res_path):
        # run EB-PCA / MF-VB with joint estimation (by default)
        _, V_joint, _ = run_rankK_EBPCA('joint', X, real_data_rank[data_name], iters_list[data_name],
                                        optimizer = optimizer, pca_method = pca_method,
                                        ebpca_ini=ebpca_ini)
        np.save(res_path, V_joint, allow_pickle=False)
        # evaluate error
        joint_error = [get_space_distance(V_joint[:, [j], -1], V_star[:, [j]])
                       for j in range(real_data_rank[data_name])]
    else:
        # load V estimates
        V_joint = np.load(res_path)

    # ----------------------------------------
    # step 5: Visualize estimated PC
    # ----------------------------------------
    if to_plot:
        # evaluate estimation error for joint estimation
        # by PC
        joint_error = [get_space_distance(V_joint[:, [j], -1], V_star[:, [j]])
                       for j in range(real_data_rank[data_name])]
        # by subspace
        joint_joint_error = get_space_distance(V_joint[:, :, -1], V_star)

        V_joint_est = redirect_pc(V_joint[:, :, -1], V_star)

        if pca_method == 'MF-VB':
            V_joint_est = match_scale(V_joint_est, V_star)
            print('Rescale the MF-VB estimates to match with PC scaling')

        # loop over sets of PCs
        for i in range(npc[data_name]):
            xRange = [l * np.sqrt(n) for l in xRange_list[data_name][i]]
            yRange = [l * np.sqrt(n) for l in yRange_list[data_name][i]]
            pc1 = pcs[data_name][i][0]
            pc2 = pcs[data_name][i][1]
            print(pcs[data_name][i])
            if pca_method == 'MF-VB' and ebpca_ini:
                method_name = '%s_RMT_%s (%i SNPs)' % (pca_method, ebpca_ini, subset_size)
            else:
                method_name = '%s (%i SNPs)' % (pca_method, subset_size)
            ax = vis_2dim_subspace(V_joint_est[:, pc1:(pc2+1)], joint_error[pc1:(pc2+1)],
                                   data_name, method_name,
                                   xRange=xRange, yRange=yRange,
                                   data_dir='data', to_save=True,
                                   PC1=pc1+1, PC2=pc2+1, legend_loc=legend_pos[data_name][i],
                                   joint_error = joint_joint_error)

    end_time = time.time()
    print('Simulation takes %.2f s' % (end_time - start_time))
    print([get_space_distance(V_joint[:, :, j], V_star) for j in range(iters_list[data_name]+1)])
