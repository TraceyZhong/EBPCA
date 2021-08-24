# -
# Make scatterplots of top PCs
# with points colored by population / cell type labels
#
# Chang Su
# c.su@yale.edu
# Aug 23, 2021
# -

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Functions for loading data
def load_data(data_name, size):
	if data_name in ['1000G', 'Hapmap3', 'PBMC']:
		if data_name == '1000G' or data_name == 'Hapmap3':
			full_data = np.load('makedata/Genotype/Processed/normalized_%s_%i.npy' % (data_name, size))
		elif data_name == 'PBMC':
			full_data = np.load('makedata/GeneExpression/pbmc_norm_clean.npy')
	else:
		print('There are only 3 datasets supported: 1000G, Hapmap3, PBMC')
	# transpose the data such that the data is #samples * #features
	# the right PCs (V) will be the vectors of interests
	# e.g. population stratifcation, cell type identity
	full_data = np.transpose(full_data)
	return full_data

# Make directories to save figures
def make_dir(data_name):
    if not os.path.exists('figures/{}'.format(data_name)):
        print('Creating directories for saving {} figures and results'.format(data_name))
        os.makedirs('results/{}'.format(data_name))
        os.makedirs('figures/{}'.format(data_name))
    else:
        print('Directories for {} already exist'.format(data_name))

# - Helper functions for loading labels
def load_sample_labels(data_name):
    if data_name in ['1000G', 'PBMC', 'Hapmap3']:
        if '1000G' == data_name:
            popu_label_df = pd.read_csv('makedata/Genotype/Processed/1000G_popu_labels.txt', sep=' ')
            popu_label = popu_label_df['Population_broad'].values
        elif data_name == 'Hapmap3':
            popu_label_df = np.load('makedata/Genotype/Processed/Hapmap3_popu.npy', allow_pickle=True)
            popu_label = popu_label_df[:, -1]
        elif data_name == 'PBMC':
            popu_label = np.load('makedata/GeneExpression/pbmc_celltype_clean.npy', allow_pickle=True)
    else:
        print('There are only 3 datasets supported: 1000G, Hapmap3, PBMC')
    return popu_label

def vis_2dim_subspace(u, errors, data_name, method_name, xRange, yRange,
                      to_save=True, **kwargs):
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['xtick.labelsize'] = 23
    plt.rcParams['ytick.labelsize'] = 23
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), constrained_layout=True)

    # get PC name
    PC1 = kwargs.get("PC1", 1)
    PC2 = kwargs.get("PC2", 2)
    # load sample label for coloring
    popu_label = load_sample_labels(data_name)
    df = pd.DataFrame(dict(x=list(u[:, 0]), y=list(u[:, 1]), label=popu_label))
    # make scatter plot
    groups = df.groupby('label')
    for name, group in groups:
        ax.scatter(group.x, group.y, marker='o', s=3, label=name) # linestyle='',
    ax.set_title('%s' % method_name)
    # add quantitative evaluation
    if method_name == 'Ground truth PCs' or data_name == 'PBMC':
        ax.set_xlabel('PC %s' % PC1)
        ax.set_ylabel('PC %s' % PC2)
    else:
        ax.set_xlabel('PC %i, error=%.2f' % (PC1, errors[0]))
        ax.set_ylabel('PC %i, error=%.2f' % (PC2, errors[1]))
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    if to_save:
        fig_prefix = kwargs.get('fig_prefix', 'figures/%s/' % data_name)
        plt.savefig(fig_prefix + '%s_%s_PC_%i_%i.png' % (method_name.replace(' ', '_'), data_name, PC1, PC2))
    return ax

def make_PC_scatterplot(V, title, data_name, error):
    # -
    # Parameters for plots
    # -
    # pairs of PCs to plot:
    pcs = {'1000G': [[0, 1], [2, 3]], 'PBMC': [[0, 1], [1, 2]], 'Hapmap3': [[0, 1], [2, 3]]}

    # x range
    xRange_list = {'1000G': [[-0.045, 0.025], [-0.055, 0.065]],
                   'PBMC': [[-0.06, 0.025], [-0.04, 0.10]],
                   'Hapmap3': [[-0.05, 0.04], [-0.09, 0.05]]}

    # y range
    yRange_list = {'1000G': [[-0.045, 0.04], [-0.045, 0.11]],
                   'PBMC': [[-0.045, 0.105], [-0.05, 0.12]],
                   'Hapmap3': [[-0.045, 0.055], [-0.12, 0.05]]}

    n = V.shape[0]
    if np.sum(V[:, 0]**2) < 1.01:
        V = V * np.sqrt(n)
    for i in range(2):
        xRange = [l * np.sqrt(n) for l in xRange_list[data_name][i]]
        yRange = [l * np.sqrt(n) for l in yRange_list[data_name][i]]
        pc1 = pcs[data_name][i][0]
        pc2 = pcs[data_name][i][1]
        ax = vis_2dim_subspace(V[:, pc1: (pc2 + 1)], error[pc1:(pc2+1)],
                               data_name, title,
                               xRange=xRange, yRange=yRange,
                               to_save=True, PC1=pc1 + 1, PC2=pc2 + 1,
                               legend_loc='upper left')

# if the method is MF-VB, the estimates will be at a different scale
# compared to EB-PCA
# provide this function to rescale MF-VB estimates and enable uniform plotting
def match_scale(U, U_star):
    n, k = U.shape
    for i in range(k):
        tmp = U[:, i]
        U[:, i] = tmp / np.sqrt(np.sum(tmp ** 2)) * np.sqrt(np.sum(U_star[:, i] ** 2)) * np.sqrt(n)
    return U