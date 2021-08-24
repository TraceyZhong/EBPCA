# -
# Preprocess genotype data 
# to statisfy the EB-PCA assumption
# 
# Chang Su
# c.su@yale.edu
# Aug 23, 2021
# -

import os
import argparse
import numpy as np
from ebpca.preprocessing import normalize_obs
from ebpca.pca import get_pca
from ebpca.amp import ebamp_gaussian
from ebpca.misc import MeanFieldVB
from real_data_helpers import make_dir, load_data, make_PC_scatterplot, match_scale
import sys
sys.path.extend(['../'])
from tutorial import get_dist_of_subspaces, redirect_pc

# -
# Step 0: set parameters
# -

# Take arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, help="which genotype data to process",
                    default='1000G', const='1000G', nargs='?')
parser.add_argument("--method", type=str, help="run either EB-PCA or Mean-field VB",
                    default='EB-PCA', const='EB-PCA', nargs='?')
args = parser.parse_args()
data_name = args.data_name
method = args.method

# Make directories to save results
make_dir(data_name)
res_prefix = 'results/%s' % data_name
fig_prefix = 'figures/%s' % data_name

# Parameters used in experiments
# Subset size (#SNPs) used in manuscript
subset_size_list = {'1000G': 1000, 'Hapmap3': 5000, 'PBMC': 13711}

# Rank (#PCs)
# Rank equals the number of PCs in the EB-PCA model
# For each dataset, we manually inspect the singular value distribution
# and identify the number of signal components
real_data_rank = {'1000G': 4, 'PBMC': 3, 'Hapmap3': 4}

size = subset_size_list[data_name]
rank = real_data_rank[data_name]

# -
# Step 1: load data
# Genotypes: for illustration purpose, we load all processed data but only perform
# EB-PCA on a subset of SNPs
# Gene expression: we load all processed data and run EB-PCA on all processed data
# -

if data_name == '1000G' or data_name == 'Hapmap3':
	subset_path = '%s/subset.npy' % res_prefix
	V_star_path = '%s/V_star.npy' % res_prefix
	if not os.path.exists(subset_path):
		# Load data
		full_data = load_data(data_name, 100000)
		print('full data have #samples=%i, #features=%i' % (full_data.shape[0], full_data.shape[1]))
		# Compute 'ground truth' PCs using the full data
		full_data = normalize_obs(full_data, rank)
		full_pcapack = get_pca(full_data, rank)
		V_star = full_pcapack.V
		np.save(V_star_path, V_star)
		# Make a subset
		np.random.seed(7621)
		Y = full_data[np.random.choice([i for i in range(100000)], size, replace=False), :]
		# Normalize data such that the noise level satisfy our model assumption
		Y = normalize_obs(Y, rank)
		np.save(subset_path, Y)
	else:
		# Load subetted data
		Y = np.load(subset_path)
		# Load 'ground truth' PCs
		V_star = np.load(V_star_path)

	print('subsetted data have #samples=%i, #features=%i' % (Y.shape[0], Y.shape[1]))
elif data_name == 'PBMC':
	Y = load_data(data_name, None)
	# Normalize data such that the noise level satisfy our model assumption
	Y = normalize_obs(Y, rank)

# -
# Step 2: Run EB-PCA or Mean-field VB
# -

# Compute PCA pack
pcapack = get_pca(Y, rank)

# Run EB-PCA or Mean-field VB
if method == 'EB-PCA':
	_, V = ebamp_gaussian(pcapack, amp_iters = 5)
elif method == 'Mean-field VB':
	# Both ebpca_scaling and start_from_f (right PC) are set to true
	# for mean-field VB to be as similar to EB-PCA as possible
	_, V = MeanFieldVB(pcapack, iters = 5, ebpca_scaling=True, start_from_f=True)

# Save estimates
np.save('%s/%s_V_est.npy' % (res_prefix, method), V)

# -
# Step 3: Evaluate estimates (with errors and plots)
# -

# Evaluate estimation error for subsetting experiments
# joint estimation error
est_error = [get_dist_of_subspaces(V[:, :, i], V_star) for i in range(5 + 1)]
if data_name == '1000G' or data_name == 'Hapmap3':
	print('Estimation errors of %s on %s (%i SNPs) across iterations:' %
		  (method, data_name, size))
print(est_error)
# marginal error, for each PC:
error = [get_dist_of_subspaces(V[:, [j], -1], V_star[:, [j]]) for j in range(rank)]

# Visualize PCs in a scatter plot
if data_name == '1000G' or data_name == 'Hapmap3':
	if not os.path.exists('%s/Ground_truth_PCs_%s_PC_1_2.png' % (fig_prefix, data_name)):
		# visualize 'ground truth' PCs
		make_PC_scatterplot(V_star, 'Ground truth PCs', data_name, [0 for i in range(4)])
		# sample PCs
		sample_PC = redirect_pc(V[:, :, 0], V_star)
		sample_error = [get_dist_of_subspaces(sample_PC[:, [j]], V_star[:, [j]]) for j in range(rank)]
		make_PC_scatterplot(sample_PC, 'Sample PCs (%s SNPs)' % size, data_name, sample_error)
	# denoised PCs
	est_PC = redirect_pc(V[:, :, -1], V_star)
	if method == 'Mean-field VB':
		# Mean-field VB estimates are potentially at a different scale compared to sample PCs
		# apply this function to match the scaling and enable uniform plots
		est_PC = match_scale(est_PC, V_star)
	make_PC_scatterplot(est_PC, method, data_name, error)
elif data_name == 'PBMC':
	make_PC_scatterplot(V[:, :, 0], 'Sample PCs', data_name)
	make_PC_scatterplot(V[:, :, -1], method, data_name)