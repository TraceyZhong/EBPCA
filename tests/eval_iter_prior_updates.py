import numpy as np
import os
import sys
sys.path.extend(['../../generalAMP'])
import argparse
from ebpca.pca import get_pca
from simulation.helpers import get_space_distance
import time
from ebpca.empbayes import NonparEB as NonparEB
from ebpca.amp import ebamp_gaussian as ebamp_gaussian

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, help="which dataset to show",
                    default='1000G', const='1000G', nargs='?')
parser.add_argument("--n_copy_total", type = int, help="number of total copies",
                    default=2, const=2, nargs="?")
parser.add_argument("--subset_size", type = int, help="subset size",
                    default=1000, const=1-00, nargs="?")
args = parser.parse_args()
data_name = args.data_name
n_copy_total = args.n_copy_total
subset_size = args.subset_size

# fixed parameters
optimizer = 'Mosek'
pca_method = 'EB-PCA'

# ---Rank---
# Rank equals the number of PCs in the EB-PCA model
# For each dataset, we manually inspect the singular value distribution
# and identify the number of signal components
real_data_rank = {'1000G': 4, 'UKBB': 2, 'PBMC': 3, 'GTEx': 2, 'Hapmap3': 4}

# ---Subset size---
# The size of the random subsets
subset_size_list = {'1000G': 1000, 'UKBB': 2000, 'PBMC': 13711, 'GTEx': 2000, 'Hapmap3': 5000}

# ---iterations---
iters_list = {'1000G': 5, 'UKBB': 5, 'PBMC': 5, 'GTEx': 5, 'Hapmap3': 5}

iters = iters_list[data_name]

print('\n\ndataset: %s, subset size=%i, iters=%i' % (data_name, subset_size, iters))

#data_dir = '/Users/chang/PycharmProjects/generalAMP/examples'
data_dir = '../examples'
norm_data = np.load('%s/results/%s/norm_data.npy' % (data_dir, data_name))

joint_error_list = []
time_list = []

if not os.path.exists('%s/results/%s/subset_size_%i_n_copy_%i.npy' % (data_dir, data_name, subset_size, n_copy_total)):
    print('Load normalized data.')
    norm_data = np.load('%s/results/%s/norm_data.npy' % (data_dir, data_name))
    # set seed
    np.random.seed(1)
    seeds = np.random.randint(0, 10000, 50)
    # generate random subsets
    prep_subsets(norm_data, subset_size, data_name, subset_size,
                 seeds[:n_copy_total], real_data_rank[data_name], n_rep=n_copy_total)


for n_copy in range(1, n_copy_total+1, 1):
    # load generated subsets
    X = np.load('%s/results/%s/subset_size_%i_n_copy_%i.npy' % (data_dir, data_name, subset_size, n_copy))

    # load PC 'truth'
    V_star = np.load('%s/results/%s/ground_truth_PC.npy' % (data_dir, data_name))

    # run EB-PCA with or without iteratively updating priors
    joint_error_by_mute = []
    time_by_mute = []
    for mute_prior_updates in [True, False]:
        # run EB-PCA with joint estimation (by default)
        # prepare the PCA pack
        pcapack = get_pca(X, real_data_rank[data_name])
        # initiate denoiser
        udenoiser = NonparEB(optimizer=optimizer, to_save=False)
        vdenoiser = NonparEB(optimizer=optimizer, to_save=False)
        # start timing
        start_time = time.time()
        _, V_joint, _ = ebamp_gaussian(pcapack, iters=iters,
                                       udenoiser=udenoiser, vdenoiser=vdenoiser,
                                       return_conv=True, mute_prior_updates=mute_prior_updates)
        end_time = time.time()
        time_by_mute.append(end_time - start_time)
        # evaluate error
        joint_error = [get_space_distance(V_joint[:, :, j], V_star)
                       for j in range(iters + 1)]
        joint_error_by_mute.append(joint_error)
    joint_error_list.append(joint_error_by_mute)
    time_list.append(time_by_mute)

np.save('result/iterates/%s_subset_size_%i_update_prior_errors.npy', joint_error_list, allow_pickle=False)
np.save('result/iterates/%s_subset_size_%i_update_prior_elapsed.npy', time_list, allow_pickle=False)

print(time_list)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)

x = [i for i in range(iters + 1)]
ax[0].plot(x, joint_error_list[-1][0], label='NPMLE: iter 0-5')
ax[0].plot(x, joint_error_list[-1][1], label='NPMLE: iter 0')
bins = np.linspace(0, 100, 5)
plt.hist([[time_list[i][0] for i in range(n_copy_total)],
          [time_list[i][1] for i in range(n_copy_total)]], bins,
         label=['NPMLE: iter 0-5', 'NPMLE: iter 0'])
plt.legend(loc='upper right')
plt.legend()
plt.suptitle('%s with %i SNPs' % (data_name, subset_size))
plt.title('time diff: %.2f' % (time_list[-1][1] - time_list[-1][0]))
plt.savefig('figures/iterates/tmp.png')
