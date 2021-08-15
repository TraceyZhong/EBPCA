# benchmark the elapsed time and errors
# of EB-PCA with and without iteratively updating priors

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
                    default=1000, const=1000, nargs="?")
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

data_dir = '/Users/chang/PycharmProjects/generalAMP/examples'
output_dir = '/Users/chang/PycharmProjects/generalAMP/tests'
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

error_file = '%s/result/iterates/%s_subset_size_%i_update_prior_errors_n_copy_%i.npy' % \
             (output_dir, data_name, subset_size, n_copy_total)
if not os.path.exists(error_file):
    for n_copy in range(1, n_copy_total + 1, 1):
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
                                           muteu=True,
                                           return_conv=True, mute_prior_updates=mute_prior_updates)
            end_time = time.time()
            time_by_mute.append(end_time - start_time)
            # evaluate error
            joint_error = [get_space_distance(V_joint[:, :, j], V_star)
                           for j in range(iters + 1)]
            joint_error_by_mute.append(joint_error)
        joint_error_list.append(joint_error_by_mute)
        time_list.append(time_by_mute)

    np.save(
        error_file,
        joint_error_list, allow_pickle=False)
    np.save(
        'result/iterates/%s_subset_size_%i_update_prior_elapsed_n_copy_%i.npy' % (data_name, subset_size, n_copy_total),
        time_list, allow_pickle=False)

else:
    joint_error_list = np.load(error_file)
    time_list = np.load('%s/result/iterates/%s_subset_size_%i_update_prior_elapsed_n_copy_%i.npy' %
                        (output_dir, data_name, subset_size, n_copy_total))

print(time_list)

import matplotlib.pyplot as plt

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

update_on_errors = [[joint_error_list[j][0][i] for j in range(n_copy_total)] for i in range(iters+1)]
update_mute_errors = [[joint_error_list[j][0][i] for j in range(n_copy_total)] for i in range(iters+1)]
colors = ['tab:red', 'tab:grey']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4.3), constrained_layout=True)
bp1 = ax[0].boxplot(update_on_errors, positions=np.array([i - 0.15 for i in range(1, iters+2, 1)]),
              widths=0.2)
bp2 = ax[0].boxplot(update_mute_errors, positions=np.array([i + 0.15 for i in range(1, iters+2, 1)]),
              widths=0.2)

set_box_color(bp1, colors[0])
set_box_color(bp2, colors[1])

ax[0].plot([i - 0.15 for i in range(1, iters+2, 1)],
           [np.median(err) for err in update_on_errors], '-', alpha=0.5,  c=colors[0], label='iter NPMLE off')
ax[0].plot([i + 0.15 for i in range(1, iters+2, 1)],
           [np.median(err) for err in update_on_errors], '-', alpha=0.5, c=colors[1], label='iter NPMLE on')
ax[0].legend()
ax[0].set_xticks([i for i in range(1, iters+2, 1)])
ax[0].set_xticklabels([i for i in range(1, iters+2, 1)])
ax[0].set_title('Estimation error (subspace distance)')

bins = np.linspace(0, 35, 1)
bp1 = ax[1].boxplot([time_list[i][0] for i in range(n_copy_total)], positions=[1])
bp2 = ax[1].boxplot([time_list[i][1] for i in range(n_copy_total)], positions=[2])
ax[1].set_xticks([1,2])
ax[1].set_xticklabels(['iter NPMLE off', 'iter NPMLE on'])
ax[1].set_title('Elapsed time (seconds)')
set_box_color(bp1, colors[0])
set_box_color(bp2, colors[1])
fig.suptitle('%s with %i SNPs, 20 experiments, 5 iterations' % (data_name, subset_size),
             size = 'x-large')
#plt.title('time diff: %.2f' % (time_list[-1][1] - time_list[-1][0]))
plt.savefig('figures/iterates/%s_subset_size_%i_n_copy_%i.png' %
            (data_name, subset_size, n_copy_total))