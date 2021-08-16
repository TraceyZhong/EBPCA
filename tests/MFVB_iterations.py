# compare MF-VB and EB-PCA of their estimation erros across iterations
# by series of boxplots

import numpy as np
import os
import sys
sys.path.extend(['../../generalAMP'])
import argparse
from simulation.helpers import get_space_distance

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, help="which dataset to show",
                    default='1000G', const='1000G', nargs='?')
parser.add_argument("--subset_size", type = int, help="subset size",
                    default=1000, const=1000, nargs="?")
args = parser.parse_args()
data_name = args.data_name
subset_size = args.subset_size

# fixed parameters
optimizer = 'Mosek'

iters = 5

print('\n\ndataset: %s, subset size=%i, iters=%i' % (data_name, subset_size, iters))

data_dir = '/Users/chang/PycharmProjects/generalAMP/examples'
output_dir = '/Users/chang/PycharmProjects/generalAMP/tests'
norm_data = np.load('%s/results/%s/norm_data.npy' % (data_dir, data_name))

ebpca = np.load('%s/results/%s/PC_estimates_iters_%i_size_%i_n_copy_1_EB-PCA.npy' %
                (data_dir, data_name, iters, subset_size))
mfvb_RMT = np.load('%s/results/%s/PC_estimates_iters_%i_size_%i_n_copy_1_MF-VB_RMT_True.npy' %
                (data_dir, data_name, iters, subset_size))
mfvb = np.load('%s/results/%s/PC_estimates_iters_%i_size_%i_n_copy_1_MF-VB_RMT_False.npy' %
                (data_dir, data_name, iters, subset_size))

V_star = np.load('%s/results/%s/ground_truth_PC.npy' % (data_dir, data_name))

est = [ebpca, mfvb_RMT, mfvb]

def get_iter_error(V_joint, V_star):
    return [get_space_distance(V_joint[:, :, j], V_star)
     for j in range(iters + 1)]

est_error = [get_iter_error(es, V_star) for es in est]

import matplotlib.pyplot as plt
est_labels = ['EB-PCA', 'MF-VB (RMT init)', 'MF-VB']
for i in range(3):
    plt.plot(est_error[i], label = est_labels[i], alpha = 0.5)
plt.legend()
plt.title('%s, %i SNPs, % i iterations' % (data_name, subset_size, iters))
plt.ylabel('Estimation error (subspace distance)')
plt.savefig('figures/iterates/MFVB_error_iterates_%s_subset_size_%s.png' %
            (data_name, subset_size))