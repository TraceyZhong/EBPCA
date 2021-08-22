# summarize estimation errors for EB-PCA and MF-VB
# across iterations on 1000G

import sys
sys.path.extend(['../../generalAMP'])
from simulation.helpers import get_space_distance
import numpy as np
import os

subset_sizes = [100, 1000, 10000]
data_name = '1000G'

os.chdir('../examples')

V_star = np.load('results/%s/ground_truth_PC.npy' % data_name)
d = V_star.shape[1]

# EB-PCA
ebpca_error_file = 'results/%s/EBPCA_error_across_iterations_summary_size_%i_n_rep_50.npy' % (data_name, 100)
if not os.path.exists(ebpca_error_file):
    for subset_size in subset_sizes:
        errors_by_iters = []
        for n_copy in range(1, 51, 1):
            PC_est = np.load('results/%s/PC_estimates_iters_5_size_%i_n_copy_%i.npy' % \
                             (data_name, subset_size, n_copy))
            # joint estimation errors across iterations
            errors_by_iters.append([get_space_distance(PC_est[:, :, i], V_star) for i in range(6)])
            np.save(ebpca_error_file, errors_by_iters)

# MF-VB
for RMT_ini in [True, False]:
    error_path = 'results/%s/MF-VB_RMT_%s_error_summary_size_%i_n_rep_50.npy' % (data_name, RMT_ini, 100)
    if not os.path.exists(error_path):
        for subset_size in subset_sizes:
            error = np.empty((50, 1 + d))
            errors_by_iters = []
            for n_copy in range(1, 51, 1):
                # if subset_size > 500 and subset_size < 2500:
                #     PC_est = np.load('results/%s/PC_estimates_iters_5_n_copy_%s.npy' % (data_name, n_copy))
                # else:
                PC_est = np.load('results/%s/PC_estimates_iters_5_size_%i_n_copy_%s_MF-VB_RMT_%s.npy' % \
                                 (data_name, subset_size, n_copy, RMT_ini))
                # joint estimation errors across iterations
                errors_by_iters.append([get_space_distance(PC_est[:, :, i], V_star) for i in range(6)])
                np.save('results/%s/MF-VB_RMT_%s_error_across_iterations_summary_size_%i_n_rep_50.npy' % (
                data_name, RMT_ini, subset_size), errors_by_iters)
                # estimation errors by PCs at the last iteration
                MFVB_est = PC_est[:, :, -1]
                # MF-VB error
                error_tmp = [get_space_distance(MFVB_est[:, [j]], V_star[:, [j]]) for j in range(d)]
                error_tmp.append(get_space_distance(MFVB_est, V_star))
                error[n_copy - 1, :] = error_tmp
            np.save(error_path, error)

float_formatter = "{:.8f}".format # "{:.2e}".format
def formatters(error):
    return [float_formatter(e) for e in error]

# summarize mean and std
for RMT_ini in [True, False]:
    print('\n############')
    print('%s' % RMT_ini)
    print('############\n')
    for subset_size in subset_sizes:
        PCA_error = np.load('results/%s/PCA_error_summary_size_%i_n_rep_50.npy' % (data_name, subset_size))
        EBPCA_error = np.load('results/%s/EBPCA_error_summary_size_%i_n_rep_50.npy' % (data_name, subset_size))
        MFVB_error = np.load('results/%s/MF-VB_RMT_%s_error_summary_size_%i_n_rep_50.npy' % (data_name, RMT_ini, subset_size))
        print('\n ############ %s ############ \n' % subset_size)
        print('from left to right: PC 1-%i, Joint' % (PCA_error.shape[1] - 1))
        print('PCA errors: ')
        print('\t mean ', formatters(np.mean(PCA_error, axis=0)))
        print('\t sd ', formatters(np.std(PCA_error, axis=0)))
        print('EBPCA errors:')
        print('\t mean ', formatters(np.mean(EBPCA_error, axis=0)))
        print('\t sd ', formatters(np.std(EBPCA_error, axis=0)))
        print('MF-VB errors:')
        print('\t mean ', formatters(np.mean(MFVB_error, axis=0)))
        print('\t sd ', formatters(np.std(MFVB_error, axis=0)))
