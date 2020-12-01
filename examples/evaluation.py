# provide quantitative assessment for EB-PCA on 1000G and UKBB

import sys
sys.path.extend(['../../generalAMP'])
from simulation.helpers import get_space_distance
import numpy as np

for data_name in ['UKBB', '1000G']:
    V_star = np.load('results/%s/ground_truth_PC.npy' % data_name)
    d = V_star.shape[1]
    PCA_error = np.empty((50, 1+d))
    EBPCA_error = np.empty((50, 1+d))
    for n_copy in range(1, 51, 1):
        PC_est = np.load('results/%s/PC_estimates_iters_5_n_copy_%s.npy' % (data_name, n_copy))
        PCA = PC_est[:, :, 0]
        EBPCA = PC_est[:, :, 5]
        # PCA_error
        PCA_error_tmp = [get_space_distance(PCA[:, [j]], V_star[:, [j]]) for j in range(d)]
        PCA_error_tmp.append(get_space_distance(PCA, V_star))
        PCA_error[n_copy - 1, :] = PCA_error_tmp
        # EBPCA error
        EBPCA_error_tmp = [get_space_distance(EBPCA[:, [j]], V_star[:, [j]]) for j in range(d)]
        EBPCA_error_tmp.append(get_space_distance(EBPCA, V_star))
        EBPCA_error[n_copy - 1, :] = EBPCA_error_tmp
    np.save('results/%s/PCA_error_summary_n_rep_50.npy' % data_name, PCA_error)
    np.save('results/%s/EBPCA_error_summary_n_rep_50.npy' % data_name, EBPCA_error)

# summarize mean and std
for data_name in ['UKBB', '1000G']:
    print('\n\n %s \n\n' % data_name)
    PCA_error = np.load('results/%s/PCA_error_summary_n_rep_50.npy' % data_name)
    EBPCA_error = np.load('results/%s/EBPCA_error_summary_n_rep_50.npy' % data_name)
    print('\n ############ %s ############ \n' % data_name)
    print('from left to right: PC 1-%i, Joint' % (PCA_error.shape[1] - 1))
    print('PCA errors: ')
    print('\t mean ', np.mean(PCA_error, axis=0))
    print('\t sd ', np.std(PCA_error, axis=0))
    print('EBPCA errors:')
    print('\t mean ', np.mean(EBPCA_error, axis=0))
    print('\t sd ', np.std(EBPCA_error, axis=0))