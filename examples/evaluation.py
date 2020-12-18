# provide quantitative assessment for EB-PCA on 1000G and UKBB

import sys
sys.path.extend(['../../generalAMP'])
from simulation.helpers import get_space_distance
import numpy as np

subset_sizes = {'1000G': [100, 1000, 10000],
                'Hapmap3': [1000, 5000, 10000]}

for data_name in ['1000G', 'Hapmap3']:
    V_star = np.load('results/%s/ground_truth_PC.npy' % data_name)
    d = V_star.shape[1]
    PCA_error = np.empty((50, 1+d))
    EBPCA_error = np.empty((50, 1+d))
    for subset_size in subset_sizes[data_name]:
        for n_copy in range(1, 51, 1):
            if subset_size > 500 and subset_size < 2500:
                PC_est = np.load('results/%s/PC_estimates_iters_5_n_copy_%s.npy' % (data_name, n_copy))
            else:
                PC_est = np.load('results/%s/PC_estimates_iters_5_size_%i_n_copy_%s.npy' % \
                                 (data_name, subset_size, n_copy))
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
        np.save('results/%s/PCA_error_summary_size_%i_n_rep_50.npy' % (data_name, subset_size), PCA_error)
        np.save('results/%s/EBPCA_error_summary_size_%i_n_rep_50.npy' % (data_name, subset_size), EBPCA_error)

float_formatter = "{:.2e}".format
def formatters(error):
    return [float_formatter(e) for e in error]

# summarize mean and std
for data_name in ['1000G', 'Hapmap3']:
    print('\n############')
    print('%s' % data_name)
    print('############\n')
    for subset_size in subset_sizes[data_name]:
        # print('\n %s \n' % subset_size)
        PCA_error = np.load('results/%s/PCA_error_summary_size_%i_n_rep_50.npy' % (data_name, subset_size))
        EBPCA_error = np.load('results/%s/EBPCA_error_summary_size_%i_n_rep_50.npy' % (data_name, subset_size))
        print('\n ############ %s ############ \n' % subset_size)
        print('from left to right: PC 1-%i, Joint' % (PCA_error.shape[1] - 1))
        print('PCA errors: ')
        print('\t mean ', formatters(np.mean(PCA_error, axis=0)))
        print('\t sd ', formatters(np.std(PCA_error, axis=0)))
        print('EBPCA errors:')
        print('\t mean ', formatters(np.mean(EBPCA_error, axis=0)))
        print('\t sd ', formatters(np.std(EBPCA_error, axis=0)))