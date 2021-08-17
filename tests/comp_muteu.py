import numpy as np
from helper_boxplots import plot_boxplot_series
import matplotlib.pyplot as plt

output_dir = '.'
data_name = '1000G'
subset_size = 100
n_copy_total = 50
muteu = True

# muteu=True errors
error_file = '%s/result/iterates/%s_subset_size_%i_update_prior_errors_n_copy_%i_muteu_%s.npy' % \
             (output_dir, data_name, subset_size, n_copy_total, muteu)
tmp = np.load(error_file)
# subset only the one with iteratively updating priors
muteu_True_errors = [tmp[i][1] for i in range(len(tmp))]

# muteu=False errors
data_dir = '../examples'
muteu_False_errors = np.load('%s/results/%s/EBPCA_error_across_iterations_summary_size_%i_n_rep_50.npy' %
                             (data_dir, data_name, subset_size))

for i in range(6):
    print(i)
    iter_error_comp = [muteu_True_errors[j][i] - muteu_False_errors[j][i] for j in range(n_copy_total)]
    if i == 5:
        print([muteu_True_errors[j][i] for j in range(10)])
        print([muteu_False_errors[j][i] for j in range(10)])
        print(iter_error_comp)
    print(np.mean(iter_error_comp))
    print(np.std(iter_error_comp))

tmp = [muteu_True_errors, muteu_False_errors]
boxplot_stats = []
iters = 5
for i in range(2):
    tmp2 = [[tmp[i][j][k] for j in range(n_copy_total)] for k in range(iters+1)]
    boxplot_stats.append(tmp2)

plot_boxplot_series(boxplot_stats, None, labels = ['muteu=True', 'muteu=False'])
plt.savefig('figures/muteu/muteu_comp_%i_SNPs_%i_copies.png' % (subset_size, n_copy_total))