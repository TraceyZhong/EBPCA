# compare MF-VB and EB-PCA of their estimation erros across iterations
# by series of boxplots
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from tests.helper_boxplots import plot_boxplot_series
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, help="which dataset to show",
                    default='1000G', const='1000G', nargs='?')
parser.add_argument("--subset_size", type = int, help="subset size",
                    default=1000, const=1000, nargs="?")
args = parser.parse_args()
data_name = args.data_name
subset_size = args.subset_size

# fixed parameters
iters = 5

print('\n\ndataset: %s, subset size=%i, iters=%i' % (data_name, subset_size, iters))

data_dir = '/Users/chang/PycharmProjects/generalAMP/examples'
output_dir = '/Users/chang/PycharmProjects/generalAMP/tests'

ebpca = np.load('%s/results/%s/EBPCA_error_across_iterations_summary_size_%i_n_rep_50.npy' %
                (data_dir, data_name, subset_size))
mfvb_RMT = np.load('%s/results/%s/MF-VB_RMT_True_error_across_iterations_summary_size_%i_n_rep_50.npy' %
                (data_dir, data_name, subset_size))
mfvb = np.load('%s/results/%s/MF-VB_RMT_False_error_across_iterations_summary_size_%i_n_rep_50.npy' %
                (data_dir, data_name, subset_size))

V_star = np.load('%s/results/%s/ground_truth_PC.npy' % (data_dir, data_name))

est = [ebpca, mfvb_RMT, mfvb]

error_groups = [[0,2], [1,2]]
label_groups = [['EB-PCA', 'MF-VB'], ['MF-VB', 'MF-VB (RMT init)']]
color_groups = [['tab:red', 'tab:blue'], ['tab:blue', 'tab:cyan']]
for i_group in range(2):
    joint_error = []
    for i in error_groups[i_group]:
        tmp = []
        for j in range(6):
            tmp.append([est[i][k, j] for k in range(50)])
        joint_error.append(tmp)
    # plot boxplots of estimation error across iterations
    # contrasting different methods
    plot_boxplot_series(joint_error, color_panel = color_groups[i_group],
                        labels = label_groups[i_group],
                        title = '1000 Genomes (1000 SNPs)')
    plt.savefig('%s/figures/%s/error_across_iterations_%s_%s_subset_size_%i.png' % \
                (data_dir, data_name, label_groups[i_group][0], label_groups[i_group][1], subset_size))