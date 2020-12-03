import numpy as np
from ebpca.pca import get_pca, check_residual_spectrum
from ebpca.preprocessing import plot_pc
from showcase import prep_subsets

subset_sizes = {'1000G': 1000, 'UKBB': 2000}
real_data_rank = {'1000G': 4, 'UKBB': 2}
sv_lim = {'1000G': [0,3], 'UKBB': [0,3]}
real_data_rank = {'1000G': 4, 'UKBB': 2, 'PBMC': 3, 'GTEx': 2}

np.random.seed(1)
seeds = np.random.randint(0, 10000, 50)

for data_name in ['1000G', 'UKBB']:
    for which_subset in range(1, 5, 1):
        # norm_data = np.load('results/%s/norm_data.npy' % data_name)
        # prep_subsets(norm_data, subset_sizes[data_name], data_name,
        #              subset_sizes[data_name], seeds[:10], real_data_rank[data_name], n_rep=10)
        X = np.load('results/%s/subset_size_%i_n_copy_%i.npy' % (data_name, subset_sizes[data_name], which_subset))
        print(which_subset)
        print(X[:5, :5])
        sub_pcapack = get_pca(X, real_data_rank[data_name])
        check_residual_spectrum(sub_pcapack, xmin=sv_lim[data_name][0], xmax=sv_lim[data_name][1],
                                to_save=True, fig_prefix=data_name,
                                label=data_name + '_subset_%i' % which_subset)
        plot_pc(sub_pcapack.X, label=data_name + '_subset_%i' % which_subset, nPCs=real_data_rank[data_name],
                to_show=False, to_save=True, fig_prefix='%s/' % data_name)

