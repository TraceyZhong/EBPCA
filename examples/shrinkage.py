# diagnose the reason for shrinkage
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.empbayes import NonparEB
from ebpca.pca import get_pca
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from ebpca.preprocessing import normalize_obs
from showcase import normalize_samples

# load ground truth of 1000G African
data_name = '1000G_African'
V_star = np.load('results/%s/ground_truth_PC.npy' % data_name)
print(V_star.shape)
V_star = V_star * np.sqrt(V_star.shape[0])
for i in range(2):
    print(np.var(V_star[:, i]))
# load population labels
popu_label_df = pd.read_csv('data/1000G/Popu_labels.txt', sep=' ')
popu_label_broad = popu_label_df['Population_broad'].values
popu_label_refined = popu_label_df['Population'].values
popu_label = popu_label_refined[(popu_label_df['Population_broad'] == 'African').values]
Af_outlier = np.load('data/1000G/subset_index.npy')
popu_label = popu_label[Af_outlier]

# -
# - task 1
# -
# - check estimates of variance of noise by RMT theory

# compute PCA
# full data
norm_data = np.load('results/%s/norm_data.npy' % data_name) # normalized
full_pcapack = get_pca(norm_data, 2)
# 5000 SNPs
X = np.load('results/%s/subset_size_%i_n_copy_%i.npy' % (data_name, 5000, 1)) # normalized
# X = normalize_samples(X)
# X = normalize_obs(X, 2)
sub_pcapack = get_pca(X, 2)

# - evaluate variance estimates
# - full data
print('============ \n full data \n============')
# theoretical
full_theo_var = (1 - full_pcapack.feature_aligns**2) * V_star.shape[0]
print('\t estimated noise var:', full_theo_var)

# empirical
purple_cluster = [(V_star[i, 0] >= -1 and V_star[i, 0] <= 0) and (V_star[i, 1] <= -1 and V_star[i, 1] >= -2.5) \
                  for i in range(V_star.shape[0])]
full_emp_var = [np.var(full_pcapack.V[purple_cluster, i]) * V_star.shape[0] for i in range(2)]
print('\t emp var of the purple cluster: ', full_emp_var)

# - 5000 SNPs
print('============ \n 5000SNPs \n============')
# theoretical
subset_theo_var = (1 - sub_pcapack.feature_aligns**2) * V_star.shape[0]
print('\t estimated noise var:', subset_theo_var)

# empirical
subset_emp_var = [np.var(sub_pcapack.V[purple_cluster, i]) * V_star.shape[0] for i in range(2)]
print('\t emp var of the purple cluster: ', subset_emp_var)

# visualize PCs with estimated variances
Vs = [V_star, sub_pcapack.V[:, :2] * np.sqrt(V_star.shape[0])]
names = ['ground truth', '5000 SNPs']
theo_vars = [full_theo_var, subset_theo_var]
emp_vars = [full_emp_var, subset_emp_var]
fig, axes = plt.subplots(ncols=1, nrows=2, figsize = (3, 6), constrained_layout=True)
plt.setp(axes, xlim=[-5, 2], ylim=[-3, 3])
for i in range(2):
    df = pd.DataFrame(dict(x=list(Vs[i][:, 0]), y=list(Vs[i][:, 1]), label=popu_label))
    # make scatter plot
    groups = df.groupby('label')
    for name, group in groups:
        axes[i].scatter(group.x, group.y, marker='o', s=3, label=name)
    axes[i].set_title(names[i])
    axes[i].set_xlabel('theo var est:%.4f, emp var:%.4f' %
                       (theo_vars[i][0], emp_vars[i][0]))
    axes[i].set_ylabel('theo var est:%.4f, emp var:%.4f' %
                       (theo_vars[i][1], emp_vars[i][1]))

plt.savefig('figures/shrinkage_full_and_5000SNPs.png')
plt.close()

# -
# - task 2
# -
# - convolve ground truth with Gaussian noise
var = 0.02
cov = 0.01
Sigma = np.array([[var, cov], [cov, var]])
mu = 1
Mu = np.array([[mu, 0], [0, mu]])
print(V_star[:, :2].dot(Mu).shape)
noisy_V = V_star[:, :2].dot(Mu) + np.random.multivariate_normal([0, 0], Sigma, V_star.shape[0])
print(noisy_V.shape)

# -  run npmle
denoiser = NonparEB(optimizer="Mosek", fig_prefix='shrinkage', to_show=False, to_save=False)
denoiser.fit(noisy_V, Mu, Sigma)
V_est = denoiser.denoise(noisy_V, Mu, Sigma)

# -  reconstruct nosiy observation
V_re = V_est.dot(Mu) + np.random.multivariate_normal([0, 0], Sigma, V_star.shape[0])

Vs = [V_star, noisy_V, V_est, V_re]
names = ['ground truth', 'noisy V', 'NPMLE', 'reconstructed V']
fig, axes = plt.subplots(ncols=1, nrows=4, figsize = (3, 3.5*4), constrained_layout=True)
plt.setp(axes, xlim=[-5, 2], ylim=[-3, 3])
# mpl.rcParams.update(mpl.rcParamsDefault)
for i in range(4):
    df = pd.DataFrame(dict(x=list(Vs[i][:, 0]), y=list(Vs[i][:, 1]), label=popu_label))
    # make scatter plot
    groups = df.groupby('label')
    for name, group in groups:
        axes[i].scatter(group.x, group.y, marker='o', s=3, label=name)
    axes[i].set_title(names[i])
plt.savefig('figures/shrinkage_NPMLE_var_%.2f_cov_%.2f.png' % (var, cov))

# -
# - task 3
# -
# - test if noise if overestimated
# X_noisy = full_pcapack.U.dot(np.diag(full_pcapack.mu[:2])).dot(noisy_V.T)
# pcapack = get_pca(X_noisy, 2)
# print('Test if noises are overestimated')
# print('noise var est in artificial noisy data: ', 1 - pcapack.feature_aligns**2)
# udenoiser = NonparEB(optimizer='Mosek', to_save=False)
# vdenoiser = NonparEB(optimizer='Mosek', to_save=False)

# run AMP
# U_est, V_est, conv = ebamp_gaussian(pcapack, iters=5,
#                                     udenoiser=udenoiser, vdenoiser=vdenoiser,
#                                     return_conv=True, muteu=True)
