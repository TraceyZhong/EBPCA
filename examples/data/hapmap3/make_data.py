from pandas_plink import read_plink1_bin
import numpy as np
import sys
sys.path.extend(['../../../../generalAMP'])
import matplotlib.pyplot as plt
import pandas as pd
import os
from tutorial import redirect_pc
from ebpca.pca import get_pca, check_residual_spectrum
from ebpca.preprocessing import normalize_obs, plot_pc

def read_plink_genotype(dir):
    G = read_plink1_bin(dir+'.bed', dir+'.bim', dir+'.fam', verbose=True)
    genotype = G.values
    return genotype

def normalize_genotype(genotype):
    af = np.nanmean(genotype, axis=0) / 2
    # fill NA with mean
    genotype_mean_m = np.repeat((2 * af).reshape(-1,1), genotype.shape[0], axis = 1).T
    genotype[np.isnan(genotype)] = genotype_mean_m[np.isnan(genotype)]
    sd = np.std(genotype, axis=0)
    # remove snps with zero variation
    zero_sd_ind = sd > 1e-6
    geno_norm = (genotype[:, zero_sd_ind] - 2 * af[np.newaxis, zero_sd_ind]) / sd[np.newaxis, zero_sd_ind]
    return geno_norm

def plot_popu_strat(inds, u, data, popu_label):
    df = pd.DataFrame(dict(x=list(u[:, inds[0]]), y=list(u[:, inds[1]]), label=popu_label))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=1.5, label=name)
    ax.legend()
    plt.title('{} population stratification'.format(data))
    plt.xlabel('PC {}'.format(inds[0]))
    plt.ylabel('PC {}'.format(inds[1]))
    plt.show()

# load metadata for hapmap3
hapmap3_popu_df = pd.read_table('relationships_w_pops_041510.txt')

# merge refined population into 5 populations
# corresponding to classification in 1000G
hapmap3_popu_df['broad_population'] = np.repeat('unknown', hapmap3_popu_df.shape[0])
popu_lookup = {'African': ["ACB","ASW","ESN","GWD","LWK","MSL","YRI", "MKK"],
               'Hispanic': ["CLM","MXL","PEL","PUR", "MEX"],
               'East-Aisan': ["CDX","CHB","CHS","JPT","KHV", "CHD"],
               'Caucasian': ["CEU","FIN","GBR","IBS","TSI"],
               'South-Asian': ["BEB","GIH","ITU","PJL","STU"]}
for i in range(5):
    hapmap3_popu_df.loc[hapmap3_popu_df['population'].isin(list(popu_lookup.values())[i]).values, 'broad_population'] = \
        list(popu_lookup.keys())[i]
np.save('hapmap3_popu.npy', hapmap3_popu_df)

# print sample size per population
print(hapmap3_popu_df.broad_population.value_counts())
print(hapmap3_popu_df.shape)

# convert plink files to .npy
for size in ['full', 1000, 10000, 100000]:
    if size == 'full':
        fig_label = 'Hapmap3_%s' % size
        if not os.path.exists('hapmap3.npy'):
            hapmap3 = read_plink_genotype('hapmap3')
            hapmap3 = normalize_genotype(hapmap3)
            print(hapmap3.shape)
            exit()
            np.save('hapmap3.npy', hapmap3, allow_pickle=False)
        else:
            hapmap3 = np.load('hapmap3.npy')
    else:
        fig_label = 'Hapmap3_%.0e' % size
        if not os.path.exists('hapmap3_%.0e.npy' % size):
            hapmap3 = read_plink_genotype('hapmap3.subset.%i' % size)
            hapmap3 = normalize_genotype(hapmap3)
            np.save('hapmap3_%.0e.npy' % size, hapmap3, allow_pickle=False)
        else:
            hapmap3 = np.load('hapmap3_%.0e.npy' % size)

    # normalize data to satisfy the EB-PCA assumption
    hapmap3 = normalize_obs(hapmap3, 4)

    # compute svd
    pcapack = get_pca(hapmap3, 4)
    if size == 'full':
        if not os.path.exists('u_star.npy'):
            np.save('u_star.npy', pcapack.U)
    u_star = np.load('u_star.npy')
    u = redirect_pc(pcapack.U, u_star)

    # check pc
    plot_pc(pcapack.X, label='Hapmap3', nPCs=4,
            to_show=False, to_save=True, fig_prefix=fig_label)
    # check singular value distribution
    check_residual_spectrum(pcapack, to_save=True, fig_prefix='Hapmap3', label=fig_label)

    for PCs in [[0,1], [2,3]]:
        plot_popu_strat(PCs, u, fig_label, hapmap3_popu_df['broad_population'].values)
        plt.savefig('figures/%s_PC_%i_%i.png' % (fig_label, PCs[0], PCs[1]))
        plt.close()