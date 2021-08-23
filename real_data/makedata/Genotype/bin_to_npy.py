# -
# For either 1000 Genomes or Hapmap3
# load genotype data in bed
# normalize it by 
# (1) centering and scaling under Binomial model
# (2) removing SNPs with low variation
# and output Python files
#
# Chang Su
# c.su@yale.edu
# Dec 15, 2020
# -

import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, help="which genotype data to process",
                    default='1000G', const='1000G', nargs='?')
parser.add_argument("--size", type=int, help="size of the genotype data",
                    default=100000, const=10000, nargs='?')
args = parser.parse_args()
data_name = args.data_name
size = args.size

genotype_100k = read_plink_genotype('%s.subset.%i'  % (data_name, size))

# create a size 1e5 subset
norm_100k= normalize_genotype(genotype_100k) 

np.save('normalized_%s_%i.npy' % (data_name, size))
