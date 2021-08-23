# -
# Normalize gene expression data
#
# Chang Su
# c.su@yale.edu
# Dec 15, 2020
# -

import os
import numpy as np
import scipy.stats
import pandas as pd

pbmc = pd.read_csv('pbmc_counts.txt', sep = ' ').values.T
celltype = pd.read_csv('pbmc_ct.txt', sep = ' ').values.reshape(-1)

def normalize(dat):
  """  Normalize gene expression data """
  row_sum = np.sum(dat, axis=1)
  # Normalize each cell by the cell size (number of sequenced counts for this cell)
  dat = np.transpose(dat.T / row_sum)
  col_mean = np.mean(dat, axis=0)
  col_sd = np.std(dat, axis=0)
  if np.any(col_sd < 1e-6):
    print('{} columns have zero sd'.format(np.sum(col_sd == 0)))
  subset = col_sd >= 1e-6
  # Remove genes with no variation in expression across cells
  dat = dat[:, subset]
  # Normalize all genes to have zero mean and sd=1
  # such that highly expressed genes do not dominate the PC structure
  norm_dat = (dat - col_mean[np.newaxis, subset]) / col_sd[np.newaxis, subset]
  return norm_dat

def plot_2dim(u, inds, col, size):
  fig, ax = plt.subplots()
  df = pd.DataFrame(dict(x=list(u[:, inds[0]]), y=list(u[:, inds[1]]), label=col))
  groups = df.groupby('label')
  for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=1.5, label=name)
  ax.legend()
  plt.xlabel('PC %i' % inds[0])
  plt.ylabel('PC %i' % inds[1])
  plt.title('Cell types, %s' % str(size))
  # plt.show()
  plt.savefig('figures/2pc_%i_%i_%s.png' % (inds[0], inds[1], str(size)))
  plt.close()

# normalize the data and remove outliers
pbmc_norm = normalize(pbmc)
u,s,vh = np.linalg.svd(pbmc_norm,full_matrices=False)
for i in range(3):
  plot_2dim(u, [2 * i, 2 * i + 1], celltype, 'with_outlier')

# remove outlier and redo normalization & SVD
pc1_outlier = u[:,1] >= -0.1
pbmc_norm_clean = normalize(pbmc[pc1_outlier,:])
np.save('pbmc_norm_clean.npy', pbmc_norm_clean)
np.save('pbmc_celltype_clean.npy', celltype[pc1_outlier])

print(pbmc_norm.shape)
print(pbmc_norm_clean.shape)
