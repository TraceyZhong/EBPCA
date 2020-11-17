import numpy as np
import os
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.pca import get_pca
from ebpca.empbayes import NonparEBChecker
from ebpca.amp import ebamp_gaussian
from ebpca.misc import ebmf
from simulation.helpers import approx_prior
from tutorial import get_alignment

# create directories to save the figures
fig_prefix = 'figures/univariate'
if not os.path.exists(fig_prefix):
    for i in range(3):
        os.makedirs('%s/Figure%i' % (fig_prefix, i + 1))

# ----------------------------------------------
# Figure 1:
# alignment comparison across methods
# boxplot
# ----------------------------------------------


# ------------------------------------------------------
# Figure 2:
# predicted density, fitted density versus observed data
# contrasting EBMF and EB-PCA
# histogram
# ------------------------------------------------------

# use data from replication 0
i = 0
prior = 'Point_normal'
s_star = 1.3
data_prefix = 'output/univariate/%s/data/s_%.1f' % (prior, s_star)
u_star = np.load('%s_copy_%i_u_star.npy' % (data_prefix, i), allow_pickle=False)
v_star = np.load('%s_copy_%i_v_star.npy' % (data_prefix, i), allow_pickle=False)
X = np.load('%s_copy_%i.npy' % (data_prefix, i), allow_pickle=False)

# make pca pack
pcapack = get_pca(X, 1)

# approximate true prior with empirical distribution
[uTruePriorLoc, uTruePriorWeight] = approx_prior(u_star, pcapack.U)
[vTruePriorLoc, vTruePriorWeight] = approx_prior(v_star, pcapack.V)

# Figure 2 folder
f2_prefix = 'univariate/Figure2/'

# run EBMF
ldenoiser = NonparEBChecker(uTruePriorLoc, uTruePriorWeight, optimizer ="Mosek",
                            to_save=True, fig_prefix=f2_prefix+'EBMF')
fdenoiser = NonparEBChecker(vTruePriorLoc, vTruePriorWeight, optimizer ="Mosek",
                            to_save=True, fig_prefix=f2_prefix+'EBMF')
U_ebmf, V_ebmf = ebmf(pcapack, ldenoiser=ldenoiser, fdenoiser=fdenoiser,
                      update_family='nonparametric', iters=5)

print('EBMF alignments:', [get_alignment(U_ebmf[:, :, j], u_star) for j in range(U_ebmf.shape[2])])

# run EB-PCA
udenoiser = NonparEBChecker(uTruePriorLoc, uTruePriorWeight, optimizer ="Mosek",
                            to_save=True, fig_prefix=f2_prefix+'EB-PCA')
vdenoiser = NonparEBChecker(vTruePriorLoc, vTruePriorWeight, optimizer ="Mosek",
                            to_save=True, fig_prefix=f2_prefix+'EB-PCA')
U_ebpca, V_ebpca = ebamp_gaussian(pcapack, iters=5, udenoiser=udenoiser, vdenoiser=vdenoiser)

print('EB-PCA alignments:', [get_alignment(U_ebpca[:, :, j], u_star) for j in range(U_ebpca.shape[2])])

