# plot marginal histogram of EB-PCA estimates overlaid with theoretical density
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from ebpca.pca import get_pca
from ebpca.empbayes import NonparEBChecker, NonparEB
from ebpca.amp import ebamp_gaussian
from simulation.helpers import approx_prior
from tutorial import normalize_pc

# parameter settings
data_name = '1000G'
rank = 4
optimizer = 'Mosek'
iters = 5

# figure prefix
fig_prefix = '%s/EB-PCA_marginal' % data_name

# use random subset 1
n_copy = 1
X = np.load('results/%s/subset_n_copy_%i.npy' % (data_name, n_copy))
# load full PC to approximate truth prior
V_star = np.load('results/%s/ground_truth_PC.npy' % data_name)
V_star = normalize_pc(V_star)
n, _ = X.shape

# run joint EB-PCA
# prepare the PCA pack
pcapack = get_pca(X, rank)
# approximate true prior with empirical distribution
vTruePriorLoc, vTruePriorWeight = approx_prior(V_star, pcapack.V)
# initiate denoiser
udenoiser = NonparEB(optimizer=optimizer, to_save=False)
vdenoiser = NonparEBChecker(vTruePriorLoc, vTruePriorWeight, optimizer="Mosek",
                            to_save=True, fig_prefix=fig_prefix, PCname='V', print_SNR=True)
# run AMP
U_est, V_est, conv = ebamp_gaussian(pcapack, iters=iters,
                                    udenoiser=udenoiser, vdenoiser=vdenoiser,
                                    return_conv=True)
