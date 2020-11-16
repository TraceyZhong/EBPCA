# -----------------------------------------------
# Benchmark EBMF by comparing with flash() in R
#
# 1) show that our Python implementation is the
#    same as the original one in R
# 2) show that EBMF also works under our NonparEB
#    denoiser
# 3) compare with AMP
# -----------------------------------------------

import numpy as np
import sys
sys.path.append('../../generalAMP')
from ebpca.empbayes import PointNormalEB, NonparEB
from ebpca.misc import ebmf
from tutorial import get_alignment
from ebpca.pca import get_pca

# matrix generated according to the rank 1 model
# generated in explore_ebmf.Rmd
A = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')
u_star = np.loadtxt('result/simulation/simu_test_ebmf_u0.txt')
v_star = np.loadtxt('result/simulation/simu_test_ebmf_v0.txt')

n, d = A.shape
pcapack = get_pca(A, 1)

# ------------------------------------------------
# test EBMF with parametric Point normal denoiser
# ------------------------------------------------

udenoiser = PointNormalEB(to_show = False)
vdenoiser = PointNormalEB(to_show = False)
# expected 1st and 2nd values of objective function: 11425635.27, 11425735.36
L_est, F_est = ebmf(pcapack, ldenoiser=udenoiser, fdenoiser=vdenoiser, iters = 21)
print('Parametric denoiser:')
# expected obj at iter=20: 11425830.34; alignment_u: 0.7594148
print([get_alignment(L_est[:, :, j], u_star) for j in range(L_est.shape[2])])

# -------------------------------------
# test EBMF with nonparametric denoiser
# -------------------------------------

udenoiser = NonparEB(optimizer="Mosek", ftol=1e-1, to_save=False, to_show = False,
                     nsupp_ratio=np.sqrt(n)/n)
vdenoiser = NonparEB(optimizer="Mosek", ftol=1e-1, to_save=False, to_show = False,
                     nsupp_ratio=np.sqrt(d)/d)
L_nonpar_est, F_nonpar_est = ebmf(pcapack, ldenoiser=udenoiser, fdenoiser=vdenoiser,
                            update_family = 'nonparametric')

print('Nonparametric denoiser:')
print([get_alignment(L_nonpar_est[:, :, j], u_star) for j in
       range(L_nonpar_est.shape[2])])