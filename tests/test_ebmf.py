import numpy as np
# ----------------------------------------
# Benchmark with implementation from EBMF
# ----------------------------------------
import sys
sys.path.append('../../generalAMP')
from ebpca.empbayes import PointNormalEB, NonparEB
from ebpca.misc import ebmf
from tutorial import get_alignment
from ebpca.pca import get_pca

# benchmark log likelihood

# the unit of computing expected log likelihood

# matrix generated according to the rank 1 model
A = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')
u_star = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')
v_star = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')
n, d = A.shape
pcapack = get_pca(A, 1)

# use parametric Point normal denoiser
L_est, F_est = ebmf(pcapack, 100) # expected 1st and 2nd values of objective function: 11425635.27, 11425735.36

# use nonparametric prior
udenoiser = NonparEB(optimizer="Mosek", ftol=1e-3, to_save=False,
                     nsupp_ratio=np.sqrt(n)/n)
vdenoiser = NonparEB(optimizer="Mosek", ftol=1e-3, to_save=False,
                     nsupp_ratio=np.sqrt(d)/d)
L_par_est, F_par_est = ebmf(pcapack, 100, ldenoiser=udenoiser, fdenoiser=vdenoiser,
                            update_family = 'nonparametric')

print('Parametric denoiser:')
print([get_alignment(L_est[:, :, j], u_star) for j in range(L_est.shape[2]-10, L_est.shape[2])])
print('Nonparametric denoiser:')
print(L_par_est.shape)
print([get_alignment(L_par_est[:, :, j], u_star) for j in
       range(L_par_est.shape[2] - 10, L_par_est.shape[2])])
