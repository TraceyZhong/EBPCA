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
u_star = np.loadtxt('result/simulation/simu_test_ebmf_u0.txt')
v_star = np.loadtxt('result/simulation/simu_test_ebmf_v0.txt')

n, d = A.shape
pcapack = get_pca(A, 1)

# use parametric Point normal denoiser
udenoiser = PointNormalEB(to_show = False)
vdenoiser = PointNormalEB(to_show = False)
# expected 1st and 2nd values of objective function: 11425635.27, 11425735.36
L_est, F_est = ebmf(pcapack, 20, ldenoiser=udenoiser, fdenoiser=vdenoiser)

print('Parametric denoiser:')
print([get_alignment(L_est[:, :, j], u_star) for j in range(L_est.shape[2])])

exit()

# use nonparametric prior
udenoiser = NonparEB(optimizer="Mosek", ftol=1e-1, to_save=False, to_show = True,
                     nsupp_ratio=np.sqrt(n)/n) # np.sqrt(n)/n
vdenoiser = NonparEB(optimizer="Mosek", ftol=1e-1, to_save=False, to_show = True,
                     nsupp_ratio=np.sqrt(d)/d) # np.sqrt(d)/d
L_par_est, F_par_est = ebmf(pcapack, 1, ldenoiser=udenoiser, fdenoiser=vdenoiser,
                            update_family = 'nonparametric')


print('Nonparametric denoiser:')
print(L_par_est.shape)
print([get_alignment(L_par_est[:, :, j], u_star) for j in
       range(L_par_est.shape[2])])
