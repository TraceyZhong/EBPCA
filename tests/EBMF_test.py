import sys
sys.path.append('/Users/chang/PycharmProjects/generalAMP')
from scipy.linalg import svd
import numpy as np
from ebpca.misc import ebmf, MeanFieldVB
from ebpca.pca import get_pca
from ebpca.empbayes import PointNormalEB, NonparEB
from ebpca.amp import ebamp_gaussian

# ----------------
# test point normal:
# An example where we show that our Python implementation is exactly the same as flashr()
# using exactly the same input data
# as shown in EBMF_test.R
# ----------------

def get_alignment(u1, u2):
    return np.sum(u1 * u2) / (np.sqrt(np.sum(u1 ** 2)) * np.sqrt(np.sum(u2 ** 2)))

def print_alignment(U_est, U_star):
    print([get_alignment(U_est[:, 0, i], U_star) for i in range(U_est.shape[2])])

# load a dataset generated in EBMF_test.R
# where the true prior used is point normal for both u0 and v0
u0 = np.loadtxt('result/simulation/simu_test_ebmf_u0.txt')
v0 = np.loadtxt('result/simulation/simu_test_ebmf_v0.txt')
# matrix generated according to the rank 1 model
A = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')

pcapack = get_pca(A, 1)

iters = 3

# run rank 1 EBMF with point normal prior
# to compare with inmplementation in R flashr()
(U1, V1, obj_funcs) = ebmf(pcapack,
                           ldenoiser=PointNormalEB(to_save=False),
                           fdenoiser=PointNormalEB(to_save=False),
                           update_family='point-normal',
                           ebpca_scaling=False,
                           iters=iters)

# values of objective functions from flashr():
# Iteration      Objective
#           1    11425635.27
#           2    11425735.36
#           3    11425770.45

print('rank 1 EBMF with point normal prior')
print_alignment(U1, u0)
print_alignment(V1, v0)

# -----------------------------
# test multivar implementation
# compare estimates using 1-dim and multivar
# implementations
# -----------------------------

# run rank 1 EBMF with nonparametric prior
(U1, V1, obj_funcs) = ebmf(pcapack,
                           ldenoiser=NonparEB(to_save = False),
                           fdenoiser=NonparEB(to_save = False),
                           update_family='nonparametric',
                           ebpca_scaling=False,
                           iters=iters)

print('EBMF with nonparametric prior')
print_alignment(U1, u0)
print_alignment(V1, v0)

# run general implementation for Mean Field VB
# with nonparametric prior
# to check that if MeanFieldVB gives exactly the same result as
# ebmf with prior_family = 'nonparametric'
(U1, V1, obj_funcs) = MeanFieldVB(pcapack,
                                  ldenoiser=NonparEB(to_save = False),
                                  fdenoiser=NonparEB(to_save = False),
                                  ebpca_scaling=False,
                                  iters=iters)

print('Multivar EBMF with nonparametric prior')
print_alignment(U1, u0)
print_alignment(V1, v0)

# similar performance as univariate EBMF
# as the implementation of MeanFieldVB is general
# such that it sould include rank=1 as a special case