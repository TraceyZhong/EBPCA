import sys
sys.path.append('/Users/chang/PycharmProjects/generalAMP')
from scipy.linalg import svd
import numpy as np
from ebpca.misc import ebmf
from ebpca.pca import get_pca
from ebpca.empbayes import PointNormalEB

# ----------------
# test point normal:
# An example where we show that our Python implementation is exactly the same as flashr()
# using exactly the same input data
# as shown in EBMF_test.R
# ----------------

# load a dataset generated in EBMF_test.R
u0 = np.loadtxt('result/simulation/simu_test_ebmf_u0.txt')
v0 = np.loadtxt('result/simulation/simu_test_ebmf_v0.txt')
# matrix generated according to the rank 1 model
A = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')

pcapack = get_pca(A, 1)

# EBMF for the rank 1 model
(U1, V1, obj_funcs) = ebmf(pcapack,
                ldenoiser=PointNormalEB(to_save = False),
                fdenoiser=PointNormalEB(to_save = False),
                update_family='point-normal',
                ebpca_scaling=False,
                iters=3)

# values of objective functions from flashr():
# Iteration      Objective
#           1    11425635.27
#           2    11425735.36
#           3    11425770.45

def get_alignment(u1, u2):
    return np.sum(u1 * u2) / (np.sqrt(np.sum(u1 ** 2)) * np.sqrt(np.sum(u2 ** 2)))

print('EBMF with point normal prior for left PC1 acc: ')
print([get_alignment(U1[:, i, 0], u0) for i in range(U1.shape[1])])
print('EBMF with point normal prior for right PC1 acc: ')
print([get_alignment(V1[:, i, 0], v0) for i in range(U1.shape[1])])


# test multivariate implementation
