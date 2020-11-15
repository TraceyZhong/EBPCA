import numpy as np
# ----------------------------------------
# Benchmark with implementation from EBMF
# ----------------------------------------
import sys
sys.path.append('/Users/chang/PycharmProjects/generalAMP')
from ebpca.empbayes import PointNormalEB
from ebpca.misc import ebmf
from scipy.linalg import svd
from ebpca.pca import get_pca

# benchmark log likelihood

# the unit of computing expected log likelihood

# matrix generated according to the rank 1 model
A = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')
u_star = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')
v_star = np.loadtxt('result/simulation/simu_test_ebmf_A.txt')

pcapack = get_pca(A, 1)
ebmf(pcapack) # expected 1st and 2nd values of objective function: 11425635.27, 11425735.36

exit()

u0, s, vh0 = svd(A, full_matrices=False)

n = A.shape[0]

# denoiser u0
EL = u0[:,0]
EL2 = u0[:,0]**2
EF = vh0[0,:]
EF2 = vh0[0,:]**2

elogl = get_cond_logl(EL, EL2, EF, EF2, A, n)
print(elogl)

# load data
u0 = np.loadtxt('/Users/chang/Documents/research/EBPCA/npmleAMPv1/result/simulation/test_ebnm_in_R.txt')
mu = 1
sigma_y = 4

# also benchmark the estimate of EX and EX^2

# the unit of Point normal denoiser is consistent with ebnm

ests = parametric_mle_point_normal(u0, mu, sigma_y)
print(ests)

# the units of denoiser is consistent with ebnm
vdenoiser = PointNormalEB(to_save=True, to_show=False,
                          pi = ests[0], sigma_x = ests[1])
EX = vdenoiser.denoise(u0, mu, sigma_y**2)
VarX = vdenoiser.ddenoise(u0, mu, sigma_y**2) * sigma_y**2 / mu
EX2 = EX**2 + VarX

print(EX[:6])
print(EX2[:6])

loglik = point_normal_e_loglik(ests, u0, sigma_y, mu)
print('Log likeihood: %.4f' % loglik)

print('input data: u0')
print(u0[:6])
# NM_posterior_e_loglik
NM_log = NM_posterior_e_loglik(u0, sigma_y**2, 1, EX, VarX)
print('NM_posterior_e_loglik')
print(NM_log)
