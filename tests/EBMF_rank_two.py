import sys
sys.path.append('../')
from scipy.linalg import svd
import numpy as np
from ebpca.pca import get_pca
from ebpca.empbayes import PointNormalEB, NonparEB
from ebpca.amp import ebamp_gaussian
from simulation.helpers import simulate_prior, signal_plus_noise_model, \
    get_marginal_alignment, get_joint_alignment, regress_out_top
from ebpca.preprocessing import normalize_obs
from simulation.rank_two_figures import plot_rank2_dePC
from simulation.rank_two import run_rankK_EBPCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--par_setting", type=int, help="which parameter setting",
                    default=1, const=1, nargs='?')
parser.add_argument("--iters", type=int, help="number of iterations",
                    default=1, const=1, nargs='?')
parser.add_argument("--prior", type=int, help="prior",
                    default='Two_points_normal', const='Two_points_normal', nargs='?')
args = parser.parse_args()

def get_alignment(u1, u2):
    return np.sum(u1 * u2) / (np.sqrt(np.sum(u1 ** 2)) * np.sqrt(np.sum(u2 ** 2)))

def print_alignment(U_est, U_star, k):
    print([get_alignment(U_est[:, k, i], U_star[:,k]) for i in range(U_est.shape[2])])

# -----------------------------------------------------
# test mean field VB with multivariate nonpar prior
# on simulated examples:
# uniform circle / three points
# in comparison to EB-PCA
# across signal strengths
# -----------------------------------------------------

# simulate data from 2-dim prior
n = 1000
gamma = 1
d = int(n * gamma)
if args.par_setting == 1:
    s_star = [4, 2]
else:
    s_star = [14, 9]
iters = args.iters
prior = args.prior
rank = 2

signal_iter_prefix = 's_%i_%i/iter_%s' % (s_star[0], s_star[1], iters)

print(signal_iter_prefix)

# initiate denoiser
optimizer="Mosek"
udenoiser = NonparEB(optimizer=optimizer, to_save=False)
vdenoiser = NonparEB(optimizer=optimizer, to_save=False)

# simulate data from signal+noise model
np.random.seed(2)
seeds = [np.random.randint(0, 10000, 1) for i in range(2)]
u_star = simulate_prior(prior, n, seed=seeds[0][0], rank=rank)
v_star = simulate_prior(prior, d, seed=seeds[1][0], rank=rank)
Y = signal_plus_noise_model(u_star, v_star, np.diag(s_star), seed=1, rank=rank)
# normalize the observational matrix
X = normalize_obs(Y, rank)

for pca_method in ['MF-VB', 'EB-PCA']:
    U_mar, V_mar, conv = run_rankK_EBPCA('marginal', X, rank, iters,
                                         optimizer="Mosek", pca_method=pca_method)
    U_joint, V_joint, conv = run_rankK_EBPCA('joint', X, rank, iters,
                                             optimizer="Mosek", pca_method=pca_method)
    plot_rank2_dePC(v_star, V_mar, V_joint, prior, s_star,
                    fig_prefix='figures/EBMF/rank_two/%s' % signal_iter_prefix, enlarge_star=False,
                    pca_method='%s V' % pca_method, plot_lim=3.5)
    plot_rank2_dePC(u_star, U_mar, U_joint, prior, s_star,
                    fig_prefix='figures/EBMF/rank_two/%s' % signal_iter_prefix, enlarge_star=False,
                    pca_method='%s U' % pca_method, plot_lim=3.5)

# naive rescaling
EBPCA_scaling = np.sqrt(np.sum(U_mar[:,:,-1]**2))
EBPCA_scaling_V = np.sqrt(np.sum(V_mar[:,:,-1]**2))

# for MF-VB with RMT initialization
pca_method = 'MF-VB'
U_mar, V_mar, conv = run_rankK_EBPCA('marginal', X, rank, iters,
                                     optimizer="Mosek", pca_method=pca_method,
                                     ebpca_ini=True)

U_joint, V_joint, conv = run_rankK_EBPCA('joint', X, rank, iters,
                                         optimizer="Mosek", pca_method=pca_method,
                                         ebpca_ini=True)

print('#####################')
print('Apply naive rescaling to match the scale of VB-MF output to that of EB-PCA')
print(np.min(V_joint[:, :, -1]))
print(np.max(V_joint[:, :, -1]))

def match_mag(U, scaling):
    U[:, :, -1] = U[:, :, -1] / np.sqrt(np.sum(U[:, :, -1]**2)) * scaling
    return U
U_mar = match_mag(U_mar, EBPCA_scaling)
U_joint =match_mag(U_joint, EBPCA_scaling)

V_mar = match_mag(V_mar, EBPCA_scaling_V)
V_joint = match_mag(V_joint, EBPCA_scaling_V)

print('#####################')
print(np.min(V_joint[:, :, -1]))
print(np.max(V_joint[:, :, -1]))

plot_rank2_dePC(u_star, U_mar, U_joint, prior, s_star,
                fig_prefix='figures/EBMF/rank_two/%s' % signal_iter_prefix, enlarge_star=False,
                pca_method='%s U (RMT init)' % pca_method, plot_lim=3.5)

plot_rank2_dePC(v_star, V_mar, V_joint, prior, s_star,
                fig_prefix='figures/EBMF/rank_two/%s' % signal_iter_prefix, enlarge_star=False,
                pca_method='%s V (RMT init)' % pca_method, plot_lim=3.5)
