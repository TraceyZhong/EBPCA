from collections import namedtuple

import numpy as np
import scipy.integrate as integrate

from ebpca.empbayes import NonparBayes
from ebpca.empbayes import PointNormalEB

np.random.seed(1997)

nsamples = 5000

print("nsamples =",nsamples)

StateEvolution = namedtuple("StateEvolutation", "gammas gammas_bar gamma_star, gamma_star_bar aspect_ratio s")

AlignmentEvolution = namedtuple("AlignmentEvolution", \
    "valigns ualigns valign_star ualign_star")

# for rank one
def get_state_evolution(s, aspect_ratio, ummse, vmmse, amp_iter = 10, ftol = 0.001):
    gamma = (aspect_ratio * s**4 - 1)/ (aspect_ratio*s**2 + 1)
    gammas = [gamma]
    gammas_bar=[]
    for _ in range(amp_iter):
        gamma_bar = s**2*aspect_ratio*(1- vmmse(gamma))
        gamma = s**2*(1-ummse(gamma_bar))
        gammas_bar.append(gamma_bar)
        gammas.append(gamma)
    for _ in range(amp_iter, 10):
        gamma_bar_new = s**2*aspect_ratio*(1- vmmse(gamma))
        gamma_new = s**2*(1-ummse(gamma_bar))
        if (abs(gamma_bar - gamma_bar_new) < ftol) and (abs(gamma - gamma_new) < ftol):
            break
        else:
            gamma_bar, gamma = gamma_bar_new, gamma_new
        
    return StateEvolution(np.array(gammas), np.array(gammas_bar), gamma, gamma_bar, aspect_ratio, s)
    # return ualign, valign

def get_alignment_evolution(se):
    svd_alignment = np.sqrt((se.aspect_ratio * se.s**4 - 1)/(se.aspect_ratio * se.s**4 + se.aspect_ratio * se.s**2))
    ualigns = np.sqrt(se.gammas) / se.s
    valigns = np.sqrt(se.gammas_bar / se.aspect_ratio)/ se.s
    valigns = np.insert(valigns, 0, svd_alignment)
    ualign_star = np.sqrt(se.gamma_star)/ se.s
    valign_star = np.sqrt(se.gamma_star_bar/ se.aspect_ratio)/ se.s
    return AlignmentEvolution(valigns, ualigns, valign_star, ualign_star)

## --- MMSE updates --- ##
# print("Running Sampling.")
# def two_points(gamma):  
#     # -1 +1
#     truth = np.repeat([1,-1], int(nsamples/2))
#     x = truth * np.sqrt(gamma) + np.random.normal(size = (nsamples,))
#     est = np.tanh(np.sqrt(gamma)*x)
#     return np.mean((est - truth)**2)

# print("Running analytical sol.")
def two_points(gamma):
    
    def integrand(x, gamma):
        return np.tanh(gamma + x * np.sqrt(gamma))**2 * np.exp(-x*x/2)

    res = integrate.quad(integrand, -np.inf, np.inf, args = (gamma,)) / np.sqrt(2 * np.pi) 
    res = res[0]

    return 1 - res

# print("Numerical integration for point normal.")
# def point_normal(gamma, sparsity = 0.5):

#     mu_x = 0
#     sigma_x = np.sqrt(1/sparsity)
#     mu_y = np.sqrt(gamma)
#     sigma_y = 1 

#     mu_y_tilde = PointNormalEB._eval_mu_y_tilde(mu_x,mu_y)
#     sigma_y_tilde = PointNormalEB._eval_sigma_y_tilde(mu_y, sigma_x, sigma_y)

#     def integrand(y, gamma):
#         mu_x_tilde = (y * mu_y * sigma_x**2 + mu_x * sigma_y**2) / (mu_y**2 * sigma_x**2 + sigma_y**2)
#         numerator = (sparsity * mu_x_tilde * np.exp(-(y - mu_y_tilde)**2/sigma_y_tilde**2) / sigma_y_tilde)
#         # sigma_y == 1
#         denominator = (1-sparsity)*np.exp(-y**2) + sparsity * np.exp(-(y - mu_y_tilde)**2/sigma_y_tilde**2)/ sigma_y_tilde
#         return numerator**2/denominator

#     res = integrate.quad(integrand, -100, 100, args = (gamma,)) / np.sqrt(2 * np.pi) 
#     res = res[0]

#     return 1 - res



def uniform(gamma):
    # [0, sqrt(3)]
    print("what is gamma ", gamma)
    truth = np.random.uniform(0, 1, size=(nsamples,1))
    truth = truth / np.sqrt(np.sum(truth**2) / nsamples)
    truePriorWeight = np.full((nsamples,), 1/nsamples)
    denoiser = NonparBayes(truth, truePriorWeight, optimizer="EM")
    x = truth * np.sqrt(gamma) + np.random.normal(size = (nsamples,1))
    est = denoiser.denoise(x,np.array([[np.sqrt(gamma)]]),np.array([[1]]))
    print("Finish denoising")
    return np.mean((est - truth)**2)
    

def point_normal(gamma, sparsity = 0.1):
    # print("what is gamma", gamma)
    mask = np.random.binomial(1, sparsity, size = (nsamples,1))
    normals = np.random.normal(size = (nsamples,1))
    truth = mask * normals
    truth = truth / np.sqrt(np.sum(truth**2) / nsamples)
    normals = np.random.normal(size = (int(nsamples*sparsity),1))
    truth = np.concatenate((normals, np.zeros(shape=(nsamples - int(nsamples*sparsity), 1))), axis = 0)
    truePriorLoc = np.append(normals, [[0]], axis = 0)
    truePriorWeight = np.append(np.full(shape=(int(nsamples*sparsity),), fill_value = sparsity/nsamples), 0)
    denoiser = NonparBayes(truePriorLoc, truePriorWeight, optimizer="EM")
    x = truth * np.sqrt(gamma) + np.random.normal(size = (nsamples,1))
    est = denoiser.denoise(x,np.array([[np.sqrt(gamma)]]),np.array([[1]]))
    print("Finish denosing")
    return np.mean((est - truth)**2)


if __name__=="__main__":
    n = 2000
    p = 1500
    rank = 1
    amp_iters = 5
    s = 1.5
    se = get_state_evolution(s, p/n, two_points, two_points, amp_iters)
    ae = get_alignment_evolution(se)