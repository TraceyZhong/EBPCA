from collections import namedtuple

import numpy as np
import scipy.integrate as integrate

from ebpca.empbayes import NonparBayes

np.random.seed(1997)

nsamples = 2000

StateEvolution = namedtuple("StateEvolutation", "gammas gammas_bar gamma_star, gamma_star_bar aspect_ratio s")

AlignmentEvolution = namedtuple("AlignmentEvolution", \
    "valigns ualigns valign_star ualign_star")

# for rank one
def get_state_evolution(s, aspect_ratio, ummse, vmmse, amp_iter = 10, ftol = 0.001):
    gamma = (aspect_ratio * s**4 - 1)/ (aspect_ratio*s**2 + 1)
    gammas = [gamma]
    gammas_bar=[]
    for _ in range(amp_iter):
        gamma_bar = s**2*aspect_ratio*(1- ummse(gamma))
        gamma = s**2*(1-vmmse(gamma_bar))
        gammas_bar.append(gamma_bar)
        gammas.append(gamma)
    for _ in range(amp_iter, 100):
        gamma_bar_new = s**2*aspect_ratio*(1- ummse(gamma))
        gamma_new = s**2*(1-vmmse(gamma_bar))
        if (abs(gamma_bar - gamma_bar_new) < ftol) and (abs(gamma - gamma_new) < ftol):
            break
        else:
            gamma_bar, gamma = gamma_bar_new, gamma_new
        
    return StateEvolution(np.array(gammas), np.array(gammas_bar), gamma, gamma_bar, aspect_ratio, s)
    # return ualign, valign

def get_alignment_evolution(se):
    valigns = np.sqrt(se.gammas) / se.s
    ualigns = np.sqrt(se.gammas_bar)/ se.s
    valign_star = np.sqrt(se.gamma_star)/ se.s
    ualign_star = np.sqrt(se.gamma_star_bar)/ se.s
    return AlignmentEvolution(valigns, ualigns, valign_star, ualign_star)

## --- MMSE updates --- ##
def two_points(gamma):  
    # -1 +1
    truth = np.repeat([1,-1], int(nsamples/2))
    x = truth * np.sqrt(gamma) + np.random.normal(size = (nsamples,))
    est = np.tanh(np.sqrt(gamma)*x)
    return np.mean((est - truth)**2)


def uniform(gamma):
    # [0, sqrt(3)]
    truth = np.random.uniform(0, 1, size=nsamples)
    truth = truth / np.sqrt(np.sum(truth**2 / nsamples))
    truePriorWeight = np.full((nsamples,), 1/nsamples)
    denoiser = NonparBayes(truth, truePriorWeight, optimizer="EM")
    x = truth * np.sqrt(gamma) + np.random.normal(size = (nsamples,))
    est = denoiser.denoise(x,np.sqrt(gamma), 1)
    return np.mean((est - truth)**2)
    

def point_normal(gamma, sparsity = 0.1):
    mask = np.random.binomial(nsamples, sparsity)
    normals = np.random.normal(size = (nsamples,))
    truth = mask * normals
    truth = truth / np.sqrt(np.sum(truth**2 / nsamples))
    truePriorWeight = np.full((nsamples,), 1/nsamples)
    denoiser = NonparBayes(truth, truePriorWeight, optimizer="EM")
    x = truth * np.sqrt(gamma) + np.random.normal(size = (nsamples,))
    est = denoiser.denoise(x,np.sqrt(gamma), 1)
    return np.mean((est - truth)**2)
    

if __name__=="__main__":
    n = 2000
    p = 1500
    rank = 1
    amp_iters = 5
    s = 1.5
    se = get_state_evolution(s, p/n, two_points, two_points, amp_iters)
    ae = get_alignment_evolution(se)