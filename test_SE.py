# test if alignment function is working

from simulation.helpers import approx_prior

from simulation.helpers import signal_plus_noise_model

from ebpca.state_evolution import get_state_evolution, get_alignment_evolution
from ebpca.state_evolution import two_points, uniform, point_normal
from ebpca.preprocessing import normalize_obs
from ebpca.pca import get_pca
from ebpca.empbayes import NonparBayes
from ebpca.amp import ebamp_gaussian

from tutorial import get_alignment

import numpy as np

n = 2000
p = 1500
rank = 1
amp_iters = 5
sparsity = 0.1

literals =["Uniform", "Two_points", "Point_normal"]
mmse_funcs = {
    "Uniform": uniform,
    "Two_points":  two_points,
    "Point_normal": point_normal
}

def simulate_prior(prior, n=2000, seed=1, rank=1):
    '''
    simulate univariate or bivariate distributions with marginal 2nd moment=1
    Univariate distributions:
      Uniform: represents continuous prior
      Two_points: represents degenerative prior / cluster structure
      Point_normal: represents sparse prior
    Bivariate distributions:
      Uniform_circle: represents continuous bivariate prior
      Three_points: represent bivariate cluster structure
    '''
    np.random.seed(seed)
    nsamples = n
    if rank == 1:
        if prior == 'Uniform':
            truth = np.random.uniform(0, 1, size=(nsamples,))
            theta = truth / np.sqrt(np.sum(truth**2) / nsamples)
        if prior == 'Two_points':
            theta = 2 * np.random.binomial(n=1, p=0.5, size=n) - 1
        if prior == 'Point_normal':
            mask = np.random.binomial(1, sparsity, size = (nsamples,))
            normals = np.random.normal(size = (nsamples,))
            truth = mask * normals
            theta = truth / np.sqrt(np.sum(truth**2) / nsamples)
    return theta

def simulate_obs(s, uliteral, vliteral, seed):
    ustar = simulate_prior(uliteral, n, seed)
    vstar = simulate_prior(vliteral, p, seed)

    X = signal_plus_noise_model(ustar, vstar, s, 1)

    X = normalize_obs(X, rank)
    pcapack = get_pca(X, rank)
    print("feature aligns are", pcapack.feature_aligns)
    [truePriorLoc, truePriorWeight] = approx_prior(ustar, pcapack.U)
    udenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)
    [truePriorLoc, truePriorWeight] = approx_prior(vstar, pcapack.V)
    vdenoiser = NonparBayes(truePriorLoc, truePriorWeight, to_save=False)

    U_est, V_est = ebamp_gaussian(pcapack, iters=amp_iters,\
        udenoiser=udenoiser, vdenoiser=vdenoiser)
    
    ualigns = get_alignments(U_est, ustar)
    valigns = get_alignments(V_est, vstar)
    
    return ualigns, valigns

def repeat_simulation(s, uliteral, vliteral, sim_rep):
    Ureps = np.empty(shape = (amp_iters, sim_rep))
    Vreps = np.empty(shape = (amp_iters, sim_rep))
    for i in range(sim_rep):
        ualigns, valigns = simulate_obs(s, uliteral, vliteral, i)
        Ureps[:,i] = ualigns
        Vreps[:,i] = valigns
        print("Finish sim_res ", i)
    np.save("output/Ualigns_%s.npy" % uliteral, Ureps)
    np.save("output/Valigns_%s.npy" % vliteral, Vreps)

def get_alignments(Uests, ustar):
    alignments = np.empty(amp_iters)
    for i in range(amp_iters):
        al = get_alignment(Uests[:,:,i], ustar)
        alignments[i] = abs(al)
    return alignments


if __name__ == "__main__":
    uliteral = "Point_normal"
    vliteral = "Point_normal"
    s = 1.5
    ummse = mmse_funcs[uliteral]
    vmmse = mmse_funcs[vliteral]
    repeat_simulation(s, uliteral, vliteral, 5)
    ualigns = np.load("output/Ualigns_%s.npy" % uliteral)
    valigns = np.load("output/Valigns_%s.npy" % vliteral)
    
    se = get_state_evolution(s, p/n, ummse, vmmse, amp_iters)
    print(se)
    ae = get_alignment_evolution(se)
    
    print("ualigns means are:")
    print(ualigns.mean(axis = 1))
    print("ualigns evolutions are:")
    print(ae.ualigns[:amp_iters])

    print("valigns means are:")
    print(valigns.mean(axis = 1))
    print("valigns evolutions are:")
    print(ae.valigns[:amp_iters])





