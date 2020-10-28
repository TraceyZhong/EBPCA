import config 

import numpy as np
# pca 
from ebpca.pca import signal_solver_gaussian
# amp 
from ebpca.amp import ebamp_gaussian_hd_no_rotation
from ebpca.amp import ebamp_gaussian_hd
# empbayes
from ebpca.empbayes import TestEBHD

def get_dist_of_subspaces(U,V,rank):
    '''some random distance measurement
    U,V: ndarray (n,rank)
    '''
    Qu, _ = np.linalg.qr(U, mode = 'reduced')
    Qv, _ = np.linalg.qr(V, mode = 'reduced')
    C = Qu.T @ Qv  
    _, cos_thetas, _ = np.linalg.svd(C)
    return np.sqrt(np.mean(cos_thetas[:rank]**2))

def simulate(signals, m,n):
    # suppose for now we have only 2 dim subspace to study.
    # return Y matrix (n,p), noise level 1/n
    ustar = np.repeat(np.array([
        [1,1],
        [1,-1],
        [-1,1],
        [-1,-1]
    ]), repeats = int(m/4), axis = 0)
    vstar = np.repeat(np.array([
        [1,1],
        [1,-1],
        [-1,1],
        [-1,-1]
    ]), repeats = int(n/4), axis = 0)
    W = np.random.normal(size = (m,n))
    Y = (ustar * signals) @ vstar.T/n + W / np.sqrt(n)
    return Y

def test_noRotation(signals,m,n, withRotation = False, rank = 2):
    ustar = np.repeat(np.array([
        [1,1],
        [1,-1],
        [-1,1],
        [-1,-1]
    ]), repeats = int(m/4), axis = 0)
    vstar = np.repeat(np.array([
        [1,1],
        [1,-1],
        [-1,1],
        [-1,-1]
    ]), repeats = int(n/4), axis = 0)
    W = np.random.normal(size = (m,n))
    Y = (ustar * signals) @ vstar.T/n + W / np.sqrt(n)

    F, S, Gh = np.linalg.svd(Y)

    lowRanks = [signal_solver_gaussian(S[i], None, m,n) for i in [0,1]]
    init_aligns = np.array([sol["sample_align"] for sol in lowRanks]).reshape((2,))
    est_signals = np.array([sol["alpha"] for sol in lowRanks]).reshape((2,))
    print(init_aligns)
    
    denoiser = TestEBHD(fig_prefix="testebhd")
    # with rotation
    U, _ = ebamp_gaussian_hd(Y, F[:,:2], Gh[:2,:].T, init_aligns, est_signals, udenoiser = denoiser, vdenoiser = denoiser)
    
    iters = U.shape[-1]
    res = []
    for i in range(iters):
        ans = get_dist_of_subspaces(U[:,:,i], ustar, rank)
        res.append(ans)
    resRotated = res

    # without rotation
    U, _ = ebamp_gaussian_hd_no_rotation(Y, F[:,:2], Gh[:2,:].T, init_aligns, est_signals, udenoiser = denoiser, vdenoiser = denoiser)
    iters = U.shape[-1]
    resNormal = []
    for i in range(iters):
        ans = get_dist_of_subspaces(U[:,:,i], ustar, rank)
        resNormal.append(ans)

    return [resRotated, resNormal]
    
signals = [1.8, 1.5]
m = n = 1000
ress = test_noRotation(signals,m,n, withRotation = False, rank = 2)
print("Signal Strength")
print(signals)
print("No Rotation")
print(ress[1])
print("With Rotation")
print(ress[0])