import numpy as np
from scipy.stats import multivariate_normal

from ebpca.empbayes import NonparEBActive as NonparEB 
from ebpca.empbayes import NonparEBChecker
from ebpca.amp import ebamp_gaussian_active as ebamp_gaussian
from ebpca.pca import get_pca

# from ebpca.pca import signal_solver_gaussian

def normalize_pc(U):
    return U/np.sqrt((U**2).sum(axis = 0)) * np.sqrt(len(U))

def redirect_pc(U, Ustar):
    M = U.T @ Ustar
    s = np.ones(len(M))
    for i in range(len(M)):
        if M[i,i] < 0:
            s[i] = -1
    return U * s

def get_dist_of_subspaces(U,V,rank):
    '''some random distance measurement
    U,V: ndarray (n,rank)
    '''
    Qu, _ = np.linalg.qr(U, mode = 'reduced')
    Qv, _ = np.linalg.qr(V, mode = 'reduced')
    C = Qu.T @ Qv  
    _, cos_thetas, _ = np.linalg.svd(C)
    return np.sqrt(np.mean(cos_thetas[:rank]**2))

def get_alignment(U,V):
    # normalize U and V
    U = U/np.sqrt((U**2).sum(axis = 0))
    V = V/np.sqrt((V**2).sum(axis = 0))
    COR = np.abs(np.transpose(U) @ V)
    return np.sqrt(np.mean(np.diag(COR)))

def get_MSE(U,V):
    U = U/np.sqrt((U**2).sum(axis = 0))
    V = V/np.sqrt((V**2).sum(axis = 0))
    COR = np.abs(np.transpose(U) @ V)
    return COR    

n = 1500
p = 1000
rank = 2

def test():
    # ustar = np.random.binomial(1,0.5, size = rank * n).reshape((n,rank)) * 2-1
    # vstar = np.random.binomial(1,0.5, size = rank * n).reshape((n,rank)) * 2-1
    rank = 2
    ustar = np.repeat(np.array([[1,1], [1,-1], [-1,0], [-1,0]]), int(n/4), axis = 0)
    vstar = np.repeat(np.array([[1,1], [1,-1], [-1,0], [-1,0]]), int(p/4), axis = 0)
    ustar = normalize_pc(ustar); vstar = normalize_pc(vstar)

    signals = np.array([2,1.5]) 

    W = np.random.normal(size = n * p).reshape((n,p))/ np.sqrt(p)

    X = ustar * signals @ vstar.T/p + W

    # pca part
    pcapack = get_pca(X, rank)
    print(pcapack.signals)
    print(pcapack.sample_aligns)

    print("init align")

    a = get_MSE(ustar, pcapack.U)
    print(a)

    truePriorLoc = normalize_pc(np.array([[1,-1],[1,1],[-1,0],[-1,0]]))
    
    udenoiser = NonparEBChecker(truePriorLoc, np.array([1/4,1/4,1/4,1/4]) , optimizer = "Mosek", ftol = 1e-3, nsupp_ratio = 1, to_save=True)
    vdenoiser = NonparEBChecker(truePriorLoc, np.array([1/4,1/4,1/4,1/4]), optimizer = "Mosek", ftol = 1e-3, nsupp_ratio = 1, to_save = True)

    U, _ = ebamp_gaussian(pcapack, iters=3, udenoiser=udenoiser, vdenoiser= vdenoiser, figprefix="tutorial", mutev = True)

    iters = U.shape[-1]

    res = []
    for i in range(iters):
        ans = get_alignment(U[:,:,i], ustar)
        res.append(ans)
    print(res)
    
def test1d():
    rank = 1
    ustar = np.repeat(np.array([1,-1]), int(n/2))
    ustar = ustar/np.sqrt((ustar**2).sum(axis = 0)) * np.sqrt(n)
    vstar = np.repeat(np.array([1,-1]), int(p/2))
    vstar = vstar/np.sqrt((vstar**2).sum(axis = 0)) * np.sqrt(p)

    signals = np.array([1.5]) 

    W = np.random.normal(size = n * p).reshape((n,p))/ np.sqrt(p)

    X = signals * np.outer(ustar, vstar)/p + W

    # pca part
    pcapack = get_pca(X, rank)
    print(pcapack.signals)
    print(pcapack.sample_aligns)

    a = get_MSE(ustar, pcapack.U)
    print(a)

    
    udenoiser = NonparEB(optimizer = "Mosek", ftol = 1e-3, nsupp_ratio = 1, to_save=True)
    vdenoiser = NonparEB(optimizer = "Mosek", ftol = 1e-3, nsupp_ratio = 1)

    U, _ = ebamp_gaussian(pcapack, iters=3, udenoiser=udenoiser, vdenoiser= vdenoiser, figprefix="tutorial", mutev = True)

    iters = U.shape[-1]

    res = []
    for i in range(iters):
        ans = get_alignment(U[:,:,i], ustar)
        res.append(ans)
    print(res)

# test()