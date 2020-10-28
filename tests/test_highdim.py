import numpy as np
from scipy.stats import multivariate_normal

from ebpca.empbayes import NonparEBHD
from ebpca.empbayes import NonparEBHDGD
from ebpca.amp import ebamp_gaussian_hd

# from ebpca.pca import signal_solver_gaussian

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
    COR = np.transpose(U) @ V
    return np.sqrt(np.sum(np.diag(COR))) / len(COR)



n = 2500
p = 1000
rank = 2

# signal 
# mu = np.array([[1,0], [0,1]])
def test():
    # ustar = np.random.binomial(1,0.5, size = rank * n).reshape((n,rank)) * 2-1
    # vstar = np.random.binomial(1,0.5, size = rank * n).reshape((n,rank)) * 2-1
    ustar1 = np.repeat(np.array([1,-1]), int(n/2))
    ustar2 = np.repeat(np.array([1,-1,0,0]), int(n/4))
    ustar = np.vstack((ustar1, ustar2)).T
    vstar1 = np.repeat(np.array([1,-1]), int(p/2))
    vstar2 = np.repeat(np.array([1,-1,0,0]), int(p/4))
    vstar = np.vstack((vstar1, vstar2)).T

    signals = np.array([2,1.5])

    W = np.random.normal(size = n * p).reshape((n,p))/ np.sqrt(p)

    X = ustar * signals @ vstar.T/n + W

    u, _, vh = np.linalg.svd(X)

    u_hat = u[:,:2]

    v_hat = vh[:2,:].T
    # get

    v_init_align = np.diag(v_hat.T @ vstar) / np.sqrt(np.diag(vstar.T @ vstar))

    print(v_init_align)


    udenoiser = NonparEBHDGD(ftol = 1e-3, nsupp_ratio = 1)
    vdenoiser = NonparEBHDGD(ftol = 1e-3, nsupp_ratio = 1)

    U, _ = ebamp_gaussian_hd(X, u_hat, v_hat, v_init_align, signals, iters = 2, rank = 2, udenoiser=udenoiser, vdenoiser= vdenoiser)

    iters = U.shape[-1]

    res = []
    for i in range(iters):
        ans = get_dist_of_subspaces(U[:,:,i], ustar, rank)
        res.append(ans)

    print(res)


test()