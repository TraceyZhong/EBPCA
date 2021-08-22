import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

from ebpca.empbayes import NonparEB
from ebpca.empbayes import NonparEBChecker
from ebpca.amp import ebamp_gaussian
from ebpca.pca import get_pca

from ebpca import ebpca_gaussian


def normalize_pc(U):
    return U/np.sqrt((U**2).sum(axis = 0)) * np.sqrt(len(U))

def redirect_pc(pc, reference):
    '''sample usage see tutorial.html
    we must redirect the truth
    redirect_pc(ustar, pcapack.U)
    '''
    U = pc 
    Ustar  = reference
    M = U.T @ Ustar
    s = np.ones(len(M))
    for i in range(len(M)):
        if M[i,i] < 0:
            s[i] = -1
    return U * s

def get_dist_of_subspaces(U,V,rank = None):
    '''some random distance measurement
    U,V: ndarray (n,rank)
    '''
    if len(U.shape) == 1:
        U = U.reshape((-1,1))
    if len(V.shape) == 1:
        V = V.reshape((-1,1))
    
    if rank is None:
        assert U.shape == V.shape, "When rank is not specified, two subspaces \
            should have the same dimension"
    else:
        U = U[:, :rank]
        V = V[:, :rank]

    Qu, _ = np.linalg.qr(U, mode = "reduced")
    Qv, _ = np.linalg.qr(V, mode = "reduced")
    _, s, _ = np.linalg.svd(Qu.T.dot(Qv))
    return np.sqrt(1 - np.min(s)**2)

def get_alignment(U,V):
    # normalize U and V
    U = U/np.sqrt((U**2).sum(axis = 0))
    V = V/np.sqrt((V**2).sum(axis = 0))
    COR = np.abs(np.transpose(U) @ V)
    return np.sqrt(np.mean(np.diag(COR**2)))

def get_MSE(U,V):
    U = U/np.sqrt((U**2).sum(axis = 0))
    V = V/np.sqrt((V**2).sum(axis = 0))
    COR = np.abs(np.transpose(U) @ V)
    return COR    

def compare_ebpca_with_svd(Ustar, Usvd, Uebpca, to_show = False, to_save = False):
    rank = Uebpca.shape[1]
    Ustar = redirect_pc(Ustar[:,:rank], Uebpca)
    if to_show or to_save:
        Usvd = normalize_pc(Usvd)
        Uebpca = normalize_pc(Uebpca)
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(7,3), sharex=True, sharey=True)
        # ground truth
        ax = axes[0]
        ax.scatter(Ustar[:,0], Ustar[:,1], s = 5)
        ax.set_title("Ground Truth")
        # svd
        ax = axes[1]
        ax.scatter(Usvd[:,0], Usvd[:,1], s = 5)
        error = get_dist_of_subspaces(Ustar, Usvd, rank)
        ax.set_title("SVD\nError = {:.4f}".format(error))
        PC0_error = get_dist_of_subspaces(Ustar[:,0], Usvd[:,0])
        ax.set_xlabel("PC1 Error={:.4f}".format(PC0_error))
        PC1_error = get_dist_of_subspaces(Ustar[:,1], Usvd[:,1])
        ax.set_ylabel("PC2 Error={:.4f}".format(PC1_error))
        # ebpca
        ax = axes[2]
        ax.scatter(Uebpca[:,0], Uebpca[:,1], s = 5)
        error = get_dist_of_subspaces(Ustar, Uebpca, rank)
        ax.set_title("EBPCA\nError = {:.4f}".format(error))
        PC0_error = get_dist_of_subspaces(Ustar[:,0], Uebpca[:,0])
        ax.set_xlabel("PC1 Error={:.4f}".format(PC0_error))
        PC1_error = get_dist_of_subspaces(Ustar[:,1], Uebpca[:,1])
        ax.set_ylabel("PC2 Error={:.4f}".format(PC1_error))
    if to_show:
        plt.show()
    if to_save:
        plt.savefig('figures/ebpca_with_svd.png')


def compare_with_truth(Ustar, U, to_show = False, to_save = False):
    if to_show or to_save:
        fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize=(10,3), sharex=True, sharey=True)
        # ground truth
        ax = axes[0]
        ax.scatter(Ustar[:,0], Ustar[:,1], s = 5)
        ax.set_title("Ground Truth")
        # svd
        ax = axes[1]
        ax.scatter(U[:,0,0], U[:,1,0], s=  5)
        alignment = get_dist_of_subspaces(Ustar, U[:,:,0])
        ax.set_title("SVD\nError = {:.4f}".format(alignment))
        # first denoising 
        ax = axes[2]
        ax.scatter(U[:,0,1], U[:,1,1], s = 5)
        alignment = get_dist_of_subspaces(Ustar, U[:,:,1])
        ax.set_title("First iteration\nError = {:.4f}".format(alignment))
        # last result
        ax = axes[3]
        ax.scatter(U[:,0,-1] , U[:,1,-1], s = 5)
        alignment = get_dist_of_subspaces(Ustar, U[:,:,-1])
        ax.set_title("Last iteration\nError = {:.4f}".format(alignment))
    if to_show:
        plt.show()
    if to_save:
        plt.savefig('figures/compare_with_truth.png')


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
    
    udenoiser = NonparEBChecker(ustar, optimizer = "EM", nsupp_ratio = 1, to_save=False)
    vdenoiser = NonparEBChecker(vstar, optimizer = "Mosek", nsupp_ratio = 1, to_save = False)

    U, _ = ebamp_gaussian(pcapack, amp_iters=3, udenoiser=udenoiser, vdenoiser= vdenoiser, \
        muteu = False, warm_start = True)

    iters = U.shape[-1]

    res = []
    for i in range(iters):
        ans = get_dist_of_subspaces(U[:,:,i], ustar)
        res.append(ans)
    print("Sample result of error across iterations. \nUsers are expected to observe a decreasing sequence of error")
    print(res)
    
def simulate_structured_PC(n, k=3):
    assert k == 3
    U = np.array([[1,3,2],[1,-2,0],[-1,2,1],[-1,-1,-1]])
    Qu, _ = np.linalg.qr(U)
    Ustar = np.repeat(Qu, int(n/4), axis = 0)
    return normalize_pc(Ustar)

def simulate_unstructured_PC(n, k =3):
    assert k==3
    p = n
    Vstar = np.random.multivariate_normal(mean = np.array([0,0,0]), cov = np.array([[1,0,0],[0,1,0],[0,0,1]]), size = int(p))
    return normalize_pc(Vstar)


def simulate_signal_plus_noise_model():
    n = 800
    p = 1000

    # signal part
    U = np.array([[1,3,2],[1,-2,0],[-1,2,1],[-1,-1,-1]])
    Qu, _ = np.linalg.qr(U)
    Ustar = np.repeat(Qu, int(n/4), axis = 0)
    Vstar = np.random.multivariate_normal(mean = np.array([0,0,0]), cov = np.array([[1,0,0],[0,1,0],[0,0,1]]), size = int(p))
    Ustar = normalize_pc(Ustar); Vstar = normalize_pc(Vstar)
    signals = [2, 1.5, 0.2]

    # noise part
    tau = 0.9
    W = np.random.normal(size = n * p).reshape((n,p)) / np.sqrt(n) * tau

    # observational matrix
    Y = Ustar * signals @ Vstar.T/n + W

    return Y 

    
if __name__=="__main__":
    test()
    print("Finish test.")


