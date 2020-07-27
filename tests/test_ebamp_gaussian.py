import numpy as np

from ebpca.amp import ebamp_gaussian
from ebpca.pca import signal_solver_gaussian
from ebpca.empbayes import NonparEB
from ebpca.empbayes import TestEB

np.random.seed(8921)

niters = 5

m = 1000
n = 1000

def test(W, alpha):
    ustar = np.random.binomial(1,0.5,size=m)*2-1
    vstar = np.random.binomial(1,0.5,size=n)*2-1
    X = alpha/n * np.outer(ustar,vstar) + W

    u, s, vh = np.linalg.svd(X,full_matrices=False)
    u = u[:,0]
    v = vh[0,:]

    res = signal_solver_gaussian(singval = s[0], n_samples= m, n_features= n)

    udenoiser = NonparEB(em_iter = 500, to_save = True, to_show = False, fig_prefix = "nonpareb_u_")
    vdenoiser = NonparEB(em_iter = 500, to_save = True, to_show = False, fig_prefix = "nonpareb_v_")

    (U,V) = ebamp_gaussian(X, u, init_align= res["sample_align"], 
        v = v, v_init_align= res["feature_align"], signal= res["alpha"],
        iters = 5, rank = 1,
        udenoiser= udenoiser, vdenoiser= vdenoiser, 
        )

    Unormsq = np.diag(np.transpose(U).dot(U))
    Vnormsq = np.diag(np.transpose(V).dot(V))
    Ualigns = np.transpose(U).dot(ustar) / np.sqrt(Unormsq*m)
    Valigns = np.transpose(V).dot(vstar) / np.sqrt(Vnormsq*n)
    print(Ualigns)
    print(Valigns)

alpha = 1.8
print("Gaussian noise, alpha = {}".format(alpha))
W = np.random.normal(size=(m,n))/np.sqrt(n)
test(W, alpha)