import config

import numpy as np

from ebpca.amp import ebamp_gaussian
from ebpca.pca import signal_solver_gaussian
from ebpca.empbayes import NonparEB
from ebpca.empbayes import TestEB


np.random.seed(1921)

niters = 5

m = 900
n = 1000

def get_alignment(u,v):
    # consider u and v are one dimmensional
    u = u.flatten()
    v = v.flatten()
    return u.dot(v)/(np.linalg.norm(u) * np.linalg.norm(v))


def test(W, alpha):
    ustar = np.random.binomial(1,0.5,size=m)*2-1
    vstar = np.random.binomial(1,0.5,size=n)*2-1
    X = alpha/n * np.outer(ustar,vstar) + W

    u, s, vh = np.linalg.svd(X,full_matrices=False)
    u = u[:,0]
    v = vh[0,:]

    res = signal_solver_gaussian(singval = s[0], n_samples= m, n_features= n)
    estimates = res 
    print("alpha [%.4f, %.4f], sample align = [%.4f, %.4f], feature align = [%.4f, %.4f]" % (alpha, estimates["alpha"], 
            get_alignment(u, ustar), estimates["sample_align"], get_alignment(v, vstar), estimates["feature_align"] ))

    udenoiser = TestEB(to_save = True, to_show = False, fig_prefix = "nonpareb_u_")
    vdenoiser = TestEB(to_save = True, to_show = False, fig_prefix = "nonpareb_v_")

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