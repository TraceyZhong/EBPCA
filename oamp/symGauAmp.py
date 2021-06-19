import numpy as np 
from utils import plot_save


def gaussian_amp_pca(Y, theta, singval, innerprod, fsvd, ustar = None, iters=5, reg = 0.01, kappas = None, **kwargs):
    # remember to renormalize data to match kappa
    noise_type= kwargs["noise_type"]
    
    n = Y.shape[0]
    
    U = np.zeros((n,iters+1)) #(u_0, ..., u_T)
    U[:,0] = fsvd/theta

    mu = np.array(innerprod)
    sigmasq = 1 - mu**2
    plot_save(fsvd,mu, sigmasq, 0, "gamp" + noise_type)
    # States

    f = fsvd 

    for t in range(1, iters + 1):
        print("at iter", t)
        sinvmu = mu/sigmasq
        plot_save(f, mu, sigmasq, t , "gamp" + noise_type)
        u = np.tanh(f * sinvmu)
        U[:,t] = u
        Euprod = np.mean(u**2)
        # b = gradu
        b = sinvmu*(1- Euprod)
        f = Y.dot(u) - b * U[:, t-1] 
        # update sigma and mu 
        sigmasq = Euprod 
        mu = np.sqrt(max(np.mean(f**2)-sigmasq, reg))
    
    return U 




