'''
==========
EBAMP
==========
finally we are here, think.
'''

import numpy as np
import scipy
from ebpca.empbayes import NonparEB
from ebpca.pca import PcaPack

def ebamp_gaussian(pcapack, iters = 5, udenoiser = NonparEB(), \
    vdenoiser = NonparEB(), figprefix = '', mutev = False):
    '''HD ebamp gaussian
    if u has shape (n, k), set iters
    return U has shape (n, k, iters+1) # with the additional init u.
    '''
    X = np.transpose(pcapack.X)
    (n,d) = X.shape
    gamma = d/n 
    k = pcapack.K
    # swap u and v
    u,v = pcapack.V, pcapack.U
    v_init_aligns = pcapack.sample_aligns
    signals = pcapack.signals
    udenoiser, vdenoiser = vdenoiser, udenoiser

    # normalize u and v
    f = u/ np.sqrt((u**2).sum(axis = 0)) * np.sqrt(n)
    g = v/np.sqrt((v**2).sum(axis = 0)) * np.sqrt(d)

    # initialize U,V 
    U = f[:,:, np.newaxis]
    V = g[:,:,np.newaxis]

    # initial corrections
    u = f * 1/(signals * np.sqrt(gamma)) * np.sqrt((signals**2*gamma + 1)/(signals**2 + 1))

    # initial states
    mu = np.diag(v_init_aligns)
    sigma_sq = np.diag(1 - v_init_aligns**2)

    for t in range(iters):
        
        # denoise right singular vector gt to get vt
        vdenoiser.fit(g, mu, sigma_sq, figname='_u_iter%02d.png' % (t))
        v = vdenoiser.denoise(g, mu, sigma_sq)
        V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
        b = gamma * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq) , axis = 0)
        
        # update left singular vector ft using vt
        f = X.dot(v) - u.dot(b.T)
        sigma_bar_sq = v.T @ v / n # non_rotation version
        mu_bar = sigma_bar_sq * signals # non_rotation version    
        
        # denoise left singular vector ft to get ut
        if not mutev:
            udenoiser.fit(f, mu_bar, sigma_bar_sq, figname='_v_iter%02d.png' % (t))
            u = udenoiser.denoise(f, mu_bar, sigma_bar_sq)
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, sigma_bar_sq), axis = 0)
            sigma_sq = u.T @ u / n
            mu = sigma_sq * signals
        
        if mutev:
            # the dernoiser is the identity map
            mu_bar_inv = np.linalg.pinv(mu_bar)
            u = f.dot(mu_bar_inv.T)
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = mu_bar_inv
            mu = np.diag(signals)
            sigma_sq = np.identity(k) + mu_bar_inv @ sigma_bar_sq @ mu_bar_inv.T
        
        # update left singular vector gt using ut
        g = np.transpose(X).dot(u) - v.dot(b_bar.T)
    
    # swap u,v
    return V, U