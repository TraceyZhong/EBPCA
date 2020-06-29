'''
==========
AMP
==========
'''

import numpy as np

from .empbayes import NonparEB
from .pca import PcaPack

def ebamp_gaussian(pcapack, amp_iters = 5, udenoiser = NonparEB(), vdenoiser = NonparEB(), \
    warm_start = True, muteu = False, mutev = False):
    '''Gaussian Bayes AMP
    Parameters
    -------
    Except for the self explanatory ones
    warm_start: boolean
        Only estimate prior once at the init step if set to be True 
    muteu, mutev: boolean
        Use the identity map as the denoiser for that direction if set to be True
    
    Returns
    -------
    U: ndarray of shape (n, rank, amp_iters + 1)
    V: ndarray of shape (p, rank, amp_iters + 1)
    '''
    X = pcapack.X
    (n,d) = X.shape
    gamma = d/n 
    k = pcapack.K
    u,v = pcapack.U, pcapack.V
    v_init_aligns = pcapack.feature_aligns
    signals = pcapack.signals

    # normalize u and v
    f = u/np.sqrt((u**2).sum(axis = 0)) * np.sqrt(n)
    g = v/np.sqrt((v**2).sum(axis = 0)) * np.sqrt(d)

    # initialize U,V 
    U = f[:,:, np.newaxis]
    V = g[:,:,np.newaxis]

    # initial states
    mu = np.diag(v_init_aligns)
    sigma_sq = np.diag(1 - v_init_aligns**2)

    # initial corrections
    u = f @ np.sqrt(sigma_sq)

    for t in range(amp_iters):
        if not mutev:
            # denoise right singular vector gt to get vt
            if ((not warm_start) or t == 0):
                vdenoiser.fit(g, mu, sigma_sq, figname='_v_iter%02d.png' % (t))
            v = vdenoiser.denoise(g, mu, sigma_sq)
            V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
            b = gamma * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq) , axis = 0)
            
            sigma_bar_sq = v.T @ v / n 
            mu_bar = sigma_bar_sq * signals    
        
        else: 
            # the linear denoiser
            mu_inv = np.linalg.pinv(mu)
            v = g.dot(mu_inv.T)
            V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
            b = mu_inv * gamma

            mu_bar = np.diag(signals) * gamma
            sigma_bar_sq = (np.identity(k) + mu_inv @ sigma_sq @ mu_inv.T) * gamma

        # update right singular vector ft using vt
        f = X.dot(v) - u.dot(b.T)
        
        # denoise left singular vector ft to get ut
        if not muteu:
            if ((not warm_start) or t == 0):
                udenoiser.fit(f, mu_bar, sigma_bar_sq, figname='_u_iter%02d.png' % (t))
            u = udenoiser.denoise(f, mu_bar, sigma_bar_sq)
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, sigma_bar_sq), axis = 0)
            sigma_sq = u.T @ u / n
            mu = sigma_sq * signals
        
        else:
            # the denoiser is the identity map
            mu_bar_inv = np.linalg.pinv(mu_bar)
            u = f.dot(mu_bar_inv.T)
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = mu_bar_inv
            mu = np.diag(signals)
            sigma_sq = np.identity(k) + mu_bar_inv @ sigma_bar_sq @ mu_bar_inv.T
        
        # update left singular vector gt using ut
        g = np.transpose(X).dot(u) - v.dot(b_bar.T)
    
    return U, V