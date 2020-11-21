'''
==========
AMP
==========
'''

import numpy as np
import scipy

from ebpca.empbayes import NonparEB
from ebpca.pca import PcaPack


def ebamp_gaussian(pcapack, iters = 5, udenoiser = NonparEB(), \
    vdenoiser = NonparEB(), figprefix = '', muteu = False):
    '''HD ebamp gaussian
    if u has shape (n, k), set iters
    return U has shape (n, k, iters+1) # with the additional init u.
    '''
    X = pcapack.X
    (n,d) = X.shape
    gamma = d/n 
    k = pcapack.K
    # swap u and v
    u,v = pcapack.U, pcapack.V
    v_init_aligns = pcapack.feature_aligns
    signals = pcapack.signals
    # udenoiser, vdenoiser = vdenoiser, udenoiser

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

    for t in range(iters):

        print('iteration %i' % t)
        # denoise right singular vector gt to get vt
        npmle_status = vdenoiser.fit(g, mu, sigma_sq, figname='_v_iter%02d.png' % (t))
        if npmle_status == 'error':
            print('EB-PCA terminated.')
            break
        v = vdenoiser.denoise(g, mu, sigma_sq)
        V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
        b = gamma * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq) , axis = 0)
        
        # update left singular vector ft using vt
        f = X.dot(v) - u.dot(b.T)
        sigma_bar_sq = v.T @ v / n # non_rotation version
        mu_bar = sigma_bar_sq * signals # non_rotation version    
        
        # denoise left singular vector ft to get ut
        if not muteu:
            npmle_status = udenoiser.fit(f, mu_bar, sigma_bar_sq, figname='_u_iter%02d.png' % (t))
            if npmle_status == 'error':
                print('EB-PCA terminated.')
                break
            u = udenoiser.denoise(f, mu_bar, sigma_bar_sq)
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, sigma_bar_sq), axis = 0)
            sigma_sq = u.T @ u / n
            mu = sigma_sq * signals
        
        if muteu:
            # the dernoiser is the identity map
            mu_bar_inv = np.linalg.pinv(mu_bar)
            u = f.dot(mu_bar_inv.T)
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = mu_bar_inv
            mu = np.diag(signals)
            sigma_sq = np.identity(k) + mu_bar_inv @ sigma_bar_sq @ mu_bar_inv.T
        
        # update left singular vector gt using ut
        g = np.transpose(X).dot(u) - v.dot(b_bar.T)
    
    # don't swap u,v
    return U,V

def ebamp_gaussian_rank_one(pcapack, iters = 5, udenoiser = NonparEB(), \
    vdenoiser = NonparEB(), figprefix = '', muteu = False):

    assert pcapack.K == 1, "For rank one model you must set rank to be one"

    X = pcapack.X
    (n,d) = X.shape 
    alpha = d/n 
    u = pcapack.U[:,0] 
    v = pcapack.V[:,0] 

    v_init_align = pcapack.feature_aligns[0] 
    signal = pcapack.signals[0] 

    # normalize u, v
    f = u / np.sqrt(np.sum(u**2)) * np.sqrt(n)
    g = v / np.sqrt(np.sum(v**2)) * np.sqrt(d)

    U = np.reshape(f,(-1,1))
    V = np.reshape(g,(-1,1))

    mu = v_init_align
    sigma_sq = 1 - mu**2

    u = f * np.sqrt(sigma_sq)

    for t in range(iters):
        # print("at amp iter {}".format(t))
        # denoise right singular vector gt to get vt
        vdenoiser.fit(g, mu, sigma_sq, figname=figprefix+'_v_iter%02d_' % (t))
        v = vdenoiser.denoise(g, mu, sigma_sq)
        V = np.hstack((V,np.reshape(v,(-1,1))))
        b = alpha * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq))
        # update left singular vector ft using vt
        f = X.dot(v) - b*u
        sigma_bar_sq = alpha * np.mean(v**2)
        mu_bar = np.sqrt(np.mean(f**2) - sigma_bar_sq)
        if not muteu:
        # denoise left singular vector ft to get ut
            udenoiser.fit(f, mu_bar, np.sqrt(sigma_bar_sq), figname='_u_iter%02d_' % (t))
            u = udenoiser.denoise(f, mu_bar, np.sqrt(sigma_bar_sq))
            U = np.hstack((U, np.reshape(u, (-1,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, np.sqrt(sigma_bar_sq)))
            sigma_sq = np.mean(u**2)
            mu = np.sqrt(np.mean(g**2) - sigma_sq)
        if muteu:
            u = f / mu_bar
            U = np.hstack((U, np.reshape(u, (-1,1))))
            b_bar = 1 / mu_bar
            mu = signal 
            sigma_sq = 1 + sigma_bar_sq / (mu**2)
        # update right singular vector gt using ut
        g = np.transpose(X).dot(u) - b_bar * v
    # return U,V, don't need to swap them
    return U,V
