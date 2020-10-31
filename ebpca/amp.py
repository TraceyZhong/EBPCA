'''
==========
EBAMP
==========
finally we are here, think.
'''
# TODO use only one 

import numpy as np
import scipy
from ebpca.empbayes import NonparEB
from ebpca.pca import PcaPack

# def ebamp_gaussian_hd_no_rotation(X, u, v, init_aligns, signals, iters = 5, rank = 2, udenoiser = NonparEBHD(), vdenoiser = NonparEBHD(), mutev = False ):
def ebamp_gaussian_active(pcapack, iters = 5, udenoiser = NonparEB(), \
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
    print("norm f {}".format(np.sum(f**2)))
    g = v/np.sqrt((v**2).sum(axis = 0)) * np.sqrt(d)

    # initialize U,V TODO new axis
    U = f[:,:, np.newaxis]
    V = g[:,:,np.newaxis]
    # k4 = np.sqrt((l**2*alpha + 1)/(l**2 + 1)) / (l*np.sqrt(alpha)) # 
    u = f * 1/(signals * np.sqrt(gamma)) * np.sqrt((signals**2*gamma + 1)/(signals**2 + 1))
    print("norm u_init {}".format(np.sum(u**2)))

    # initial states TODO
    mu = np.diag(v_init_aligns)
    sigma_sq = np.diag(1 - v_init_aligns**2)
    # initial correction TODO 

    for t in range(iters):
        # print("at amp iter {}".format(t))
        # denoise right singular vector gt to get vt
        # print("before denoise, g shape {}".format(g.shape))
        vdenoiser.fit(g, mu, sigma_sq, figname='_u_iter%02d.png' % (t))
        print("mu={}, simg_sq={}".format(mu, sigma_sq))
        # print("finish fit")
        v = vdenoiser.denoise(g, mu, sigma_sq)
        # print("v shape {}".format(v.shape))
        # print("finish denoise")
        V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
        b = gamma * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq) , axis = 0)
        # print("I want to look at b")
        # print(b)
        # print("finish ddenoise")
        # update left singular vector ft using vt
        f = X.dot(v) - u.dot(b.T)
        print("X shape {}".format(X.shape))
        print("v shape {}".format(v.shape))
        print("Xv {}".format(np.sum((X.dot(v))**2)))
        print("ub {}".format(np.sum((u.dot(b.T))**2)))
        print("X norm {}".format(np.sum(X**2)))
        print("f norm {}".format(np.sum(f**2)))
        print("v norm {}".format(np.sum(v**2)))
        print("u norm {}".format(np.sum(u**2)))
        print("b norm {}".format(np.sum(b**2)))
        sigma_bar_sq = v.T @ v / n # non_rotation version
        # tmp
        mu_bar = scipy.linalg.sqrtm(np.mean(f**2) - sigma_bar_sq)
        print("fz mubar {}".format(mu_bar))
        mu_bar = sigma_bar_sq * signals # non_rotation version    
        print("mubar {}".format(mu_bar))
        # denoise left singular vector ft to get ut
        print("step {}".format(t))
        print("mubar: {}".format(mu_bar))
        print("sigmabarsq: {}".format(sigma_bar_sq))
        if not mutev:
            # print("before denoise, f shape {}".format(f.shape))
            udenoiser.fit(f, mu_bar, sigma_bar_sq, figname='_v_iter%02d.png' % (t))
            # print("finish fit")
            u = udenoiser.denoise(f, mu_bar, sigma_bar_sq)
            # print("finish denoise")
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, sigma_bar_sq), axis = 0)
            # print("This is b_bar")
            # print(b_bar)
            # print("finish ddenoise")
            sigma_sq = u.T @ u / n
            mu = sigma_sq * signals # non_rotation version
        if mutev:
            # in this case, u is f
            # u = f
            mu_bar_inv = np.linalg.pinv(mu_bar)
            u = f.dot(mu_bar_inv.T)
            # u fit pass
            # u denoise pass 
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            # u ddenoise
            # b_bar = np.identity(k) 
            b_bar = mu_bar_inv
            mu = np.diag(signals)
            sigma_sq = np.identity(k) + mu_bar_inv @ sigma_bar_sq @ mu_bar_inv.T
        # update left singular vector gt using ut
        # print("X shape={}, u shape={}, v shape = {} b_bar shape ={}".format(X.shape, u.shape, v.shape, b_bar.shape))
        g = np.transpose(X).dot(u) - v.dot(b_bar.T)
    # swap u,v
    return V, U

def ebamp_gaussian_active2(pcapack, iters = 5, udenoiser = NonparEB(), \
    vdenoiser = NonparEB(), figprefix = '', mutev = False):
    
    X = np.transpose(pcapack.X)
    v = pcapack.U # this is the sample direction
    u = pcapack.V # this is the feature direction
    k = pcapack.K # does this K matters?
    signals = pcapack.signals
    v_init_aligns = pcapack.sample_aligns
    n,d = X.shape
    gamma = d/n

    # normalize u and v
    f = u/np.sqrt((u**2).sum(axis = 0)) * np.sqrt(n)
    g = v/np.sqrt((v**2).sum(axis = 0)) * np.sqrt(d)

    # u^{-1}
    u = f * 1/(signals * np.sqrt(gamma)) * np.sqrt((signals**2*gamma + 1)/(signals**2 + 1))

    # initialize U,V
    U = f[:,:, np.newaxis]
    V = g[:,:,np.newaxis]

    mu = np.diag(v_init_aligns)
    sigma_sq = np.diag(1 - v_init_aligns**2)

    for t in range(iters):
        # print("at amp iter {}".format(t))
        # denoise right singular vector gt to get vt
        # print("before denoise, g shape {}".format(g.shape))
        vdenoiser.fit(g, mu, sigma_sq, figname='_u_iter%02d.png' % (t))
        # print("finish fit")
        v = vdenoiser.denoise(g, mu, sigma_sq)
        # print("v shape {}".format(v.shape))
        # print("finish denoise")
        V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
        b = gamma * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq) , axis = 0)
        # print("finshi ddenoise")
        # update left singular vector ft using vt
        f = X.dot(v) - u.dot(b)
        sigma_bar_sq = v.T @ v / n
        mu_bar = scipy.linalg.sqrtm(f.T @ f / n - sigma_bar_sq)        
        # denoise left singular vector ft to get ut
        if not mutev:
            # print("before denoise, f shape {}".format(f.shape))
            udenoiser.fit(f, mu_bar, sigma_bar_sq, figname='_v_iter%02d.png' % (t))
            # print("finish fit")
            u = udenoiser.denoise(f, mu_bar, sigma_bar_sq)
            # print("finish denoise")
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, sigma_bar_sq), axis = 0)
            # print("finish ddenoise")
        if mutev:
            # in this case, u is the derived f
            u = f
            # u fit pass
            # u denoise pass 
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            # u ddenoise
            b_bar = np.identity(2)
        # update left singular vector gt using ut
        g = np.transpose(X).dot(u) - v.dot(b_bar)
        sigma_sq = u.T @ u / n
        mu = scipy.linalg.sqrtm(g.T @ g / d - sigma_sq)
    # swap u,v
    return V, U



def ebamp_gaussian(X, u, v, init_pars, iters = 5,
                   udenoiser = NonparEB(), vdenoiser = NonparEB(),
                   figprefix='', mutev = False):
    # get input paramaters
    init_align = init_pars["sample_align"][0]
    signal = init_pars["alpha"][0]

    # Normalize u, v and initialize U V
    # n is the direction of features, must be the inverse scale of the variance.
    X = np.transpose(X) # shape becomes A = n*d
    (n,d) = X.shape
    alpha = d/n

    # swap u and v
    tmp = u
    u = v
    v = tmp
    v_init_align = init_align

    U = np.reshape(u,(-1,1))
    V = np.reshape(v,(-1,1))

    u = 1/(signal * alpha) * np.sqrt((signal**2*alpha + 1)/(signal**2 + 1)) * u / np.linalg.norm(u) * np.sqrt(n)
    g = v/np.linalg.norm(v) * np.sqrt(d)

    mu = v_init_align
    sigma_sq = 1 - mu**2
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
        if not mutev:
        # denoise left singular vector ft to get ut
            udenoiser.fit(f, mu_bar, np.sqrt(sigma_bar_sq), figname='_u_iter%02d_' % (t))
            u = udenoiser.denoise(f, mu_bar, np.sqrt(sigma_bar_sq))
            U = np.hstack((U, np.reshape(u, (-1,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, np.sqrt(sigma_bar_sq)))
        if mutev:
            u = f
            U = np.hstack((U, np.reshape(u, (-1,1))))
            b_bar = 1
        # update right singular vector gt using ut
        g = np.transpose(X).dot(u) - b_bar * v
        sigma_sq = np.mean(u**2)
        mu = np.sqrt(np.mean(g**2) - sigma_sq)
    # return U,V, need to change direction back
    return V,U

def ebamp_gaussian_hd(X, u, v, init_pars, iters = 5, rank = 2,
                      udenoiser = NonparEB(), vdenoiser = NonparEB(),
                      figprefix='', mutev = False):
    '''HD ebamp gaussian
    if u has shape (n, k), set iters
    return U has shape (n, k, iters+1) # with the additional init u.
    '''

    init_aligns = [init_pars[i]['sample_align'][0] for i in range(rank)]
    signals = [init_pars[i]['alpha'][0] for i in range(rank)]

    X = np.transpose(X)
    (n,d) = X.shape
    gamma = d/n 
    k = rank 
    # swap u and v
    u,v = v,u
    v_init_aligns = init_aligns

    # normalize u and v
    f = u/ np.sqrt((u**2).sum(axis = 0)) * np.sqrt(n)
    g = v/np.sqrt((v**2).sum(axis = 0)) * np.sqrt(d)

    # initialize U,V TODO new axis
    U = f[:,:, np.newaxis]
    V = g[:,:,np.newaxis]

    u = f * 1/(signals * np.sqrt(gamma)) * np.sqrt((signals**2*gamma + 1)/(signals**2 + 1))

    # initial states TODO
    mu = np.diag(v_init_aligns)
    sigma_sq = np.diag(1 - v_init_aligns**2)
    # initial correction TODO 

    for t in range(iters):
        # print("at amp iter {}".format(t))
        # denoise right singular vector gt to get vt
        # print("before denoise, g shape {}".format(g.shape))
        vdenoiser.fit(g, mu, sigma_sq, figname='_u_iter%02d.png' % (t))
        # print("finish fit")
        v = vdenoiser.denoise(g, mu, sigma_sq)
        # print("v shape {}".format(v.shape))
        # print("finish denoise")
        V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
        b = gamma * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq) , axis = 0)
        # print("finshi ddenoise")
        # update left singular vector ft using vt
        f = X.dot(v) - u.dot(b)
        sigma_bar_sq = v.T @ v / n
        mu_bar = scipy.linalg.sqrtm(f.T @ f / n - sigma_bar_sq)        
        # denoise left singular vector ft to get ut
        if not mutev:
            # print("before denoise, f shape {}".format(f.shape))
            udenoiser.fit(f, mu_bar, sigma_bar_sq, figname='_v_iter%02d.png' % (t))
            # print("finish fit")
            u = udenoiser.denoise(f, mu_bar, sigma_bar_sq)
            # print("finish denoise")
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, sigma_bar_sq), axis = 0)
            # print("finish ddenoise")
            sigma_sq = u.T @ u / n # non_rotation version
            mu = sigma_sq * signals # non_rotation version
        if mutev:
            # in this case, u is the derived f
            u = f
            # u fit pass
            # u denoise pass 
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            # u ddenoise
            b_bar = np.identity(2)
        # update left singular vector gt using ut
        g = np.transpose(X).dot(u) - v.dot(b_bar)
        sigma_sq = u.T @ u / n
        mu = scipy.linalg.sqrtm(g.T @ g / d - sigma_sq)
    # swap u,v
    return V, U

def ebamp_gaussian_hd_hd(X, u, v, init_aligns, signals, iters = 5, rank = 2, udenoiser = NonparEB(), vdenoiser = NonparEB(), mutev = False ):
    '''HD ebamp gaussian
    if u has shape (n, k), set iters
    return U has shape (n, k, iters+1) # with the additional init u.
    '''

    X = np.transpose(X)
    (n,d) = X.shape
    gamma = d/n 
    k = rank 
    # swap u and v
    u,v = v,u
    v_init_aligns = init_aligns

    # normalize u and v
    f = u/ np.sqrt((u**2).sum(axis = 0)) * np.sqrt(n)
    g = v/np.sqrt((v**2).sum(axis = 0)) * np.sqrt(d)

    # initialize U,V TODO new axis
    U = f[:,:, np.newaxis]
    V = g[:,:,np.newaxis]

    u = f * 1/(signals * np.sqrt(gamma)) * np.sqrt((signals**2*gamma + 1)/(signals**2 + 1))

    # initial states TODO
    mu = np.diag(v_init_aligns)
    sigma_sq = np.diag(1 - v_init_aligns**2)
    # initial correction TODO 

    for t in range(iters):
        # print("at amp iter {}".format(t))
        # denoise right singular vector gt to get vt
        # print("before denoise, g shape {}".format(g.shape))
        vdenoiser.fit(g, mu, sigma_sq, figname='_u_iter%02d.png' % (t))
        # print("finish fit")
        v = vdenoiser.denoise(g, mu, sigma_sq)
        # print("v shape {}".format(v.shape))
        # print("finish denoise")
        V = np.dstack((V, np.reshape(v,(-1,k,1)) ))
        b = gamma * np.mean(vdenoiser.ddenoise(g,mu,sigma_sq) , axis = 0)
        # print("I want to look at b")
        # print(b)
        # print("finish ddenoise")
        # update left singular vector ft using vt
        f = X.dot(v) - u.dot(b)
        sigma_bar_sq = v.T @ v / n # non_rotation version
        mu_bar = sigma_bar_sq * signals # non_rotation version    
        # denoise left singular vector ft to get ut
        if not mutev:
            # print("before denoise, f shape {}".format(f.shape))
            udenoiser.fit(f, mu_bar, sigma_bar_sq, figname='_v_iter%02d.png' % (t))
            # print("finish fit")
            u = udenoiser.denoise(f, mu_bar, sigma_bar_sq)
            # print("finish denoise")
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, sigma_bar_sq), axis = 0)
            # print("finish ddenoise")
        if mutev:
            # in this case, u is the derived f
            u = f
            # u fit pass
            # u denoise pass 
            U = np.dstack((U, np.reshape(u,(-1,k,1))))
            # u ddenoise
            b_bar = np.identity(2)
        # update left singular vector gt using ut
        g = np.transpose(X).dot(u) - v.dot(b_bar)
        sigma_sq = u.T @ u / n
        mu = sigma_sq * signals # non_rotation version
    # swap u,v
    return V, U

