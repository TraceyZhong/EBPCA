'''
==========
EBAMP
==========
finally we are here, think.
'''

import numpy as np

from ebpca.AMP_rect_free import cumulants_from_moments, compute_moments, compute_a, compute_b, compute_Omega, compute_Sigma
from ebpca.empbayes import NonparEB



def ebamp_gaussian_old(X, u, init_align, iters = 5, rank = 1, udenoiser = NonparEB(), vdenoiser = NonparEB, v = None, v_init_align = None, signal = 0):
    # Normalize u, v and initialize U V
    # n is the direction of features, must be the inverse scale of the variance.
    X = np.transopose(X) # shape becomes n*d 
    (n,d) = X.shape
    alpha = d/n

    # U = np.reshape(u,(-1,1))
    # V = np.reshape(v,(-1,1))

    U = np.reshape(v,(-1,1))
    V = np.reshape(u,(-1,1))

    u = 1/(signal * alpha) * np.sqrt((signal**2*alpha + 1)/(signal**2 + 1)) * u / np.linalg.norm(u) * np.sqrt(m)
    g = v/np.linalg.norm(v) * np.sqrt(n) 

    mu = v_init_align
    sigma_sq = 1 - mu**2
    for t in range(iters):
        print("at amp iter {}".format(t))
        # denoise right singular vector gt to get vt
        vdenoiser.fit(g, mu, np.sqrt(sigma_sq), figname='iter%02d.png' % (t))
        v = vdenoiser.denoise(g, mu, np.sqrt(sigma_sq))
        V = np.hstack((V,np.reshape(v,(-1,1))))
        b = alpha * np.mean(vdenoiser.ddenoise(g,mu,np.sqrt(sigma_sq)))
        # update left singular vector ft using vt
        f = X.dot(v) - b*u
        sigma_bar_sq = alpha * np.mean(v**2)
        mu_bar = np.sqrt(np.mean(f**2) - sigma_bar_sq)
        # denoise left singular vector ft to get ut
        udenoiser.fit(f, mu_bar, np.sqrt(sigma_bar_sq), figname='iter%02d.png' % (t))
        u = udenoiser.denoise(f, mu_bar, np.sqrt(sigma_bar_sq))
        U = np.hstack((U, np.reshape(u, (-1,1))))
        b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, np.sqrt(sigma_bar_sq)))
        # update right singular vector gt using ut
        g = np.transpose(X).dot(u) - b_bar * v
        sigma_sq = np.mean(u**2)
        mu = np.sqrt(np.mean(g**2) - sigma_sq)
    return U,V


def ebamp_gaussian(X, u, init_align, iters = 5, rank = 1, udenoiser = NonparEB(), vdenoiser = NonparEB, v = None, v_init_align = None, signal = 0):
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
        print("at amp iter {}".format(t))
        # denoise right singular vector gt to get vt
        vdenoiser.fit(g, mu, np.sqrt(sigma_sq), figname='iter%02d.png' % (t))
        v = vdenoiser.denoise(g, mu, np.sqrt(sigma_sq))
        V = np.hstack((V,np.reshape(v,(-1,1))))
        b = alpha * np.mean(vdenoiser.ddenoise(g,mu,np.sqrt(sigma_sq)))
        # update left singular vector ft using vt
        f = X.dot(v) - b*u
        sigma_bar_sq = alpha * np.mean(v**2)
        mu_bar = np.sqrt(np.mean(f**2) - sigma_bar_sq)
        # denoise left singular vector ft to get ut
        udenoiser.fit(f, mu_bar, np.sqrt(sigma_bar_sq), figname='iter%02d.png' % (t))
        u = udenoiser.denoise(f, mu_bar, np.sqrt(sigma_bar_sq))
        U = np.hstack((U, np.reshape(u, (-1,1))))
        b_bar = np.mean(udenoiser.ddenoise(f, mu_bar, np.sqrt(sigma_bar_sq)))
        # update right singular vector gt using ut
        g = np.transpose(X).dot(u) - b_bar * v
        sigma_sq = np.mean(u**2)
        mu = np.sqrt(np.mean(g**2) - sigma_sq)
    # return U,V, need to change direction back
    return V,U

def ebamp_orthog(X, u, innerprod, iters = 5, rank =1, reg = 0.001, udenoiser = NonparEB(), vdenoiser = NonparEB(),*args, **kwargs):
    '''
    Input: denoiser is a empbayes object
    '''
    # Compute moments and cumulants of noise
    (m,n) = X.shape
    gamma = float(m)/n
    moments = compute_moments(X,2*iters,remove=rank)
    kappa = cumulants_from_moments(moments,gamma,2*iters)
    # Initialize U, V, F, G, mu, nu, Delta, Gamma, Phi, Psi
    U = np.reshape(u,(-1,1))
    V = np.zeros(shape=(n,0))
    F = np.zeros(shape=(m,0))
    G = np.zeros(shape=(n,0))
    mu = np.array([])
    nu = np.array([])
    Delta = np.zeros((iters+1,iters+1))
    Phi = np.zeros((iters+1,iters+1))
    Gamma = np.zeros((iters+1,iters+1))
    Psi = np.zeros((iters+1,iters+1))
    Delta[0,0] = np.mean(u**2)
    for t in range(1,iters+1):
        # Compute g_t
        if t == 1:
            g = np.transpose(X).dot(U[:,-1])
        else:
            b = compute_b(Phi,Psi,gamma,t,kappa)
            g = np.transpose(X).dot(U[:,-1])-V.dot(b)
        G = np.hstack((G,np.reshape(g,(-1,1))))
        # Update Omega
        #     (For numerical stability, ensure lambda_min(Omega) >= reg)
        Omega = compute_Omega(Delta,Gamma,Phi,Psi,gamma,t,kappa)
        D,Q = np.linalg.eigh(Omega[:t,:t])
        D = np.maximum(D,reg)
        Omega = Q.dot(np.diag(D)).dot(np.transpose(Q))
        # Update nu empirically using E[G_t^2] = nu_t^2 + Omega_{t,t}
        #     (For numerical stability, ensure nu_t >= reg)
        nu_t = np.sqrt(max(np.mean(g**2)-Omega[t-1,t-1],reg**2))
        nu = np.append(nu,nu_t)
        # Compute v_t
        Oinvnu = np.linalg.solve(Omega,nu)
        # transform and denoise
        margin = G.dot(Oinvnu)
        state = nu.dot(Oinvnu)
        # print("v itr {}, {}".format(t, state))
        vdenoiser.fit(margin, mu = state, sigma = np.sqrt(state), figname='iter%02d.png' % (t))
        v = vdenoiser.denoise(margin, state, np.sqrt(state))
        # finish transform and denoise
        V = np.hstack((V,np.reshape(v,(-1,1)))) 
        # # Plot empirical distribution, for debugging
        # if plot_prefix is not None:
        #     filename = '%s_v_iter%02d.png' % (plot_prefix,t)
        #     plot_save(G.dot(Oinvnu), nu.dot(Oinvnu), nu.dot(Oinvnu), filename)
        # Update Gamma empirically
        Gamma[:t,:t] = np.transpose(V).dot(V)/n
        # Update Psi empirically
        gradv = Oinvnu * np.mean(vdenoiser.ddenoise(margin, mu = state, sigma = np.sqrt(state)))
        Psi[t-1,:t] = gradv
        # Compute f_t
        a = compute_a(Phi,Psi,gamma,t,kappa)
        f = X.dot(V[:,-1])-U.dot(a)
        F = np.hstack((F,np.reshape(f,(-1,1))))
        # Update Sigma
        #     (For numerical stability, ensure lambda_min(Sigma) >= reg)
        Sigma = compute_Sigma(Delta,Gamma,Phi,Psi,gamma,t,kappa)
        D,Q = np.linalg.eigh(Sigma[:t,:t])
        D = np.maximum(D,reg)
        Sigma = Q.dot(np.diag(D)).dot(np.transpose(Q))
        # Update mu empirically using E[F_t^2] = mu_t^2 + Sigma_{t,t}
        #     (For numerical stability, ensure mu_t >= reg)
        mu_t = np.sqrt(max(np.mean(f**2)-Sigma[t-1,t-1],reg**2))
        mu = np.append(mu,mu_t)
        # Compute u_t
        Sinvmu = np.linalg.solve(Sigma,mu)
        # transform and denoise
        margin = F.dot(Sinvmu)
        state = mu.dot(Sinvmu)
        print("u itr {}, {}".format(t, state))
        udenoiser.fit(margin, mu = state, sigma = np.sqrt(state), figname = 'iter%02d.png' % (t))
        u = udenoiser.denoise(margin, state, np.sqrt(state))
        # finish transform and denoise
        U = np.hstack((U,np.reshape(u,(-1,1))))
        # # Plot empirical distribution, for debugging
        # if plot_prefix is not None:
        #     filename = '%s_u_iter%02d.png' % (plot_prefix,t)
        #     plot_save(F.dot(Sinvmu), mu.dot(Sinvmu), mu.dot(Sinvmu), filename)
        # Update Delta empirically
        Delta[:(t+1),:(t+1)] = np.transpose(U).dot(U)/m
        # Update Phi empirically
        gradu = Sinvmu * np.mean(udenoiser.ddenoise(margin, mu = state, sigma = np.sqrt(state)))
        Phi[t,:t] = gradu
    return U,V
    



