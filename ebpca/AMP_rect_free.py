# Implementation of AMP algorithm for symmetric +/- 1 priors
# and rectangular bi-rotationally invariant noise

import numpy as np
from numpy.polynomial import polynomial
# from debug import plot_save

# Compute first K rectangular free cumulants from moments
#
# Input: m[0] = 1 and m[k] = mu_{2k} for k = 1,...,K
# Output: kappa[0] = 0 and kappa[k] = kappa_{2k} for k = 1,...,K
def cumulants_from_moments(m,gamma,K):
    M = np.copy(m)
    M[0] = 0
    A = polynomial.polyadd(gamma*M,[1])
    B = polynomial.polyadd(M,[1])
    S = polynomial.polymul([0,1],polynomial.polymul(A,B))
    kappa = [0,M[1]]
    for k in range(2,K+1):
        P = [0]
        for j in range(1,k):
            P = polynomial.polyadd(P,kappa[j]*polynomial.polypow(S,j))
        kappa.append(M[k]-P[k])
    return kappa

# Remove top singular values and compute moments of remaining singular value
# distribution
def compute_moments(X,k,remove=1):
    eigs = np.linalg.eigvalsh(X.dot(np.transpose(X)))[:-remove]
    m = []
    for j in range(k+1):
        m.append(np.mean(eigs**j))
    return np.array(m)

# Compute debiasing coefficients
#
# Input:
# Phi: [[ 0,             , 0                , ... , 0                        , 0 ],
#             [ <d_1 u_2>, 0                , ... , 0                        , 0 ],
#             [ <d_1 u_3>, <d_2 u_3>, ... , 0                        , 0 ],
#                 ...
#             [ <d_1 u_T>, <d_2 u_T>, ... , <d_{T-1} u_T>, 0 ]]
#
# Psi: [[ <d_1 v_1>, 0                , ... , 0                 ],
#             [ <d_1 v_2>, <d_2 v_2>, ... , 0                 ],
#                 ...
#             [ <d_1 v_T>, <d_2 v_T>, ... , <d_t v_T> ]]
#
# Phi should be correct up to its upper (t x t) submatrix, and
# Psi should be correct up to its upper (t-1 x t-1) submatrix 
#
# Returns [ b_{t,1}, ..., b_{t,t-1} ]
def compute_b(Phi,Psi,gamma,t,kappa):
    prod = Phi
    Btrans = gamma * kappa[1] * prod
    for k in range(2,t):
        prod = prod.dot(Psi).dot(Phi)
        Btrans += gamma * kappa[k] * prod
    return Btrans[t-1,:(t-1)]

# Input: Phi, Psi same as above
#
# Phi and Psi should both be correct up to their upper (t x t) submatrices
#
# Returns [ a_{t,1}, ..., a_{t,t} ]
def compute_a(Phi,Psi,gamma,t,kappa):
    prod = Psi
    Atrans = kappa[1] * prod
    for k in range(2,t+1):
        prod = prod.dot(Phi).dot(Psi)
        Atrans += kappa[k] * prod
    return Atrans[t-1,:t]

# Compute noise covariance
#
# Input: Phi, Psi same as above
#
# Delta: [[ <u_1^2>     , <u_1 u_2> , ... , <u_1 u_2> ],
#                 [ <u_2 u_1> , <u_2^2>     , ... , <u_2 u_T> ],
#                    ...
#                 [ <u_T u_1> , <u_T u_2> , ... , <u_T^2>     ]]
#
# Gamma: [[ <v_1^2>     , <v_1 v_2> , ... , <v_1 v_2> ],
#                 [ <v_2 v_1> , <v_2^2>     , ... , <v_2 v_T> ],
#                    ...
#                 [ <v_T v_1> , <v_T v_2> , ... , <v_T^2>     ]]
#
# Delta,Phi should be correct up to their upper (t x t) submatrices, and
# Gamma,Psi should be correct up to their upper (t-1 x t-1) submatrices
#
# Returns    [[ omega_{11}, ... , omega_{1t} ],
#                     [ omega_{21}, ... , omega_{2t} ],
#                         ...
#                     [ omega_{t1}, ... , omega_{tt} ]]
def compute_Omega(Delta,Gamma,Phi,Psi,gamma,t,kappa):
    Omega = gamma * kappa[1] * Delta
    for j in range(2,2*t):
        Theta = np.zeros(Omega.shape)
        for k in range(2*j-1):
            prod = np.identity(Omega.shape[0])
            for m in range(2*j-1):
                if m == k and m % 2 == 0:
                    prod = prod.dot(Delta)
                elif m == k and m % 2 == 1:
                    prod = prod.dot(Gamma)
                elif m < k and m % 2 == 0:
                    prod = prod.dot(Phi)
                elif m < k and m % 2 == 1:
                    prod = prod.dot(Psi)
                elif m > k and m % 2 == 0:
                    prod = prod.dot(np.transpose(Phi))
                elif m > k and m % 2 == 1:
                    prod = prod.dot(np.transpose(Psi))
            Theta += prod
        Omega += gamma * kappa[j] * Theta
    return Omega[:t,:t]

# Input: Delta, Gamma, Phi, Psi same as above
#
# Delta,Gamma,Phi,Psi should be correct up to their upper (t x t) submatrices
#
# Returns    [[ sigma_{11}, ... , sigma_{1t} ],
#                     [ sigma_{21}, ... , sigma_{2t} ],
#                         ...
#                     [ sigma_{t1}, ... , sigma_{tt} ]]
def compute_Sigma(Delta,Gamma,Phi,Psi,gamma,t,kappa):
    Sigma = kappa[1] * Gamma
    for j in range(2,2*t+1):
        Xi = np.zeros(Sigma.shape)
        for k in range(2*j-1):
            prod = np.identity(Sigma.shape[0])
            for m in range(2*j-1):
                if m == k and m % 2 == 0:
                    prod = prod.dot(Gamma)
                elif m == k and m % 2 == 1:
                    prod = prod.dot(Delta)
                elif m < k and m % 2 == 0:
                    prod = prod.dot(Psi)
                elif m < k and m % 2 == 1:
                    prod = prod.dot(Phi)
                elif m > k and m % 2 == 0:
                    prod = prod.dot(np.transpose(Psi))
                elif m > k and m % 2 == 1:
                    prod = prod.dot(np.transpose(Phi))
            Xi += prod
        Sigma += kappa[j] * Xi
    return Sigma[:t,:t]

# AMP with free cumulant corrections
#
# Inputs: X, the data matrix
#                 u = u_1, the initialization
#                 innerprod = <u_1 u_*>
#
# For symmetric +/- 1 prior, the posterior mean denoiser E[U_* | F] when
#     F ~ N(U_* mu, Sigma)
# is
#     u(F) = tanh( <F, Sigma^{-1} mu> )
# Its gradient is
#     grad u(F) = Sigma^{-1} mu * [1 - u(F)^2]
#
def AMP_rect_free(X,u,innerprod,iters=5,rank=1,reg=0.001,plot_prefix=None):
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
        v = np.tanh(G.dot(Oinvnu)) # TODO change to denoise v
        V = np.hstack((V,np.reshape(v,(-1,1)))) 
        # # Plot empirical distribution, for debugging
        # if plot_prefix is not None:
        #     filename = '%s_v_iter%02d.png' % (plot_prefix,t)
        #     plot_save(G.dot(Oinvnu), nu.dot(Oinvnu), nu.dot(Oinvnu), filename)
        # Update Gamma empirically
        Gamma[:t,:t] = np.transpose(V).dot(V)/n
        # Update Psi empirically
        gradv = Oinvnu * (1-Gamma[t-1,t-1]) # TODO not sure if should change
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
        u = np.tanh(F.dot(Sinvmu)) # TODO change eb estimate 
        U = np.hstack((U,np.reshape(u,(-1,1))))
        # # Plot empirical distribution, for debugging
        # if plot_prefix is not None:
        #     filename = '%s_u_iter%02d.png' % (plot_prefix,t)
        #     plot_save(F.dot(Sinvmu), mu.dot(Sinvmu), mu.dot(Sinvmu), filename)
        # Update Delta empirically
        Delta[:(t+1),:(t+1)] = np.transpose(U).dot(U)/m
        # Update Phi empirically
        gradu = Sinvmu * (1-Delta[t,t]) # TODO change to empirical version
        Phi[t,:t] = gradu
    return U,V

