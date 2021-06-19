'''
==========
Date: Jun 10
AMP algorithm with PCA initialization for symmetric matrices:
==========
'''
import numpy as np
from numpy.polynomial import polynomial

from utils import plot_save

## --- get the kappas, zetas and dzetas --- ##

def get_moments_from_empirical_measure(mu, K):
    m = []
    for j in range(K + 1):
        m.append(np.mean(mu**j))
    return np.array(m)

def get_cumulants_from_moments(m,K):
    '''
    Return kappas[0] to be kappa_0
    '''
    S = polynomial.polymul([0,1],m)
    kappa = [0,m[1]]
    for k in range(2,K+1):
        P = [0]
        for j in range(1,k):
            P = polynomial.polyadd(P,kappa[j]*polynomial.polypow(S,j))
            # print("k = {}, j = {}, P =".format(k,j), P)
        if len(P) < k:
            kappa.append(m[k]-0)
        else:
            kappa.append(m[k]-P[k])
    return kappa

def get_zetas_from_kappas(zeta0, theta, kappas):
    '''
    Return zetas[i] to be zeta_i
    '''
    zetas = [zeta0]
    prev = zeta0
    for i in range(len(kappas)):
        prev = (prev - kappas[i])*theta
        zetas.append(prev)
    return np.array(zetas)

def get_dzetas_from_zetas(dzeta0, theta, zetas):
    '''
    Return dzetas[i] to be dzetas_i
    '''
    dzetas = [dzeta0]
    prev = dzeta0
    for i in range(len(zetas)):
        prev = (prev - zetas[i])*theta
        dzetas.append(prev)
    return np.array(dzetas)

def get_zeta0(lmda, theta, kappa0):
    return lmda/theta - 1 + kappa0

def get_dzeta0(mu_pca, zeta0, kappa0):
    return 1 - mu_pca**2 - kappa0 + 2*zeta0

## --- Debiasing coefficients --- ##

def get_B(t, Phi, kappas, zetas):
    '''
    B0 associates with all debiasing terms doesn't involve u_0
    B1 assciates with all debiasing terms invoving u_0
    '''
    B0 = np.zeros(Phi.shape)
    B1 = np.zeros(Phi.shape)
    Phij = np.identity(Phi.shape[0]) # Phi to the power of j
    for j in range(t+1):
        B0 += kappas[j+1] * Phij 
        B1 += zetas[j+1] * Phij 
        Phij = Phij @ Phi
    B = np.zeros(Phi.shape)
    B[:,1:] = B0[:,1:]
    B[:, 0] = B1[:,0]
    return B

def get_Theta(j, Phi, Delta):
    Theta = np.zeros(Delta.shape)
    for i in range(j+1):
        prod = np.linalg.matrix_power(Phi,i).dot(Delta)
        prod = prod.dot(np.linalg.matrix_power(np.transpose(Phi),j-i))
        Theta += prod
    return Theta

def get_Sigma(t, Phi, Delta, kappas, zetas, dzetas):
    '''
    0 indicates the term doen's involve u_0
    1 indicates the term involving 1 u_0
    2 indicates the term involving 2 u_0
    we get the Sigma for f0, ... , fT
    '''
    # TODO check length for kappas, zetas and dzetas.
    assert Phi.shape == Delta.shape
    Sigma = np.zeros(Delta.shape)

    Delta0 = np.zeros(Delta.shape)
    Delta1 = np.zeros(Delta.shape)
    Delta2 = np.zeros(Delta.shape)
    Delta2[0,0] = Delta[0,0]
    Delta1[1:,0] = Delta[1:,0]
    Delta1[0,1:] = Delta[0,1:]
    Delta0[1:,1:] = Delta[1:,1:]

    for j in range(2*t+1):
        Theta0 = get_Theta(j, Phi, Delta0)
        Theta1 = get_Theta(j, Phi, Delta1)
        Theta2 = get_Theta(j, Phi, Delta2)
        Sigma += kappas[j+2] * Theta0 + zetas[j+2] * Theta1 + dzetas[j+2] * Theta2
    
    return Sigma[:(t+1), :(t+1)]

def oamp_pca(Y, theta, singval, innerprod, fsvd, ustar = None, iters=5, reg = 0.01, kappas = None, **kwargs):
    if "noise_type" in kwargs.keys():
        noise_type = kwargs["noise_type"]
    else:
        noise_type = None
    n = Y.shape[0]
    K = 4*iters
    if kappas is None or (len(kappas) < K):
        eigs = np.linalg.eigvalsh(Y)[:-1]
        if "moments" in kwargs.keys():
            moments = kwargs["moments"]
        else:
            moments = get_moments_from_empirical_measure(eigs, K)# TODO : don't know when suffice
        kappas = get_cumulants_from_moments(moments, K)
    # Note the init for zetas and dzetas 
    zeta0 = get_zeta0(singval, theta, kappas[0])
    zetas = get_zetas_from_kappas(zeta0, theta, kappas)
    dzeta0 = get_dzeta0(innerprod, zetas[0], kappas[0])
    dzetas = get_dzetas_from_zetas(dzeta0,theta, zetas)

    # print("kappas", kappas)
    # print("zetas", zetas)
    # print("dzetas", dzetas)
    # Iterates
    U = np.zeros((n,iters+1)) #(u_0, ..., u_T)
    U[:,0] = fsvd/theta
    F = np.reshape(fsvd, (-1,1)) # (f_0, ..., f_{T-1})
    mu = np.array([innerprod])
    Sigma = np.array([1 - mu**2])
    plot_save(fsvd,mu[-1],Sigma[-1,-1], 0, "oamp"+ noise_type)
    # States
    Delta = np.zeros((iters+1, iters+1))
    Phi = np.zeros((iters+1, iters+1))
    Delta[0,0] = 1/theta**2 # = np.mean(u**2)
    
    # Given u_0, we try to get u_{iters}
    for t in range(1, iters+1):
        print("At iter", t)
        # Step 1: Denoise: we need to first get u_{t}
        # Note the only the dimension of Sigma and mu and F is different
        Sinvmu = np.linalg.solve(Sigma,mu)
        muSinvmu = mu.dot(Sinvmu)
        plot_save(F[:,:t].dot(Sinvmu), muSinvmu,muSinvmu, t, "oamp"+  noise_type)
        u = np.tanh(F[:,:t].dot(Sinvmu))
        
        U[:,t] = u 
        # Update Delta empirically
        Euprod = np.transpose(U).dot(u) / n
        Delta[t,:] = Euprod
        Delta[:,t] = Euprod
        # Update Phi empirically
        gradu = Sinvmu * (1-Euprod[t])
        Phi[t, :t] = gradu
        # Step 2: Update, get f_{t}
        if t == iters:
            return U

        B = get_B(t, Phi, kappas, zetas)
        b = B[t,:]
        f = Y.dot(u) - U.dot(b)
        F = np.hstack((F,np.reshape(f,(-1,1))))
        Sigma = get_Sigma(t, Phi, Delta, kappas, zetas, dzetas)
        mu_t = np.sqrt(max(np.mean(f**2)-Sigma[t,t], reg))
        mu = np.append(mu, mu_t)
        # plot_save(f,mu_t,Sigma[-1,-1], t, noise_type)
        

