# Implementation of AMP algorithm for symmetric +/- 1 priors
# and rotationally invariant noise

import numpy as np
import math
from numpy.polynomial import polynomial

import matplotlib.pyplot as plt
# from AMP_gg import plot_save

def pdf_mu_sigmasq(x, mu,sigmasq):
    return 1/(2*np.sqrt(2*np.pi*sigmasq)) * np.exp(-(x-mu)**2/(2*sigmasq)) + 1/(2*np.sqrt(2*np.pi*sigmasq)) * np.exp(-(x+mu)**2/(2*sigmasq)) 

def plot_save(f, mu, sigmasq, t):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7*1, 5))
    ax.hist(f, bins = 50, density = True)
    start = np.quantile(f, 0.05)*1.5
    end = np.quantile(f,0.95)*1.5
    xgrid = np.linspace(start, end, 50)
    ax.plot(xgrid, pdf_mu_sigmasq(xgrid, mu, sigmasq), color = 'orange')
    ax.vlines(x = mu, ymin = 0, ymax = pdf_mu_sigmasq(mu, mu, sigmasq), color = 'orange')
    ax.vlines(x = -mu, ymin = 0, ymax = pdf_mu_sigmasq(-mu, mu, sigmasq), color = 'orange')     
    ax.set_title("at_iter {}".format(t))
    fig.savefig("figures/Free_at_iter_{}.png".format(t))
    plt.close()

# Compute first K free cumulants from moments
#
# Input: m[0] = 1 and m[k] = <lambda^k> for k = 1,...,K
# Output: kappa[0] = 0 and kappa[k] = kappa_k for k = 1,...,K
def cumulants_from_moments(m,K):
  S = polynomial.polymul([0,1],m)
  kappa = [0,m[1]]
  for k in range(2,K+1):
    P = [0]
    for j in range(1,k):
      P = polynomial.polyadd(P,kappa[j]*polynomial.polypow(S,j))
    kappa.append(m[k]-P[k])
  return kappa

# Remove top eigenvalues, and compute moments of remaining eigenvalue
# distribution
def compute_moments(Y,k,remove=1):
  eigs = np.linalg.eigvalsh(Y)[:-remove]
  m = []
  for j in range(k+1):
    m.append(np.mean(eigs**j))
  return np.array(m)

# Compute debiasing coefficients
#
# Input:
# Phi = [[ 0,       , 0        , ... , 0            , 0 ],
#        [ <d_1 u_2>, 0        , ... , 0            , 0 ],
#        [ <d_1 u_3>, <d_2 u_3>, ... , 0            , 0 ],
#          ...
#        [ <d_1 u_T>, <d_2 u_T>, ... , <d_{T-1} u_T>, 0 ]]
#
# Phi should be correct up to its upper (t x t) submatrix
#
# Returns [ b_{t1}, ..., b_{tt} ]
def compute_b(Phi,t,kappa):
  B = np.zeros(Phi.shape)
  for j in range(t):
    B += kappa[j+1] * np.linalg.matrix_power(np.transpose(Phi),j)
  return B[:t,t-1]

# Compute noise covariance
#
# Input: Phi same as above
#
# Delta = [[ <u_1^2>   , <u_1 u_2> , ... , <u_1 u_2> ],
#          [ <u_2 u_1> , <u_2^2>   , ... , <u_2 u_T> ],
#           ...
#          [ <u_T u_1> , <u_T u_2> , ... , <u_T^2>   ]]
#
# Delta should be correct up to its upper (t x t) submatrix
#
# Returns  [[ sigma_{11}, ... , sigma_{1t} ],
#           [ sigma_{21}, ... , sigma_{2t} ],
#             ...
#           [ sigma_{t1}, ... , sigma_{tt} ]]

def compute_Sigma(Delta,Phi,t,kappa):
  Sigma = np.zeros(Delta.shape)
  for j in range(2*t-1):
    Theta = np.zeros(Delta.shape)
    for i in range(j+1):
      prod = np.linalg.matrix_power(Phi,i).dot(Delta)
      prod = prod.dot(np.linalg.matrix_power(np.transpose(Phi),j-i))
      Theta += prod
    Sigma += kappa[j+2] * Theta
  return Sigma[:t,:t]

# AMP with free cumulant corrections
#
# Inputs: u = u_1, the initialization
#         innerprod = <u_1 u_*>
#
# For symmetric +/- 1 prior, the posterior mean denoiser E[U_* | F] when
#   F ~ N(U_* mu, Sigma)
# is
#   u(F) = tanh( <F, Sigma^{-1} mu> )
# Its gradient is
#   grad u(F) = Sigma^{-1} mu * [1 - u(F)^2]
#
def AMP_free(Y,s,u,innerprod,iters=10,rank=1,reg=0.01):
  n = Y.shape[0]
  moments = compute_moments(Y,2*iters,remove=rank)
  kappa = cumulants_from_moments(moments,2*iters)
  U = np.reshape(u,(-1,1))
  Delta = np.zeros((iters+1,iters+1))
  Phi = np.zeros((iters+1,iters+1))
  Delta[0,0] = np.mean(u**2)
  mu = np.array([])
  for t in range(1,iters+1):
    # Compute f_t
    b = compute_b(Phi,t,kappa)
    f = Y.dot(U[:,-1])-U.dot(b)
    if t == 1:
      F = np.reshape(f,(-1,1))
    else:
      F = np.hstack((F,np.reshape(f,(-1,1))))
    # Update Sigma
    #   (For numerical stability, ensure lambda_min(Sigma) >= reg)
    Sigma = compute_Sigma(Delta,Phi,t,kappa)
    D,Q = np.linalg.eigh(Sigma[:t,:t])
    D = np.maximum(D,reg)
    Sigma = Q.dot(np.diag(D)).dot(np.transpose(Q))
    # Update mu empirically using E[F_t^2] = mu_t^2 + Sigma_{t,t}
    #   (For numerical stability, ensure mu_t >= reg)
    mu_t = np.sqrt(max(np.mean(f**2)-Sigma[t-1,t-1], reg))
    mu = np.append(mu,mu_t)
    # sanity check
    plot_save(f,mu[-1],Sigma[-1,-1], t)
    # Compute u_{t+1}
    Sinvmu = np.linalg.solve(Sigma,mu)
    u = np.tanh(F.dot(Sinvmu))
    U = np.hstack((U,np.reshape(u,(-1,1))))
    # Update Delta empirically
    Euprod = np.transpose(U).dot(u)/n
    Delta[t,:(t+1)] = Euprod
    Delta[:(t+1),t] = Euprod
    # Update Phi empirically
    gradu = Sinvmu * (1-Euprod[t])
    Phi[t,:t] = gradu
  return U

