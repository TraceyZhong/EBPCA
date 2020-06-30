# standard libraries 
import math 
import numpy as np 
import numpy.polynomial as polynomial 
import matplotlib.pyplot as plt

# local 
from AMP_free import cumulants_from_moments
from AMP_free import compute_moments
from AMP_free import compute_b
from AMP_free import compute_Sigma


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
    fig.savefig("figures/at_iter_{}.png".format(t))
    plt.close()

def get_align(a,b):
    ''' get inner product of a and b
    a, b: 1d np array. 
    '''
    if (a.shape != b.shape):
        raise Exception("get_align wrong dim.")
    return abs(np.inner(a,b))/ (np.linalg.norm(a) * np.linalg.norm(b))

# def get_mu(alpha, beta, p_star):
#     return alpha + p_star*beta

def get_tSigma(Sigma, beta, q_star):
    '''
    Sigma is t*t
    beta is (t+1)*(t+1)
    tSigma is (t+1)*(t+1)
    '''
    n = Sigma.shape[0]
    tmp = np.vstack(( np.zeros((1, n)) ,Sigma))
    tmp = np.hstack((np.zeros((n+1, 1)), tmp))
    # print("Sigma")
    # print(tmp[-1,-1])
    # print("qstar {}".format(q_star))    
    return q_star**2*np.outer(beta,beta) + tmp

def get_alpha_t(EUu, ELu, s, p_star, t):
    # EUu =( E[Uu1], E[Uu2], ... ]
    return s*(EUu[t-1] - p_star* ELu[t-1])

def get_beta_t(EUu, ELu, z, s, p_star, b, t):
    I = (z + s*p_star**2) * ELu[t-1]
    II = s * p_star * EUu[t-1]
    if (b.shape[0] != t or ELu.shape[0] != t):
        raise Exception("get_beta: size don't align")
    return I - II - np.inner(b, ELu)

def get_mu(alpha, beta, pstar):
    return alpha + pstar*beta

def get_EUu_t(ustar, ut):
    return np.mean(ustar * ut)

def get_ELu_t(ustar, sqrtNphi):
    return np.mean(sqrtNphi * ustar)

def denoise(F,mu,Sigma):
    Sinvmu = np.linalg.solve(Sigma,mu)
    return np.tanh(F.dot(Sinvmu))

def AMP_gg(Y, s, ustar, iters = 10, rank = 1, reg=0.01):
    
    # pars
    n = Y.shape[0]
    eigs = np.linalg.eigvalsh(Y).max()
    w, v = np.linalg.eigh(Y)
    # z = s + 1/s
    z = w[-1]
    f = -v[:,-1]*np.sqrt(n)
    sign = 1 if np.inner(f, ustar)>0 else -1
    phi = sign * v[:,-1]
    sqrtNphi = sign* f
    # pstar and qstar by get_align is not very accurate
    # pstar = np.sqrt(1-1/s**2)
    pstar = get_align(phi, ustar)
    print("Init align")
    print(pstar)
    qstar = np.sqrt(1- pstar**2)

    # some maths :)
    moments = compute_moments(Y,2*iters,remove=rank)
    kappa = cumulants_from_moments(moments,2*iters) 

    # when t = 0:
    alpha = np.array([0])
    beta = np.array([1])
    mu = get_mu(alpha, beta, pstar)
    tSigma = np.array([[qstar**2]])


    # key objects
    F = np.reshape(sign*f,(-1,1))
    plot_save(F[:,0], mu[0], tSigma[0,0], 0)
    t = 0
    # print("alpha {:.4f}; beta {:.4f}; mu {:.4f}; sigma {:.4f}".format(alpha[t], beta[t], mu[t], tSigma[t,t]))
    u = denoise(F, mu, tSigma)
    U = np.reshape(u, (-1,1))

    # key objects
    Delta = np.zeros((iters+1,iters+1))
    Phi = np.zeros((iters+1,iters+1))
    u_tilde = u-np.inner(u,phi)*phi
    Delta[0,0] = np.mean(u_tilde**2) # Tracey   
    # key objects
    EUu = np.array([])
    ELu = np.array([])

    for t in range(1, iters + 1):
        # print("At iter {}".format(t))
        b = compute_b(Phi,t,kappa)
        # print("b {}".format(b))
        f = Y.dot(U[:,-1])-U.dot(b)
        F = np.hstack((F,np.reshape(f,(-1,1))))
        # update EUu and ELu
        EUu = np.append(EUu, get_EUu_t(ustar,u))
        ELu = np.append(ELu, get_ELu_t(ustar,sqrtNphi))
        # print("EUu {}".format(EUu))
        # print("ELu {}".format(ELu))
        # update alpha, beta, mu
        alpha_t = get_alpha_t(EUu, ELu, s, pstar, t)
        alpha = np.append(alpha, alpha_t)
        beta_t = get_beta_t(EUu, ELu, z, s, pstar, b, t)
        beta = np.append(beta, beta_t)
        mu = get_mu(alpha, beta, pstar)
        # update Sigma thus tSigma
        Sigma = compute_Sigma(Delta,Phi,t,kappa)
        D,Q = np.linalg.eigh(Sigma[:t,:t])
        D = np.maximum(D,reg)
        Sigma = Q.dot(np.diag(D)).dot(np.transpose(Q))
        tSigma = get_tSigma(Sigma, beta, qstar)
        # check if the states are correclty characterizing the distribution      
        plot_save(F[:,t], mu[t], tSigma[t,t], t)

        print("Iter {}: alpha {:.4f}; beta {:.4f}; mu {:.4f}; sigma {:.4f}".format(t, alpha[t], beta[t], mu[t], tSigma[t,t]))
        # compute u_{t+1}
        tSinvmu = np.linalg.solve(tSigma,mu)
        u = np.tanh(F.dot(tSinvmu))
        U = np.hstack((U,np.reshape(u,(-1,1))))  
        # Update Delta empirically
        coeff = np.dot(np.transpose(U),phi)
        U_tilde = U - np.outer(phi, coeff)
        u_tilde = u - np.inner(u, phi)*phi
        Euprod = np.transpose(U_tilde).dot(u_tilde)/n # Tracey
        Delta[t,:(t+1)] = Euprod # Tracey
        Delta[:(t+1),t] = Euprod # Tracey
        # Update Phi empirically
        Euprod = np.transpose(U).dot(u)/n
        gradu = tSinvmu * (1-Euprod[t])
        Phi[t,:t] = gradu[1:]         

    return U

     




