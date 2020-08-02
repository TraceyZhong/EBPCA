import config

import numpy as np

from ebpca.pca import signal_solver_gaussian
from ebpca.pca import signal_solver


n_trials = 3

m = 1000
n = 2000
innerprod = 0.2
sigma = np.sqrt(1-innerprod**2)

def get_alignment(u,v):
    # consider u and v are one dimmensional
    u = u.flatten()
    v = v.flatten()
    return u.dot(v)/(np.linalg.norm(u) * np.linalg.norm(v))


def test(W, alpha, noise_type):
    ustar = np.random.binomial(1,0.5,size=m)*2-1
    vstar = np.random.binomial(1,0.5,size=n)*2-1  
    X = alpha/n * np.outer(ustar,vstar) + W
    u, s, vh = np.linalg.svd(X,full_matrices=False)
    estimates = signal_solver_gaussian(singval = s[0], mu = s[1:], n_samples=m, n_features=n)
    print("alpha [%.4f, %.4f], sample align = [%.4f, %.4f], feature align = [%.4f, %.4f]" % (alpha, estimates["alpha"], 
            get_alignment(u[:,0], ustar), estimates["sample_align"], get_alignment(vh[0,:], vstar), estimates["feature_align"] ))

alpha = float(2)
print('Gaussian noise, alpha = 2')
for i in range(n_trials):
    W = np.random.normal(size=(m,n))/np.sqrt(n)
    test(W,alpha, "Gaussian")

'''
alpha = float(2)
print('Noise with all singular values +1, alpha = 2')
for i in range(n_trials):
    W = np.random.normal(size=(m,m))
    O = np.linalg.qr(W)[0]
    W = np.random.normal(size=(n,n))
    Q = np.linalg.qr(W)[0]
    D = np.zeros((m,n))
    for i in range(min(m,n)): D[i,i] = 1
    W = O.dot(D).dot(Q)
    test(W, alpha, "PointMass")


alpha = float(3)
print('Noise with beta-distributed singular values, alpha = 3')
for i in range(n_trials):
    W = np.random.normal(size=(m,m))
    O = np.linalg.qr(W)[0]
    W = np.random.normal(size=(n,n))
    Q = np.linalg.qr(W)[0]
    D = np.zeros((m,n))
    d = np.random.beta(1,2,size=min(m,n))
    d *= np.sqrt(min(m,n))/np.linalg.norm(d)
    for i in range(min(m,n)): D[i,i] = d[i]
    W = np.dot(np.dot(O,D),Q)
    test(W,alpha, "Beta")




Sample Usage:

estimates = signal_solver(singval, mu, n_samples, n_features)

singval: one of the leading singular values
mu: bulk eigenvalues with the leading k's removed

other parameters to control
spike must be tol away form the bulk. <=> singval > supp_max + tol


Output: [truth, estimate]
Gaussian noise, alpha = 2
alpha = [2.0000, 2.0486], sample align = [0.8458, 0.8464], feature align = [0.7726, 0.7753]
alpha = [2.0000, 2.0139], sample align = [0.8287, 0.8407], feature align = [0.7583, 0.7684]
alpha = [2.0000, 2.0360], sample align = [0.8505, 0.8447], feature align = [0.7768, 0.7731]
Noise with all singular values +1, alpha = 2
alpha = [2.0000, 2.0034], sample align = [0.9228, 0.9240], feature align = [0.8541, 0.8539]
alpha = [2.0000, 2.0247], sample align = [0.9264, 0.9250], feature align = [0.8571, 0.8558]
alpha = [2.0000, 2.0280], sample align = [0.9291, 0.9252], feature align = [0.8572, 0.8560]
Noise with beta-distributed singular values, alpha = 3
alpha = [3.0000, 3.0126], sample align = [0.8515, 0.8588], feature align = [0.8061, 0.8119]
alpha = [3.0000, 2.9904], sample align = [-0.8602, 0.8675], feature align = [-0.8083, 0.8204]
alpha = [3.0000, 3.0300], sample align = [-0.8666, 0.8633], feature align = [-0.8065, 0.8166]
'''