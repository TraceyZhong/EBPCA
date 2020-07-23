# Test algorithm in AMP_free.py on some simple examples

import numpy as np
from AMP_free import AMP_free
from debug import hist_save

np.random.seed(123)

ntrials = 5
niters = 10

n = 3000
innerprod = 0.2
sigma = np.sqrt(1-innerprod**2)

def test(W,prefix):
  ustar = np.random.binomial(1,0.5,size=n)*2-1
  X = alpha/n * np.outer(ustar,ustar) + W
  hist_save(np.linalg.eigvalsh(X), '%s_eigvals.png' % prefix)
  u = ustar * innerprod + sigma * np.random.normal(size=n)
  U = AMP_free(X,u,innerprod,iters=niters,plot_prefix=prefix)
  Unormsq = np.diag(np.transpose(U).dot(U))
  aligns = np.transpose(U).dot(ustar) / np.sqrt(Unormsq*n)
  print(aligns)

# Gaussian noise
alpha = 2
print('Gaussian noise, alpha = 2')
for i in range(ntrials):
  prefix='figures/square_gaussian_%d' % i
  W = np.random.normal(size=(n,n))
  W = (W+np.transpose(W))/np.sqrt(2*n)
  test(W,prefix)

# W = ODO' where D is uniform over +/- 1
alpha = 2
print('Noise with eigenvalues +/- 1, alpha = 2')
for i in range(ntrials):
  prefix='figures/square_uniform_%d' % i
  W = np.random.normal(size=(n,n))
  O = np.linalg.qr(W)[0]
  D = np.diag(np.random.binomial(1,0.5,size=n)*2-1)
  W = O.dot(D).dot(np.transpose(O))
  test(W,prefix)

# W = ODO' where D is centered and scaled beta
alpha = 2.5
print('Noise with beta-distributed eigenvalues, alpha = 2.5')
for i in range(ntrials):
  prefix='figures/square_beta_%d' % i
  W = np.random.normal(size=(n,n))
  O = np.linalg.qr(W)[0]
  d = np.random.beta(1,2,size=n)
  d -= np.mean(d)
  d *= np.sqrt(n)/np.linalg.norm(d)
  D = np.diag(d)
  W = O.dot(D).dot(np.transpose(O))
  test(W,prefix)

