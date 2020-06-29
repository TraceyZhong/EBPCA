# Test algorithm in AMP_free.py on some simple examples

import numpy as np
from AMP_free import AMP_free

np.random.seed(3872)

n = 1000
s = 1.4
innerprod = 0.2
sigma = 1-innerprod**2

'''
# Gaussian noise
print('Gaussian noise')
for i in range(1):
  W = np.random.normal(size=(n,n))
  W = (W+np.transpose(W))/np.sqrt(2*n)
  ustar = np.random.binomial(1,0.5,size=n)*2-1
  Y = s/n * np.outer(ustar,ustar) + W
  u = ustar * innerprod + np.sqrt(sigma) * np.random.normal(size=n)
  U = AMP_free(Y,s,u,innerprod,iters=10)
  Unormsq = np.diag(np.dot(np.transpose(U),U))
  aligns = np.dot(np.transpose(U),ustar) / np.sqrt(Unormsq*n)
  print(aligns)

'''

# W = ODO' where D is uniform over +/- 1
print('Noise with eigenvalues +/- 1')
for i in range(1):
  W = np.random.normal(size=(n,n))
  O = np.linalg.qr(W)[0]
  D = np.diag(np.random.binomial(1,0.5,size=n)*2-1)
  W = np.dot(np.dot(O,D),np.transpose(O))
  ustar = np.random.binomial(1,0.5,size=n)*2-1
  Y = s/n * np.outer(ustar,ustar) + W
  u = ustar * innerprod + np.sqrt(sigma) * np.random.normal(size=n)
  U = AMP_free(Y,s,u,innerprod,iters=10)
  Unormsq = np.diag(np.dot(np.transpose(U),U))
  aligns = np.dot(np.transpose(U),ustar) / np.sqrt(Unormsq*n)
  print(aligns)

'''
# W = ODO' where D is centered and scaled beta
print('Noise with beta-distributed eigenvalues')
for i in range(10):
  W = np.random.normal(size=(n,n))
  O = np.linalg.qr(W)[0]
  d = np.random.beta(1,2,size=n)
  d -= np.mean(d)
  d *= np.sqrt(n)/np.linalg.norm(d)
  D = np.diag(d)
  W = np.dot(np.dot(O,D),np.transpose(O))
  ustar = np.random.binomial(1,0.5,size=n)*2-1
  Y = s/n * np.outer(ustar,ustar) + W
  u = ustar * innerprod + np.sqrt(sigma) * np.random.normal(size=n)
  U = AMP_free(Y,s,u,innerprod,iters=10)
  Unormsq = np.diag(np.dot(np.transpose(U),U))
  aligns = np.dot(np.transpose(U),ustar) / np.sqrt(Unormsq*n)
  print(aligns)
'''
