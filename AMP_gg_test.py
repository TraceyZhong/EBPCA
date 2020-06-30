import numpy as np 
from AMP_gg import AMP_gg

np.random.seed(402)

n = 2000

s = 1.3
'''
# Gaussian noise 
print('Gaussian noise')
for i in range(1):
    W = np.random.normal(size=(n,n))
    W = (W+np.transpose(W))/np.sqrt(2*n)
    ustar = np.random.binomial(1,0.5,size=n)*2-1
    Y = s/n * np.outer(ustar,ustar) + W
    U = AMP_gg(Y,s,ustar,iters=5)
    Unormsq = np.diag(np.dot(np.transpose(U),U))
    aligns = np.dot(np.transpose(U),ustar) / np.sqrt(Unormsq*n)
    print(aligns)
'''
s = 1
# W = ODO' where D is uniform over +/- 1
print('Noise with eigenvalues +/- 1')
for i in range(1):
  W = np.random.normal(size=(n,n))
  O = np.linalg.qr(W)[0]
  D = np.diag(np.random.binomial(1,0.5,size=n)*2-1)
  W = np.dot(np.dot(O,D),np.transpose(O))
  ustar = np.random.binomial(1,0.5,size=n)*2-1
  Y = s/n * np.outer(ustar,ustar) + W
  # u = ustar * innerprod + np.sqrt(sigma) * np.random.normal(size=n)
  U = AMP_gg(Y,s,ustar,iters=5)
  Unormsq = np.diag(np.dot(np.transpose(U),U))
  aligns = np.dot(np.transpose(U),ustar) / np.sqrt(Unormsq*n)
  print(aligns)