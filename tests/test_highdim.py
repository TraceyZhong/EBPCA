import config 


import numpy as np
from scipy.stats import multivariate_normal

from ebpca.empbayes import NonparEBHD

def normalize_col(x):
    return x/np.sqrt(np.sum(x**2, axis = 0))

denoiser = NonparEBHD(em_iter = 50, to_save= True)

mu = np.array([[1,0], [0,1]])
cov = np.array([[1,0], [0,1]])
m = 500

# z = np.linspace([0,0], [50,50], num = 50)

z = np.array([np.array([1,1])]*m)
x = (mu.dot(z.T) + np.random.normal(size = 2*m).reshape((2,m)) ).T


denoiser.fit(x, mu, cov)

u = denoiser.ddenoise(x, mu, cov)
print(u.shape)

'''

u = normalize_col(u)
z = normalize_col(z)
x = normalize_col(x)

Ualigns = np.diag(np.transpose(u).dot(z)) / np.sqrt(m)
Xaligns = np.diag(np.transpose(x).dot(z)) / np.sqrt(m)
print("ualigns")
print(Ualigns)
print("xaligns")
print(Xaligns)
'''