import config 

import numpy as np

from ebpca.empbayes import PointNormalEB
from ebpca.empbayes import NonparEB

def get_alignment(u,v):
    # consider u and v are one dimmensional
    u = u.flatten()
    v = v.flatten()
    return u.dot(v)/(np.linalg.norm(u) * np.linalg.norm(v))


denoiser = PointNormalEB(em_iter = 1000)

size = 2000

pi = 1

sigma_x = 0.5

x = np.tile(0, reps = int(round((1-pi)*size)))

x = np.concatenate([x, np.random.normal(scale = sigma_x, size = int(pi*size))])

mu = 0.5
sigma = 0.2

y = mu*x + np.random.normal(scale = sigma, size = size) 

# denoiser.fit(y, mu, sigma)

print(denoiser.pi)
print(denoiser.sigma_x)

denoised = denoiser.denoise(y, mu, sigma)

print("Alignment: before {}, after {}".format(np.std(y), np.std(denoised)))