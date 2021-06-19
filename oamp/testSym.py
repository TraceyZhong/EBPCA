import numpy as np
from symAmp import oamp_pca
from symGauAmp import gaussian_amp_pca as gamp_pca

from utils import *

import warnings
warnings.filterwarnings("error")

n = 2000

# np.random.seed(19732)

def compare(s, n, noise_type):
    W = get_noise(n, noise_type)
    # normalize noise
    normsq = np.sum(W**2)/n
    W = W/np.sqrt(normsq)

    # get observational matrix
    ustar = np.random.binomial(1,0.5,size=n)*2-1
    Y = s/n * np.outer(ustar,ustar) + W
    # PCA
    w, v = np.linalg.eigh(Y)
    assert len(w) == n
    singval = w[-1]
    f = v[:,-1] * np.sqrt(n)
    init_align = np.mean(f * ustar)
    if init_align < 0:
        f = -f
        init_align = -init_align
    if init_align < 0.5:
        print(init_align)
        exit()
    # Oamp
    U = oamp_pca(Y, s, singval, init_align, f, ustar = None, iters=5, noise_type = "CenteredBeta")
    Unormsq = np.diag(np.dot(np.transpose(U),U))
    aligns = np.dot(np.transpose(U),ustar) / np.sqrt(Unormsq*n)
    print("init_align", init_align)
    print(aligns)
    write_to_record(s, "oamp", noise_type, aligns)
    # Gamp
    U = gamp_pca(Y, s, singval, init_align, f, ustar = None, iters=5, noise_type = noise_type)
    Unormsq = np.diag(np.dot(np.transpose(U),U))
    aligns = np.dot(np.transpose(U),ustar) / np.sqrt(Unormsq*n)
    print("init_align", init_align)
    print(aligns)
    write_to_record(s, "gamp", noise_type, aligns)

compare(1.2, n, "Rademacher")


# noise_type = "Rademacher"
# signal_strength = 1.2
# i = 0
# while i < 20:
#     try:
#         compare(signal_strength, n, noise_type)
#     except RuntimeWarning:
#         continue
#     else:
#         i += 1

noise_type = "CenteredBeta"
signal_strength = 2.6
i = 0
while i < 20:
    try:
        compare(signal_strength, n, noise_type)
    except RuntimeWarning as e:
        print("Error", e)
        continue
    else:
        i += 1




