import config

import numpy as np
import matplotlib.pyplot as plt
from ebpca.pca import check_gaussian_spectra

n = 500
p = 500 


X = np.random.normal(size = n*p).reshape(n,p) / np.sqrt(p)

_,s,_ = np.linalg.svd(X, full_matrices= False) 

check_gaussian_spectra(s, n, p, to_show = False, to_save = True)


