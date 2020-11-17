# EBPCA

This library is a Python implementation of the algorithms described in ``Empirical Bayes PCA for high-dimensional data``. You can cite this work by
```
@article{
    something,
}
```

## Installation

This package requires a working installation of Python 3.6, numpy, scipy, matplotlib and numba. [MOSEK](https://www.mosek.com) is an optional optimizer, which will substantially improve computation time compared with EM.

Installing from source code
```bash
$ git clone git@github.com:TraceyZhong/generalAMP.git
$ cd generalAMP
# Build from source:
$ python setup.py build
# And install:
$ python setup.py install
```

## Usage 

Check `tutorial.html` for a synthetic example using EB-PCA. In short, suppose *Y* is the observational matrix and *k* is the number of outlying signals, then

```python
# Normalize the observational matrix.
from ebpca.preprocessing import normalize_obs
from ebpca.pca import get_pca
X = normalize_obs(Y, rank)
# Prepare the PCA pack.
pcapack = get_pca(X, rank)

# Empirical Bayes for the compound decision problem 
# and iterative refinement via amp.
from ebpca.empbayes import NonparEBActive as NonparEB
from ebpca.amp import ebamp_gaussian_active as ebamp_gaussian

udenoiser = NonparEB(optimizer = "Mosek", ftol = 1e-3, nsupp_ratio = 1)

U, V = ebamp_gaussian(pcapack, iters=3, udenoiser=udenoiser, figprefix="tut", mutev = True)
``` 

`U[:,:,i]` and `V[:,:,i]` are the denoised results at iteration *i*.

## Directory Structure

* [__ebpca__]: Implementation of the EB-PCA algorithm.
    * [__amp__]: Iterative refinement using AMP.
    * [__empbayes__]: Empirical Bayes for the multivariate compound decision problem.
    * [__pca__]: Random matrix asymptotics for sample PCs.
    * [__preprocessing__]: Preprocessing for the observational matrix.