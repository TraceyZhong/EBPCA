# EBPCA

This library is a Python implementation of the algorithms described in ``Empirical Bayes PCA for high-dimensional data``. You can cite this work by
```
@misc{zhong2020empirical,
      title={Empirical Bayes PCA in high dimensions}, 
      author={Xinyi Zhong and Chang Su and Zhou Fan},
      year={2020},
      eprint={2012.11676},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```

## Installation

This package requires a working installation of Python 3.6, numpy, scipy, matplotlib and numba. [MOSEK](https://www.mosek.com) is an optional optimizer, which will solve NMPLE will higher log likelihood compared with EM.

Installing from source code
```bash
$ git clone git@github.com:TraceyZhong/EBPCA.git
$ cd EBPCA
# Build from source:
$ python setup.py build
# And install:
$ python setup.py install
# Test
$ python tutorial.py
# To uninstall:
$ pip uninstall ebpca
```

## Sample Usage 

User's should first determine the number of outlying signals *k* of the observational
matrix *Y*, and the number of iterations for AMP correction. Then EBPCA derives the following estimates of sample PCs.

```python
from ebpca import ebpca_gaussian
uest, vest = ebpca_gaussian(Y, rank = k, amp_iters = 5)
```

See `tutorial.html` for details. 

## Directory Structure

* [__src__]: Implementation of the EB-PCA algorithm.
    * [__amp__]: Iterative refinement using AMP.
    * [__empbayes__]: Empirical Bayes for the multivariate compound decision problem.
    * [__pca__]: Random matrix asymptotics for sample PCs.
    * [__preprocessing__]: Preprocessing for the observational matrix.
    * [__misc__]: Implementation of Mean-field VB approach.

