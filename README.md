# EBPCA

This library is a Python implementation of the algorithms described in ``Empirical Bayes PCA for high-dimensional data``. You can cite this work by
```
@article{10.1111/rssb.12490,
    author = {Zhong, Xinyi and Su, Chang and Fan, Zhou},
    title = {Empirical Bayes PCA in High Dimensions},
    journal = {Journal of the Royal Statistical Society Series B: Statistical Methodology},
    volume = {84},
    number = {3},
    pages = {853-878},
    year = {2022},
    month = {01},
    abstract = {When the dimension of data is comparable to or larger than the number of data samples, principal components analysis (PCA) may exhibit problematic high-dimensional noise. In this work, we propose an empirical Bayes PCA method that reduces this noise by estimating a joint prior distribution for the principal components. EB-PCA is based on the classical Kiefer–Wolfowitz non-parametric maximum likelihood estimator for empirical Bayes estimation, distributional results derived from random matrix theory for the sample PCs and iterative refinement using an approximate message passing (AMP) algorithm. In theoretical ‘spiked’ models, EB-PCA achieves Bayes-optimal estimation accuracy in the same settings as an oracle Bayes AMP procedure that knows the true priors. Empirically, EB-PCA significantly improves over PCA when there is strong prior structure, both in simulation and on quantitative benchmarks constructed from the 1000 Genomes Project and the International HapMap Project. An illustration is presented for analysis of gene expression data obtained by single-cell RNA-seq.},
    issn = {1369-7412},
    doi = {10.1111/rssb.12490},
    url = {https://doi.org/10.1111/rssb.12490},
    eprint = {https://academic.oup.com/jrsssb/article-pdf/84/3/853/49322184/jrsssb\_84\_3\_853.pdf},
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

