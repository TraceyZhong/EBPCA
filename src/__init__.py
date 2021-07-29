import os
from collections import namedtuple

if not os.path.exists("figures"):
    os.makedirs("figures")

from . import amp
from . import pca
from . import empbayes
from . import preprocessing


def ebpca_gaussian(Y, rank, amp_iters=5):
    assert rank > 0
    # Step 1: PCA analysis.
    X = preprocessing.normalize_obs(Y, rank)
    pcapack = pca.get_pca(X, rank)
    # Step 2: AMP iterative corrections
    U, V = amp.ebamp_gaussian(pcapack, amp_iters = amp_iters, warm_start = True)
    return U[:,:,-1], V[:,:,-1]


