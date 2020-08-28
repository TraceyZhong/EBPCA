'''
setting:
X =  (n*k) array
X ~ MZ + Sigma
Z = sum_{i=1}^m tau_i (2,)
estimate each tau_i
with two methods
1. EM
2. Interior Point Method
'''
import numpy as np
from numba import jit
import scipy 
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

MIN_FLOAT = -1e+12
MAX_FLOAT = 1e+12

# W*pi[i,j] = pi[j] * W[i,j] 
# sum_j pi[j] W[i,j] = np.sum(W*pi, axis = 1) 
@jit(nopython = True)
def negloglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    return -np.sum(np.log( np.sum(W * pi, axis = 1)))

@jit(nopython = True)
def loglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    return np.sum(np.log( np.sum(W * pi, axis = 1)))

# @jit(nopython = True)
def dnegloglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    res = np.sum( W / np.sum(W*pi, axis = 1)[:,np.newaxis], axis = 0)
    if np.isnan(res).any():
        raise ValueError("dloglik has nan value")
    return -res
    # -np.maximum(np.minimum(res, MAX_FLOAT), MIN_FLOAT)


# @jit(nopython = True)
def ddnegloglik(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    normedW = W / np.sum(W*pi, axis = 1)[:,np.newaxis]
    # return my_m_dot(np.transpose(normedW), normedW)
    res = normedW.T @ normedW
    if np.isnan(res).any():
        raise ValueError("dloglik has nan value")
    return res
    # np.maximum(np.minimum(res, MAX_FLOAT), MIN_FLOAT)

@jit(nopython = True)
def evaluate(pi, f, z, mu, covInv, truth):
    ll = negloglik(pi, f, z, mu, covInv)
    return ll - truth

@jit(nopython = True)
def _get_phi(f, z, mu, covInv):
    return np.exp(-(covInv.dot(f - mu.dot(z)).dot(f - mu.dot(z)))/2)

def get_truth(pi, f, z, mu, covInv):
    W = get_W(f, z, mu, covInv)
    return np.sum( W / np.sum(W*pi, axis = 1)[:,np.newaxis], axis = 0)

@jit(nopython = True)
def my_dot(mat, vec):
    nrow, ncol = mat.shape
    res = np.array([0.0]*nrow)
    for i in range(nrow):
        for j in range(ncol):
            res[i] += mat[i,j] * vec[j]
    return res

@jit(nopython = True)
def my_m_dot(A, B):
    Arow, Acol = A.shape
    Brow, Bcol = B.shape
    if Acol != Brow:
        raise ValueError("Can't do matrix multiplication, shape doesn't align")
    I = Arow
    K = Bcol
    res = np.array([0.0]*(I*K)).reshape((I,K))
    for i in range(I): 
        for k in range(K):
            res[i,k] = A[i,:] * B[:,k]
    
    return res
        

# W[i,j] = f(x_i|z_j)
@jit(nopython = True)
def get_W(f, z, mu, covInv):
    W = np.empty(shape = (nsample, nsupp),)
    for i in range(nsample):
        for j in range(nsupp):
            vec = f[i] - my_dot(mu, z[j])
            res = np.exp(-np.sum(my_dot(covInv, vec) * vec)/2)
            W[i,j] = res
    return W

## --- em algorithm --- ##

def test_em(f, Z, mu, covInv, em_iter, nsample, nsupp, ndim, TRUTH):
    pi = np.array([1/nsupp] * nsupp)
    em_rounds = int(em_iter / 10)
    for itr_round in range(em_rounds):
        pi = _npmle_em_hd(f, Z, pi, mu, covInv, 10, nsample, nsupp, ndim)
        acc = evaluate(pi, f, Z, mu, covInv, TRUTH)
        print("at round {}, delta loglik {}".format(itr_round, acc))
        # print("some of pi")
        # print(pi[:10])
    return pi


@jit(nopython=True)
def _npmle_em_hd(f, Z, pi, mu, covInv, em_iter, nsample, nsupp, ndim):
    # pi = np.full((nsupp,), 1/nsupp, dtype= float)
    # pi = np.array([1/nsupp] * nsupp)
    
    for _ in range(em_iter):
        # W_ij = f(xi|zj)
        W = np.empty(shape = (nsample, nsupp))
        for i in range(nsample):
            for j in range(nsupp):
                vec = f[i] - my_dot(mu, Z[j])
                res = np.exp(-np.sum(my_dot(covInv, vec) * vec)/2)
                W[i,j] = res
        
        denom = my_dot(W, pi) # denom[i] = \sum_j pi[j]*W[i,j]
        # print("denom shape {}".format(denom.shape))

        for j in range(nsupp):
            pi[j] = pi[j]*np.mean(W[:,j]/denom)
        
        # pi = np.array([pi[j]*np.mean(W[:,j]/denom) for j in range(nsupp)])
        # pi = np.mean(pi * (W / denom[:, np.newaxis]), axis = 0) # use normal representation

    return pi


## --- Trust-Region Constrained Algorithm --- ##
# last_val = 0
# TOL = 1e-5
def callbackF(x, state):
    # global last_val
    print("niters {nit:4d}; funvar {fun:3.6f}".format(nit = state.nit, fun = state.fun))
    # print("jac")
    # print(state["jac"])
    # if abs(last_val - state.fun) < TOL:
    #     return True
    # last_val = state.fun

def gradient_descent(f, Z, mu, covInv, em_iter, nsample, nsupp, ndim, TRUTH):
    # trust region constrained algorithm
    bounds = Bounds( [0.001e-12] * nsupp, [1]* nsupp)
    # Idm = scipy.sparse.csc_matrix(np.identity(nsupp))
    linear_constraint = LinearConstraint(np.ones((1,nsupp)), 1, 1)

    ## --- Solving the optimization problem --- ##
    pi0 = np.full(nsupp, [1/nsupp])
    res = scipy.optimize.minimize(negloglik, pi0, args = (f, Z, mu, covInv),  method = "trust-constr", 
            jac = dnegloglik, hess = ddnegloglik,
            constraints = [linear_constraint],
            options = {"verbose": 1}, bounds = bounds
            ,callback = callbackF
        )

    return res

## start working

testEM = False
testOPT = True

## --- Simulate Data --- ##

piTruth = np.array([1])
zTruth = np.array([0,0]).reshape((1,2))
muTruth = np.array([[1,0], [0,1]])
covTruth = np.array([[1,0],[0,1]])

mu = muTruth
covInv = np.linalg.inv(covTruth)

nsample = 100
rank = 2

f = np.random.normal(size = nsample*rank).reshape((nsample, rank))

# testZ = np.array([[1,1], [0,0], [-1,1]])
# testZ = f
testZ = f[np.random.choice(f.shape[0], 3, replace=False), :]
nsupp = testZ.shape[0]


TRUTH = negloglik(piTruth, f, zTruth, mu, covInv)
print("TRUTH is {:3.6f}".format(TRUTH) )

## --- test npmle --- ##
em_iter = 50

if testEM:
    res = test_em(f, testZ, mu, covInv, em_iter, nsample, nsupp, rank, TRUTH)
    print("reslts for EM is {}".format(res))

if testOPT:
    res = gradient_descent(f, testZ, mu, covInv, em_iter, nsample, nsupp, rank, TRUTH)
    print("fun var to opt for OPT is {}".format(TRUTH - res.fun))
    print("reslts for OPT is {}".format(res.x))
    print("testZ {}".format(testZ))




