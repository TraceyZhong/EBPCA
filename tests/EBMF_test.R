# ------------------------
# check_EBMF.R
#
# Oct 22, 2020
# Chang Su, c.su@yale.edu
# ------------------------

# In this script we run flash() in R package flashr,
# to show that our Python implementation of EBMF, as in EBMF.py and EBMF_util.py
# are exactly the same as the R implementation.

# Reference:
# https://arxiv.org/abs/1802.06931
# https://github.com/stephenslab/flashr

# Note that flash() in R assumes parametric priors
# To do a valid comparison, we simulate data from the Point Normal prior and
# use parametric empirical bayes denoiser PointNormalBayes in EBPCA
# to estimate the prior and denoise the observation
# rather than using our default nonparametric denoiser in the EBPCA.

# ---------------
# Install flashr
# ---------------
if(!require('flashr',character.only = TRUE)) install.packages('flashr')
library(flashr)

# --------------------------------------
# Simulate data under a rank 1 AMP model
# with Point Normal prior for both PCs
# --------------------------------------
simu_data = function(family, n, alpha, l, pars, seed=1){
  
  # Simulate left and right singular vector for AMP matrix A
  # according to the point normal prior
  
  d = (n * alpha)
  
  set.seed(seed)
  
  u0 = rbinom(n, 1, pars[[1]][1]) * rnorm(n, 0, pars[[1]][2])
  v0 = rbinom(d, 1, pars[[2]][1]) * rnorm(d, 0, pars[[2]][2])
  
  W = matrix(rnorm(n*d, 0, 1), nrow=n, ncol=d) / sqrt(n)
  
  A = l/n * tcrossprod(u0, v0) + W
  
  return (list(A=A, u0=u0, v0=v0))
}

n = 2000
l = 1.4
w = 0.5
sigma_x = sqrt(1/w)

set.seed(1)

dat1 = simu_data('point-normal', 2000, 1.2, l, list(c(w[i], sigma_x[i]),c(w[i], sigma_x[i])))

if (!dir.exists('result/simulation/')){
  dir.create('result/simulation/')
  write.table(dat1[[1]], 'result/simulation/simu_test_ebmf_A.txt', row.names = F, col.names = F, quote=F)
  write.table(dat1[[2]], 'result/simulation/simu_test_ebmf_u0.txt', row.names = F, col.names = F, quote=F)
  write.table(dat1[[3]], 'result/simulation/simu_test_ebmf_v0.txt', row.names = F, col.names = F, quote=F)
}

# -----------------------------
# Run ebmf to estimate u and v
# the result are the same as EBMF_py_
# -----------------------------
# Reference: https://stephenslab.github.io/flashr/articles/flash_intro.html
flash_dat = flash_set_data(dat1$A, S = sqrt(1/n))
f = flash(flash_dat, Kmax=1, var_type = 'zero', init_fn = 'udv_svd', ebnm_fn = 'ebnm_pn', nullcheck = FALSE, tol=1e-1)
ldf = flash_get_ldf(f)

get_alignment <- function(l_hat, l){
  return (sum(l_hat * l) / (sqrt(sum(l_hat**2)) * sqrt(sum(l**2))))
}

print('l estimate:')
print(get_alignment(ldf$l, dat1[[2]]))
print('f estimate:')
print(get_alignment(ldf$f, dat1[[3]]))

# --------------------------------------
# unit tests for objective functions
# test if objective functions 
# are consistent across implementations
# --------------------------------------

# https://github.com/stephenslab/flashr/blob/master/R/get_functions.R
# @title Get the expected squared residuals from a flash data and fit
#   object.
flash_get_R2 = function(data, f) {
  if (is.null(f$EL)) {
    return(data$Y^2)
  } else {
    LF = f$EL %*% t(f$EF)
    return((data$Y - LF)^2 + f$EL2 %*% t(f$EF2) - f$EL^2 %*% t(f$EF^2))
  }
}

# https://github.com/stephenslab/flashr/blob/master/R/objective.R
# Compute the expected log-likelihood (at non-missing locations) based
#   on expected squared residuals and tau.
e_loglik_from_R2_and_tau = function(R2, tau, data) {
  # tau can be either a scalar or a matrix:
  if (data$anyNA) {
    R2 = R2[!data$missing]
    if (is.matrix(tau)) {
      tau = tau[!data$missing]
    }
  }
  return(-0.5 * sum(log(2 * pi / tau) + tau * R2))
}

e_loglik = function(data, f) {
  return(e_loglik_from_R2_and_tau(flash_get_R2(data, f), f$tau, data))
}

# simulate data

data = list()
data$Y = dat1[[1]]
data$anyNA = FALSE

f = list()
svd_res <- svd(dat1[[1]])
f$EL <- svd_res$u[,1]
f$EL2 <- svd_res$u[,1]**2
f$EF <- svd_res$v[,1]
f$EF2 <- svd_res$v[,1]**2
f$tau = n

loglik <- e_loglik(data, f)
print('expected log likelihood')
print(loglik)

# --------------------------------------
# unit tests for EBNM
# test if EBNM 
# are consistent across implementations
# --------------------------------------

library(ebnm)
set.seed(1)
# u0 = dat1[[2]]
u0 = rbinom(n, 1, 0.9) * rnorm(n, 0, sqrt(1 / 0.5))
mu_y = 1
sigma_y = 4
y = u0 * mu_y + rnorm(n, 0, sigma_y)
x.ebnm = ebnm_point_normal(y, sigma_y, mode = 0,
                           output = c("posterior_mean", "posterior_second_moment",
                                      "fitted_g", "log_likelihood"))
print('fitted prior:')
print(x.ebnm$fitted_g)
print('posterior mean and second moment')
print(head(x.ebnm$posterior))
print('log likelihood:')
print(x.ebnm$log_likelihood)

write.table(y, 'result/simulation/test_ebnm_in_R.txt', row.names = F, col.names = F, quote=F)
write.table(x.ebnm$posterior, 'result/simulation/test_ebnm_in_R_posterior.txt', row.names = F, col.names = F, quote=F)


# --------------------------------------
# unit tests for NM_posterior_e_loglik
# test if NM_posterior_e_loglik
# are consistent across implementations
# --------------------------------------
NM_posterior_e_loglik = function(x, s, Et, Et2) {
  # Deal with infinite SEs:
  idx = is.finite(s)
  x = x[idx]
  s = s[idx]
  Et = Et[idx]
  Et2 = Et2[idx]
  return(-0.5 * sum(log(2*pi*s^2) + (1/s^2) * (Et2 - 2*x*Et + x^2)))
}

NM_loglik <- NM_posterior_e_loglik(y, sigma_y, x.ebnm$posterior$mean, x.ebnm$posterior$second_moment)
print(NM_loglik)
