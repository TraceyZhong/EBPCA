#!/usr/bin/env Rscript

# --------------------------------------------------
# Rank one simulation: comparison with sparse PCA
# 
# Description:
#   Run sparse PCA on data simulated in rank_one.py,
#   to compare sparse PCA with EB-PCA
#
# Reference:
# https://rdrr.io/cran/elasticnet/man/spca.html
# --------------------------------------------------

setwd('/gpfs/ysm/project/zf59/cs/empiricalbayespca/generalAMP/simulation')
args = commandArgs(trailingOnly=TRUE)

# install dependency packages
dependencies <- c('RcppCNPy', 'elasticnet', 'optparse')
for (d in dependencies){
  if(! d %in% installed.packages()[ , "Package"]){
    print(sprintf('Installing R package %s', d))
    install.packages(d, repos='https://cloud.r-project.org')
  }
  library(d, character.only=T)
}

# parse simulation parameters
option_list = list(
  make_option(c("-n", "--n_rep"), type="integer", default=1, 
              help="enter number of independent data to be simulated",  metavar="integer"),
  make_option(c("-s", "--s_star"), type="numeric", default=1.3, 
              help="enter signal strength",  metavar="numeric")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

# parameters
n_rep = opt$n_rep
s_star = opt$s_star
n = 1000
d = 2000
prop = 0.5
prior = 'Point_normal'

# helper function to get alignments
get_alignment <- function(u1, u2){
  sum(u1 * u2) / sqrt(sum(u1**2) * sum(u2**2))
}

# save alignments and denoised PCs
left_align <- numeric(length = n_rep)
right_align <- numeric(length = n_rep)
left_dePC <- matrix(0, n, n_rep)
right_dePC <- matrix(0, d, n_rep)

data_prefix = sprintf('output/univariate/%s/data/s_%.1f', prior, s_star)
for (i in 1:n_rep){
  print(sprintf('iteration %i', i))
  # load simulated data
  u_star <- npyLoad(sprintf('%s_copy_%i_u_star.npy', data_prefix, i-1))
  v_star <- npyLoad(sprintf('%s_copy_%i_v_star.npy', data_prefix, i-1))
  X <- npyLoad(sprintf('%s_copy_%i.npy', data_prefix, i-1))
  # run sparse PCA
  # left PC
  left_res <- spca(t(X), 1, para=n*0.5, type="predictor",
              sparse="varnum", use.corr=FALSE, lambda=1e-6,
              max.iter=200, trace=T, eps.conv=1e-3)
  left_align[i] <- get_alignment(left_res$loadings[,1], u_star)
  left_dePC[,i] <- left_res$loadings[, 1]
  # right PC
  right_res <- spca(X, 1, para=d*0.5, type="predictor",
              sparse="varnum", use.corr=FALSE, lambda=1e-6,
              max.iter=200, trace=T, eps.conv=1e-3)
  right_align[i] <- get_alignment(right_res$loadings[,1], v_star)
  right_dePC[,i] <- right_res$loadings[, 1]
  print('Left PC alignments:')
  print(left_align[i])
  print('Right PC alignments:')
  print(right_align[i])
}


print('Left PC alignments:')
print(left_align)
print('Right PC alignments:')
print(right_align)

# save alignments and denoised PCs
prior_prefix = sprintf('output/univariate/%s', prior)
write.table(abs(left_align), sprintf('%s/alignments/spca_u_s_%.1f_n_rep_%i.txt', 
                    prior_prefix, s_star, n_rep),
            col.names = F, row.names = F, quote = F)
write.table(abs(right_align), sprintf('%s/alignments/spca_v_s_%.1f_n_rep_%i.txt', 
                    prior_prefix, s_star, n_rep),
            col.names = F, row.names = F, quote = F)

write.table(left_dePC, sprintf('%s/denoisedPC/spca_leftPC_s_%.1f_n_rep_%i.txt', 
                    prior_prefix, s_star, n_rep),
            col.names = F, row.names = F, quote = F)
write.table(right_dePC, sprintf('%s/denoisedPC/spca_rightPC_s_%.1f_n_rep_%i.txt', 
                    prior_prefix, s_star, n_rep),
            col.names = F, row.names = F, quote = F)
