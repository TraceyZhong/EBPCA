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
  make_option(c("-n", "--n_copy"), type="integer", default=1, 
              help="the number of indpt data copy",  metavar="integer"),
  make_option(c("-s", "--s_star"), type="numeric", default=1.3, 
              help="enter signal strength",  metavar="numeric"),
  make_option(c("-p", "--prefix"), type="character", 
              default='n_2000_gamma_2.0_nsupp_ratio_1.0_1.0_useEM_False', 
              help="enter experiment name",  metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

# parameters
n_copy = opt$n_copy
s_star = opt$s_star
prefix = opt$prefix
n = 2000
d = 4000
prop = 0.5
prior = 'Point_normal_0.1'

# helper function to get alignments
get_alignment <- function(u1, u2){
  abs(sum(u1 * u2) / sqrt(sum(u1**2) * sum(u2**2)))
}

# tune parameters: para
# https://crude2refined.wordpress.com/2013/07/30/sparse-pca-example-in-r-part-1/
# para with sparse='varnum' means number of nonzero entries
# true parameter is 0.1; span a 5 point grid around 0.1
para_prop = seq(0.025, 0.175, by=0.025)
len_para = length(para_prop)

# save alignments and denoised PCs
left_align <- right_align <- numeric(1)
left_dePC <- numeric(length=n)
right_dePC <- numeric(length=d)

data_prefix = sprintf('output/univariate/%s/%s/data/s_%.1f', prior, prefix, s_star)
for (i in n_copy){
  cat('\n')
  print(sprintf('Replication %i', i))
  cat('\n')
  
  # load simulated data
  u_star <- npyLoad(sprintf('%s_copy_%i_u_star_n_2000_gamma_2.0.npy', data_prefix, i-1))
  v_star <- npyLoad(sprintf('%s_copy_%i_v_star_n_2000_gamma_2.0.npy', data_prefix, i-1))
  X <- npyLoad(sprintf('%s_copy_%i_n_2000_gamma_2.0.npy', data_prefix, i-1))
  
  left_align_m <- right_align_m <- numeric(length=len_para)
  names(left_align_m) = names(right_align_m) = para_prop
  left_dePC_list <- right_dePC_list <- list()
  for (j in 1:len_para){
    print(sprintf('Tuning parameter: %.2f', n*para_prop[j]))
    # run sparse PCA
    # left PC
    left_res <- spca(t(X), 1, para=n*para_prop[j], type="predictor",
                     sparse="varnum", use.corr=FALSE, lambda=1e-6,
                     max.iter=200, trace=T, eps.conv=1e-3)
    left_align_m[j] <- get_alignment(left_res$loadings[,1], u_star)
    # right PC
    right_res <- spca(X, 1, para=d*para_prop[j], type="predictor",
                      sparse="varnum", use.corr=FALSE, lambda=1e-6,
                      max.iter=200, trace=T, eps.conv=1e-3)
    right_align_m[j] <- get_alignment(right_res$loadings[,1], v_star)
    # attach dePC
    left_dePC_list[[as.character(para_prop[j])]] <- left_res$loadings[, 1]
    right_dePC_list[[as.character(para_prop[j])]] <- right_res$loadings[, 1]
  }
  print('Left PC alignments:')
  print(left_align_m)
  print('Right PC alignments:')
  print(right_align_m)
  # attach only the best results
  j_max = which.max((left_align_m + right_align_m)/2)
  cat('\n')
  print(sprintf("The best para: %f", para_prop[j_max]))
  cat('\n')
  left_dePC <- left_dePC_list[[as.character(para_prop[j_max])]]
  right_dePC <- right_dePC_list[[as.character(para_prop[j_max])]]
  left_align <- left_align_m[j_max]
  right_align <- right_align_m[j_max]

  # remove used objects
  rm(left_align_m)
  rm(right_align_m)
  rm(left_dePC_list)
  rm(right_dePC_list)
}

cat('\n')
print('Replications finished.')
cat('\n')
print('Left PC alignments:')
print(left_align)
print('Right PC alignments:')
print(right_align)

# save alignments and denoised PCs
prior_prefix = sprintf('output/univariate/%s/%s', prior, prefix)
print(sprintf('Saving alignments and denoised PCs to %s', prior_prefix))

write.table(left_align, sprintf('%s/alignments/spca_u_s_%.1f_n_copy_%i.txt', 
                    prior_prefix, s_star, n_copy),
            col.names = F, row.names = F, quote = F)
write.table(right_align, sprintf('%s/alignments/spca_v_s_%.1f_n_copy_%i.txt', 
                    prior_prefix, s_star, n_copy),
            col.names = F, row.names = F, quote = F)

write.table(left_dePC, sprintf('%s/denoisedPC/spca_leftPC_s_%.1f_n_copy_%i.txt', 
                    prior_prefix, s_star, n_copy),
            col.names = F, row.names = F, quote = F)
write.table(right_dePC, sprintf('%s/denoisedPC/spca_rightPC_s_%.1f_n_copy_%i.txt', 
                    prior_prefix, s_star, n_copy),
            col.names = F, row.names = F, quote = F)
