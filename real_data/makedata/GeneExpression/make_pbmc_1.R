# -
# R codes for importing an R object of 
# PBMC data and save it as text files
#
# Chang Su
# c.su@yale.edu
# Dec 15, 2020
# -

# Download data from
# 

library(Seurat)
pbmc <- readRDS('pbmc3k_final.rds')

pbmc_count <- GetAssayData(pbmc, slot = 'counts')
write.table(as.matrix(pbmc_count), 'pbmc_counts.txt')
write.table(Idents(pbmc), 'pbmc_ct.txt')