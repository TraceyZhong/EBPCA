# -
# 3 steps for making PBMC raw data
#
# Chang Su
# c.su@yale.edu
# Dec 15, 2020
# -

# -
# Step 0: download an R object of PBMC data from 10X Genomics
# Reference: https://satijalab.org/seurat/articles/pbmc3k_tutorial.html
# -

wget https://www.dropbox.com/s/63gnlw45jf7cje8/pbmc3k_final.rds?dl=1

# - 
# Step 1: convert the R object into a text file
# -
Rscript make_pbmc_1.R

# - 
# Step 2: normalize gene expressions and save as .npy file
# -
python make_pbmc_2.R