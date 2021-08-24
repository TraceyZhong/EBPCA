#!/bin/bash 

# -
# Make raw data for both 1000 Genomes and Hapmap3 genotype data
# 
# Chang Su
# c.su@yale.edu
# Dec 15, 2020
# -

# -
# 1000 Genome
# -

# Step 1: download, clean and subset genotypes; store in bed
bash make_1000G.sh
# Step 2: convert .bed files to .npy files; normalize genotypes
python bed_to_npy.py 1000G 
# Step 3: obtain population labels 
Rscript label_1000G.R

# -
# Hapmap3
# -

# Step 1: download, clean and subset genotypes; store in bed
bash make_hapmap3.sh
# Step 2: convert .bed files to .npy files; normalize genotypes
python bed_to_npy.py hapmap3
# Step 3: obtain population labels 
Python label_hapmap3.py
