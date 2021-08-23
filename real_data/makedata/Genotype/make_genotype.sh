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
python make_1000G.py
bash bin_to_npy.py 1000G 

# -
# Hapmap3
# -
python make_hapmap3.py
bash bin_to_npy.py hapmap3
