#!/bin/bash

# -
# Perform filtering and preprocessing on hapmap3 .ped file
# and generate genotype matrices with approx. indpt SNPs in .bed
#
# Chang Su
# c.su@yale.edu
# Dec 15, 2020
# -

# -
# - Reference:
# - biostars.org/p/318856/
# - https://www.biostars.org/p/271694/
# - https://www.biostars.org/p/335605/

# - Step 0: download hapmap3 data
hapmap3_link="ftp://ftp.ncbi.nlm.nih.gov/hapmap/genotypes/2010-05_phaseIII"
wget ${hapmap3_link}/relationships_w_pops_041510.txt
for ext in map ped
do 
	wget ${hapmap3_link}/plink_format/hapmap3_r3_b36_fwd.consensus.qc.poly.${ext}.gz
	unzip hapmap3_r3_b36_fwd.consensus.qc.poly.${ext}.gz
done
# - For detailed comments on data,
# - see README at ${hapmap3_link}/plink_format/00README.txt

# - Remove duplicated SNPs
# -

mkdir DupsRemoved ;

plink --file hapmap3_r3_b36_fwd.consensus.qc.poly \
    --list-duplicate-vars ids-only ;

plink --file hapmap3_r3_b36_fwd.consensus.qc.poly \
    --exclude plink.dupvar --make-bed \
    --out DupsRemoved/hapmap3_r3_b36_fwd.consensus.qc.poly ;

rm plink.dupvar ;
## It seems that there is no duplicated SNPs in this dataset


# - Prune variants and obtain approx. indpt SNPs
# -

mkdir Pruned ;

cp hapmap3_r3_b36_fwd.consensus.qc.poly.map DupsRemoved/hapmap3_r3_b36_fwd.consensus.qc.poly.map

plink --bfile DupsRemoved/hapmap3_r3_b36_fwd.consensus.qc.poly \
    --maf 0.10 --indep 50 5 1.5 \
    --out Pruned/hapmap3 ;

plink --bfile DupsRemoved/hapmap3_r3_b36_fwd.consensus.qc.poly \
    --extract Pruned/hapmap3.prune.in --make-bed \
    --out Pruned/hapmap3 ;

## 145536 variants and 1397 people pass filters and QC.

# - Remove variants on sex chromosomes
# - 

mkdir Processed

# generate processed genotype in .bed
plink --bfile Pruned/hapmap3 \
	  --not-chr X Y XY --make-bed \
	  --out Processed/hapmap3

# generate .map file for pruned variants
plink --bfile Pruned/hapmap3 \
	  --not-chr X Y XY --recode \
	  --out Processed/hapmap3

## 142185 variants and 1397 people pass filters and QC.

# - Perform PCA

cd Processed
plink --bfile hapmap3 --pca 
mv plink.eigenval plink.eigenval.full
mv plink.eigenvec plink.eigenvec.full

# - extract subsets
# - perform PCA 

cut -f 2 hapmap3.map > snps.map

for size in 1000 10000 100000
do 
	# draw random snps
	shuf -n $size snps.map > snps.subset."$size".map
	# extract subsets
	plink --bfile hapmap3 \
	--extract snps.subset."$size".map \
	--make-bed --out hapmap3.subset."$size"
	# perform PCA
	plink --bfile hapmap3.subset."$size" --pca
	# change the default file names
	mv plink.eigenval plink.eigenval."$size"
	mv plink.eigenvec plink.eigenvec."$size"
done
