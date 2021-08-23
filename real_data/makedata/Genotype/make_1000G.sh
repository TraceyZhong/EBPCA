#!/bin/bash

# -
# Perform filtering and preprocessing on 1000 Genomes genotype data
# and generate genotype matrices with approx. indpt SNPs in .bed
#
# Chang Su
# c.su@yale.edu
# July 21, 2020
# -

# -
# - Reference
# - https://www.biostars.org/p/271694/
# - https://www.biostars.org/p/335605/ 

# - Note
# - Step 1-8 are from https://www.biostars.org/p/335605/. One can refer to the link for detailed comments.
# - Step 9 is on subsampling SNPs and evaluate PCA

# - Step 1: Download data
# -
prefix="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr" ;

suffix=".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" ;

for chr in {1..22}; do
    wget "${prefix}""${chr}""${suffix}" "${prefix}""${chr}""${suffix}".tbi ;
done

# - Step 2: Download 1000 Genomes PED file
# -
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_g1k.ped ;

# - Step 3: Download the GRCh37 / hg19 reference genome
# -
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.gz ;

wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.fai ;

gunzip human_g1k_v37.fasta.gz ;

# - Step 4: Convert the 1000 Genomes files to BCF
# - takes ~one hour to convert each chromosome
# -
for chr in {12..17}; do
    echo ${chr}
    bcftools norm -m-any --check-ref w -f human_g1k_v37.fasta \
      ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz | \
      bcftools annotate -x ID -I +'%CHROM:%POS:%REF:%ALT' | \
        bcftools norm -Ob --rm-dup both \
          > output/ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.bcf ;
    bcftools index ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.bcf ;
done

# - Step 5: Convert the BCF files to PLINK format
# - 
for chr in {1..22}; do
    /gpfs/ysm/project/zf59/cs/software/plink/plink --noweb \
      --bcf output/ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.bcf \
      --keep-allele-order \
      --vcf-idspace-to _ \
      --const-fid \
      --allow-extra-chr 0 \
      --split-x b37 no-fail \
      --make-bed \
      --out output/ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes ;
done


# - Step 6: Prune variants from each chromosome
# - 
mkdir Pruned ;

for chr in {1..22}; do
    plink --noweb \
      --bfile ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes \
      --maf 0.10 --indep 50 5 1.5 \
      --out Pruned/ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes ;

    plink --noweb \
      --bfile ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes \
      --extract Pruned/ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.prune.in \
      --make-bed \
      --out Pruned/ALL.chr"${chr}".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes ;
done

# - Step 7:
# - Get a list of all PLINK files
find . -name "*.bim" | grep -e "Pruned" > ForMerge.list ;

sed -i 's/.bim//g' ForMerge.list ;

# - Step 8:
# - Merge all projects into a single PLINK file
mkdir Processed
plink --merge-list ForMerge.list --out Processed/1000G ;

# - Step 9: 
# - Create random subsets and perform PCA on random subsets

cd Processed

cut -f 2 1000G.map > snps.map

for size in 1000 10000 100000
do 
	# draw random snps
	shuf -n $size snps.map > snps.subset."$size".map
	# extract subsets
	plink --bfile 1000G \
	--extract snps.subset."$size".map \
	--make-bed --out 1000G.subset."$size"
	# perform PCA
	plink --bfile 1000G.subset."$size" --pca
	# change the default file names
	mv plink.eigenval plink.eigenval."$size"
	mv plink.eigenvec plink.eigenvec."$size"
done

