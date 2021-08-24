# -
# Generate population labels for 1000 Genomes samples
# Reference: https://www.biostars.org/p/335605/
# 
# Chang Su
# c.su@yale.edu
# July 22, 2020
# -

read_plink_eigen <- function(ver){
  eigenvec <- read.table(paste0("Processed/plink.eigenvec.", ver), header = FALSE, skip=0, sep = " ")
  rownames(eigenvec) <- eigenvec[,2]
  eigenvec <- eigenvec[,3:ncol(eigenvec)]
  colnames(eigenvec) <- paste("Principal Component ", c(1:20), sep = "")
  return(eigenvec)
}

if(!file.exists('Popu_labels.txt')){
  # load sample IDs in eigenvec to subset g1k individuals
  eigenvec <- read_plink_eigen(1000)
  
  # read in the PED data
  PED <- read.table("20130606_g1k.ped", header = TRUE, skip = 0, sep = "\t")
  PED <- PED[which(PED$Individual.ID %in% rownames(eigenvec)), ]
  PED <- PED[match(rownames(eigenvec), PED$Individual.ID),]
  all(PED$Individual.ID == rownames(eigenvec)) == TRUE
  
  # from: http://www.internationalgenome.org/category/population/
  #   http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/README_populations.md
  #   ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20131219.populations.tsv
  PED$Population <- factor(PED$Population, levels=c(
    "ACB","ASW","ESN","GWD","LWK","MSL","YRI",
    "CLM","MXL","PEL","PUR",
    "CDX","CHB","CHS","JPT","KHV",
    "CEU","FIN","GBR","IBS","TSI",
    "BEB","GIH","ITU","PJL","STU"))

  PED$Population_broad <- rep('African', nrow(PED))
  PED$Population_broad[PED$Population %in% c("CLM","MXL","PEL","PUR")] = 'Hispanic'
  PED$Population_broad[PED$Population %in% c("CDX","CHB","CHS","JPT","KHV")] = 'East-Asian'
  PED$Population_broad[PED$Population %in% c("CEU","FIN","GBR","IBS","TSI")] = 'Caucasian'
  PED$Population_broad[PED$Population %in% c("BEB","GIH","ITU","PJL","STU")] = 'South Asian'

  write.table(PED, 'Popu_labels.txt')
}
