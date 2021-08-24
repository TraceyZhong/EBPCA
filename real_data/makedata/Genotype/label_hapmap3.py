# -
# Generate population labels for hapmap3 samples
# 
# Chang Su
# c.su@yale.edu
# July 22, 2020
# -

import numpy as np
import pandas as pd

# load metadata for hapmap3
hapmap3_popu_df = pd.read_csv('relationships_w_pops_041510.txt', sep = '\t')

# merge refined population into 5 populations
# using the same classification as in label_1000G.R
# from: http://www.internationalgenome.org/category/population/
#		http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/README_populations.md
#		ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20131219.populations.tsv

hapmap3_popu_df['broad_population'] = np.repeat('unknown', hapmap3_popu_df.shape[0])
popu_lookup = {'African': ["ACB","ASW","ESN","GWD","LWK","MSL","YRI", "MKK"],
               'Hispanic': ["CLM","MXL","PEL","PUR", "MEX"],
               'East-Aisan': ["CDX","CHB","CHS","JPT","KHV", "CHD"],
               'Caucasian': ["CEU","FIN","GBR","IBS","TSI"],
               'South-Asian': ["BEB","GIH","ITU","PJL","STU"]}
for i in range(5):
    hapmap3_popu_df.loc[hapmap3_popu_df['population'].isin(list(popu_lookup.values())[i]).values, 'broad_population'] = \
        list(popu_lookup.keys())[i]
np.save('hapmap3_popu.npy', hapmap3_popu_df)

# print sample size per population
print(hapmap3_popu_df.broad_population.value_counts())
print(hapmap3_popu_df.shape)