#!/bin/bash
for data in 1000G UKBB PBMC
do
  python showcase.py --data_name="$data" > output/"$data"_plot_log.txt
done

# make Mean-field VB figures
# without RMT initialization
for subset_size in 100 10000 10000
do
  python showcase.py --data_name=1000G --subset_size $subset_size --pca_method MF-VB \
 --n_copy 1 --ebpca_ini no --to_plot yes
done