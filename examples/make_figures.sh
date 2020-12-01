#!/bin/bash
for data in 1000G UKBB PBMC
do
  python showcase.py --data_name="$data" > output/"$data"_plot_log.txt
done