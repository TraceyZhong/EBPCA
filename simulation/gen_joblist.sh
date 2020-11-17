#!/bin/bash
#for method in EB-PCA BayesAMP EBMF
# do 
method=$1
for prior in Point_normal Two_points Uniform
  do for s in 1.3 1.8 3.0;
    do
      echo "module load miniconda; conda init bash; source deactivate; conda activate py3; python rank_one.py --method=$method --prior=$prior --s_star=$s --nsupp_ratio=1"
    done
  done
#done
