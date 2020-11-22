#!/bin/bash
s=$1
echo '\n EB-PCA'
echo '\n Point Normal \n'
python rank_one.py --prior=Point_normal --iters=20 --s_star=$s
echo '\n Two points \n'
python rank_one.py --prior=Two_points --iters=20 --s_star=$s
echo '\n Uniform \n'
python rank_one.py --prior=Uniform --iters=20 --s_star=$s
echo '\n EBMF\n'
echo '\n Point Normal \n'
python rank_one.py --prior=Point_normal --iters=20 --method=EBMF --s_star=$s
echo '\n Two points \n'
python rank_one.py --prior=Two_points --iters=20 --method=EBMF --s_star=$s
echo '\n Uniform \n'
python rank_one.py --prior=Uniform --iters=20 --method=EBMF --s_star=$s
