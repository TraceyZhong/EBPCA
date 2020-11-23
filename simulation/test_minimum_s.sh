#!/bin/bash
s=$1
iters=$2
n_rep=$3
echo "s=$s"
echo ""
echo "EB-PCA"
echo ""
echo "Point Normal"
echo ""
python rank_one.py --prior=Point_normal --iters=$iters --s_star=$s --n_rep=$n_rep
echo ""
echo "Two points"
echo ""
python rank_one.py --prior=Two_points --iters=$iters --s_star=$s --n_rep=$n_rep
echo ""
echo "Uniform"
echo ""
python rank_one.py --prior=Uniform --iters=$iters --s_star=$s --n_rep=$n_rep
echo ""
echo "EBMF"
echo ""
echo "Point Normal"
echo ""
python rank_one.py --prior=Point_normal --iters=$iters --method=EBMF --s_star=$s --n_rep=$n_rep
echo ""
echo "Two points"
echo ""
python rank_one.py --prior=Two_points --iters=$iters --method=EBMF --s_star=$s --n_rep=$n_rep
echo ""
echo "Uniform"
echo ""
python rank_one.py --prior=Uniform --iters=$iters --method=EBMF --s_star=$s --n_rep=$n_rep
