
from showcase import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, help="number of PC",
                    default=2, const=2, nargs='?')
args = parser.parse_args()
rank = args.rank

data_name = 'PBMC'
iters = 5
norm_data = np.load('results/%s/norm_data.npy' % data_name)
X = norm_data
sub_pcapack = get_pca(X, 6)
import os
# V_star_12 = np.load('results/%s/ground_truth_PC.npy' % data_name)
if not os.path.exists('results/%s/ground_truth_PC_1_6.npy' % data_name):
    V_star = sub_pcapack.V
    np.save('results/%s/ground_truth_PC_1_6.npy' % data_name, V_star)
else:
    V_star = np.load('results/%s/ground_truth_PC_1_6.npy' % data_name)

# run EB-PCA with joint estimation (by default)
est_dir = 'results/%s/PC_estimates_iters_%i_test_full_rank_%i' % (data_name, iters, rank)
import os
if not os.path.exists(est_dir + '.npy'):
    _, V_joint, _ = run_rankK_EBPCA('joint', X, rank, iters, optimizer="EM")
    np.save(est_dir + '.npy', V_joint, allow_pickle=False)
else:
    V_joint = np.load(est_dir + '.npy', allow_pickle=False)

# evaluate alignments
joint_align = [get_alignment(V_joint[:, [j], -1], V_star[:, [j]]) \
                for j in range(rank)]
np.save('results/%s/joint_alignment_test_full_rank_%i.npy' % (data_name, rank), joint_align, allow_pickle=False)

print(joint_align)

# plot estimated PC
V_joint_est = align_pc(V_joint[:, :, -1], V_star[:, :rank])
xRange = [-0.06,0.025]
yRange = [-0.045, 0.105]
ax = vis_2dim_subspace(V_joint_est[:,:2], joint_align[:2], data_name, 'joint_estimation', 
                  xRange=xRange, yRange=yRange,
                  data_dir='data/', to_save=False, legend_loc='upper left')
plt.savefig('figures/%s/test_full_rank_%i.png' % (data_name, rank))

if V_joint_est.shape[1] > 2:
    ax = vis_2dim_subspace(V_joint_est[:, 1:3], joint_align[1:3], 
                    data_name, 'joint_estimation', 
                    xRange=[-0.04, 0.10], yRange=[-0.05, 0.12],
                    data_dir='data/', to_save=False, legend_loc='upper left', PC1=2, PC2=3)

plt.savefig('figures/%s/test_full_rank_%i_PC_2_3.png' % (data_name, rank))

# visualize PC 2 and PC 3 in naive PCA

ax = vis_2dim_subspace(V_star[:, 1:3], [1, 1],
                        data_name, 'naive_PCA', xRange=[-0.04, 0.10], yRange=[-0.05, 0.12],
                                   data_dir='data/', to_save=False, legend_loc='upper left',
                                   PC1=2, PC2=3)
plt.savefig('figures/%s/test_full_ground_truth_PC_2_3.png' % (data_name))
