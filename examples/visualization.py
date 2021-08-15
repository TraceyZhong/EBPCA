import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from simulation.helpers import get_space_distance

def plot_legend(labels):
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 26
    # 5 methods
    fig = plt.figure()
    fig_legend = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    for i in range(len(labels)):
        ax.scatter(range(2), range(2), label=labels[i], s=2)
    fig_legend.legend(labels, loc='center', frameon=True)
    plt.show()


from simulation.helpers import get_joint_alignment, get_error

def load_sample_labels(data_name, data_dir = 'data'):
    if data_name in ['1000G', 'UKBB', 'PBMC', 'GTEx', 'Hapmap3',
                     '1000G_African', '1000G_Caucasian',
                      '1000G_East-Asian', '1000G_Hispanic',
                      '1000G_South-Asian']:
        if '1000G' in data_name:
            popu_label_df = pd.read_csv('data/1000G/Popu_labels.txt', sep=' ')
            popu_label_broad = popu_label_df['Population_broad'].values
            popu_label_refined = popu_label_df['Population'].values
            popu_label = popu_label_broad
            for sub_popu in ['African', 'Caucasian', 'East-Asian', 'South-Asian', 'Hispanic']:
                if sub_popu in data_name:
                    popu_label = popu_label_refined[(popu_label_df['Population_broad'] == sub_popu).values]
            if data_name == '1000G_African':
                # remove outliers in African population
                Af_outlier = np.load('data/1000G/subset_index.npy')
                popu_label = popu_label[Af_outlier]
        elif data_name == 'Hapmap3':
            popu_label_df = np.load('%s/%s/hapmap3_popu.npy' % (data_dir, data_name), allow_pickle=True)
            popu_label = popu_label_df[:, -1]
        elif data_name == 'UKBB':
            popu_label_df = pd.read_csv('%s/%s/sample_5k_lookup.csv' % (data_dir, data_name), sep=',')
            popu_label_df.set_index('f_eid', inplace=True)
            fam_match = pd.read_csv('%s/%s/ukbb3.100.fam' % (data_dir, data_name), sep=' ',
                                    header=None)
            popu_label_df = popu_label_df.loc[fam_match[[0]].values.flatten()]
            popu_label = popu_label_df[['board_label']].values.flatten()
        elif data_name == 'PBMC':
            popu_label = np.load('%s/%s/pbmc_celltype_clean.npy' % (data_dir, data_name), allow_pickle=True)
        elif data_name == 'GTEx':
            popu_label = np.load('%s/%s/gtex_tissue_labels.npy' % (data_dir, data_name), allow_pickle=True)
    else:
        print('There are only 4 datasets supported: 1000G, UKBB, PBMC, GTEx')
    return popu_label

def vis_2dim_subspace(u, errors, data_name, method_name, xRange, yRange,
                      data_dir = 'data', to_save=True, plot_error=True, legend_loc='lower right',
                      plot_legend=True,joint_error=0, **kwargs):
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['xtick.labelsize'] = 23
    plt.rcParams['ytick.labelsize'] = 23
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), constrained_layout=True)

    # get PC name
    PC1 = kwargs.get("PC1", 1)
    PC2 = kwargs.get("PC2", 2)
    print('Plot PC %i and PC %i ' % (PC1, PC2))
    # load sample label for coloring
    popu_label = load_sample_labels(data_name, data_dir)
    print(popu_label.shape)
    print(u.shape)
    df = pd.DataFrame(dict(x=list(u[:, 0]), y=list(u[:, 1]), label=popu_label))
    # make scatter plot
    groups = df.groupby('label')
    for name, group in groups:
        ax.scatter(group.x, group.y, marker='o', s=3, label=name) # linestyle='',
    # add title
    # ax.set_title('%s, %.2f' % (method_name, joint_error))
    ax.set_title('%s' % method_name)
    # add quantitative evaluation
    if method_name == 'Ground truth PCs' or data_name == 'PBMC':
        ax.set_xlabel('PC %s' % PC1)
        ax.set_ylabel('PC %s' % PC2)
    else:
        ax.set_xlabel('PC %i, error=%.2f' % (PC1, errors[0]))
        ax.set_ylabel('PC %i, error=%.2f' % (PC2, errors[1]))
    # set aesthetics
    # if (method_name == 'Ground truth PCs' and PC1 == 1) or (data_name == 'PBMC' and PC1 == 1):
        # https://stackoverflow.com/questions/24706125/setting-a-fixed-size-for-points-in-legend
    #     plt.legend(loc=legend_loc, fancybox=True, fontsize=18, markerscale=4, framealpha=0.5) # scatterpoints=1,

    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    if to_save:
        fig_prefix = kwargs.get('fig_prefix', 'figures/%s/' % data_name)
        plt.savefig(fig_prefix + '%s_%s_PC_%i_%i.png' % (method_name.replace(' ', '_'), data_name, PC1, PC2))
    return ax
