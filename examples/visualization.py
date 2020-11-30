import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.extend(['../../generalAMP'])
from simulation.helpers import get_joint_alignment, get_error

def load_sample_labels(data_name, data_dir = 'data'):
    if data_name in ['1000G', 'UKBB', 'PBMC', 'GTEx']:
        if data_name == '1000G':
            popu_label_df = pd.read_csv('%s/%s/Popu_labels.txt' % (data_dir, data_name), sep=' ')
            popu_label = popu_label_df[['Population_broad']].values.flatten()
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

def vis_2dim_subspace(u, plot_aligns, data_name, method_name, xRange, yRange,
                      data_dir = 'data', to_save=True,
                      plot_error=True, legend_loc='lower right', **kwargs):
    # tune aesthetics
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['font.size'] = 10

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 6))

    # get PC name
    PC1 = kwargs.get("PC1", 1)
    PC2 = kwargs.get("PC2", 2)
    print('Plot PC %i and PC %i ' % (PC1, PC2))

    # load sample label for coloring
    popu_label = load_sample_labels(data_name, data_dir)
    df = pd.DataFrame(dict(x=list(u[:, 0]), y=list(u[:, 1]), label=popu_label))

    # metrics to plot
    metric = [get_joint_alignment(plot_aligns, iterates=False), plot_aligns[0], plot_aligns[1]]
    metric_name = 'alignment'
    if plot_error:
        metric = [get_error(align) for align in metric]
        metric_name = 'error'

    groups = df.groupby('label')
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=1, label=name)

    if method_name == 'ground_truth_PCA':
        ax.set_title('%s' % method_name.replace('_', ' '))
        ax.set_xlabel('PC %s' % PC1)
        ax.set_ylabel('PC %s' % PC2)
    else:
        ax.set_title('%s \n bivariate %s=%.2f' % \
                     (method_name.replace('_', ' '), metric_name, metric[0]))
        ax.set_xlabel('PC %i, %s=%.2f' % (PC1, metric_name, metric[1]))
        ax.set_ylabel('PC %i, %s=%.2f' % (PC2, metric_name, metric[2]))

    # set aesthetics
    ax.legend(loc=legend_loc)
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)

    if to_save:
        fig_prefix = kwargs.get('fig_prefix', 'figures/%s/' % data_name)
        print(fig_prefix + '%s_%s_PC_%i_%i.png' % (method_name, data_name, PC1, PC2))
        plt.savefig(fig_prefix + '%s_%s_PC_%i_%i.png' % (method_name, data_name, PC1, PC2))
    return ax