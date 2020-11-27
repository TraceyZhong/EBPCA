import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from simulation.helpers import get_joint_alignment

def load_sample_labels(data_name, data_dir = 'data'):
    if data_name in ['1000G', 'UKBB', 'pbmc', 'GTEx']:
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
        elif data_name == 'pbmc':
            popu_label = np.load('%s/%s/pbmc_celltype_clean.npy' % (data_dir, data_name), allow_pickle=True)
        elif data_name == 'GTEx':
            popu_label = np.load('%s/%s/gtex_tissue_labels.npy' % (data_dir, data_name), allow_pickle=True)
    else:
        print('There are only 4 datasets supported: 1000G, UKBB, pbmc, GTEx')
    return popu_label

def vis_2dim_subspace(ax, u, plot_aligns, data_name, method_name,
                      data_dir = 'data/', to_save=True, **kwargs):
    popu_label = load_sample_labels(data_name, data_dir)
    df = pd.DataFrame(dict(x=list(u[:, 0]), y=list(u[:, 1]), label=popu_label))
    groups = df.groupby('label')
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=1, label=name) # plt.scatter(u1, u2, s=1.5)
    ax.legend(loc='lower right')
    ax.set_title('%s \n bivariate alignment=%.2f' % (method_name.replace('_', ' '), get_joint_alignment(plot_aligns)))
    ax.set_xlabel('PC 1, alignment={:.2f}'.format(plot_aligns[0]))
    ax.set_ylabel('PC 2, alignment={:.2f}'.format(plot_aligns[1]))
    if to_save:
        fig_prefix = kwargs.get('fig_prefix', 'figures/%s/' % data_name)
        plt.savefig(fig_prefix + '%s_%s.png' % (method_name, data_name))
        plt.close()
    else:
        plt.show()
    return ax