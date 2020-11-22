import numpy as np
import matplotlib.pyplot as plt
import scipy
'''
Before proceed, you should:
Step 1: Normalize data however you think is appropriate;
Step 2: Observe singular values and determine the number of spikes K;
Step 3: Look at the distribution of each pc. If there is no clear structure, i.e.
        the distribution is approximately Gaussian, there is little improvement
        can be down by EB-PCA.
Step 4: Rescale data to standarize the noise level, and then test if noise 
        matrix satisfy the assumption.
'''

def normalize_obs(Y, K = 0):
    '''Normalize the noise level of the observational matrix after regressing out
    top K pcs.
    '''
    if K == 0:
        raise(ValueError("# PC can not be zero."))
    n_features = Y.shape[1]
    U, Lambda, Vh = np.linalg.svd(Y, full_matrices = False)
    U = U[:,:K]
    Lambda = Lambda[:K]
    Vh = Vh[:K,:]
    R = Y - U * Lambda @ Vh
    tauSq = np.sum(R**2) / n_features
    print("estimated tau={}".format(np.sqrt(tauSq)))
    return Y / np.sqrt(tauSq)

def normalize_pc(U):
    return U/np.sqrt((U**2).sum(axis = 0)) * np.sqrt(len(U))

# TODO
# 1. add a function to do column normalization for real data
# 2. think of better names for these normalization functions

def plot_pc(samples,label="",nPCs=2,to_show=False, to_save=False, **kwargs):
    fig_prefix = kwargs.get("fig_prefix", "")
    u,s,vh = np.linalg.svd(samples,full_matrices=False)
    plt.figure(figsize = (10,3))
    plt.scatter(range(len(s)), s)
    plt.title('Singular values')
    if to_save:
        plt.savefig('figures/singvals_%s.png' % label)
    if to_show:
        plt.show()
    plt.close()
    for i in range(nPCs):
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows = 1, ncols = 4, figsize=(10,3))
        ax1.hist(u[:,i],bins=50)
        scipy.stats.probplot(u[:,i],plot=ax2)
        ax1.set_title('left PC %d' % (i+1))
        ax2.set_title('left PC %d' % (i+1))
        ax3.hist(vh[i,:],bins=50)
        scipy.stats.probplot(vh[i,:],plot=ax4)
        ax3.set_title('right PC %d' % (i+1))
        ax4.set_title('right PC %d' % (i+1))
        if to_save:
            fig.savefig('figures/%sPC_%d_%s.png' % (fig_prefix, i,label))
        if to_show:
            plt.show()
        plt.close()

def get_diff_L2(diff):
    return np.sqrt(np.mean(np.sum(diff**2, axis=0)))