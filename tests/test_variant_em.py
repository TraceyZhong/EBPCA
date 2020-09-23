import numpy as np
import matplotlib.pyplot as plt
import config
from ebpca.empbayes import NonparEBHDTest


def test(zTruthVar, covInvMispec, em_iter):
    n = 1000
    nsample = 1000
    rank = 2

    figPrefix = "VEM_%d_%d_%d" % (em_iter, zTruthVar, covInvMispec)

    ustar1 = np.repeat(np.array([1,-1]), int(n/2))
    ustar2 = np.repeat(np.array([1,-1,0,0]), int(n/4))
    zTruth = np.vstack((ustar1, ustar2)).T + np.random.normal(scale= zTruthVar/10, size = nsample*rank).reshape((nsample, rank))
    zTruth = zTruth / np.sqrt(np.diag(zTruth.T @ zTruth)) * np.sqrt(nsample)

    muTruth = np.array([[2,0], [0,1.5]])
    covTruth = np.array([[1, 0.1],[0.1, 1]])

    # muTruth = np.array([[2,0], [0,2]])
    # covTruth = np.array([[1,0],[0,1]])

    f = zTruth.dot(muTruth.T) + np.random.multivariate_normal([0,0], covTruth, size=nsample)

    muPerceived = muTruth
    covPercevied = covTruth + covInvMispec


    denoiser = NonparEBHDTest(em_iter = em_iter ,nsupp_ratio=1, fig_prefix=figPrefix) # fig_prefix we need parameters
    # use the three methods
    for knowledge in [0,1,2,3]:
        denoiser.test_estimate_prior(f, muPerceived, covPercevied, knowledge, Z = zTruth, pi = np.full(len(zTruth), 1/ len(zTruth)))

    fig, axes = plt.subplots(ncols = 2, figsize = (2*6,5))
    axes[0].set(xlim=(-2, 2), ylim=(-2, 2))
    axes[0].scatter(zTruth[:,0], zTruth[:,1])
    axes[0].set_title("Ground Truth Z")
    # axes[1].set(xlim=(-5, 5), ylim=(-5, 5))
    axes[1].scatter(f[:,0], f[:,1])
    axes[1].set_title("noisy F")
    fig.savefig("./figures/" + figPrefix + 'Truth.png')



## --- get animation --- ##

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation 

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

def animateEachMethod(file_prefix, method):

    # get data
    ZFilePath = file_prefix + method + "Z.npy"
    Z = np.load(ZFilePath)
    PiFilePath = file_prefix + method + "Pi.npy"
    Pi = np.load(PiFilePath)
    DenoisedFilePath = file_prefix + method + "Denoised.npy"
    Denoised = np.load(DenoisedFilePath)

    nFrames = Denoised.shape[-1]

    fig, axes = plt.subplots(ncols = 2, figsize=(7, 3))
    axes[0].set(xlim=(-2, 2), ylim=(-2, 2))
    axes[0].set_title("Estimated Prior")
    axes[1].set(xlim=(-2, 2), ylim=(-2, 2))
    axes[1].set_title("Denoised")

    cmap = mpl.cm.Blues

    scatPi = axes[0].scatter(Z[:,0], Z[:,1], s=2, cmap = cmap, alpha = 0.5)
    scatDenoised = axes[1].scatter(Denoised[:,0,0], Denoised[:,1,0], alpha = 0.5)

    scats = [scatPi, scatDenoised]

    def animate(i):
        pi = Pi[:,i]
        scats[0].set_color(cmap(pi*1000))
        scats[1].set_offsets(Denoised[:,:,i])

        return scats

    ani = animation.FuncAnimation(fig, animate, frames=nFrames, interval = 200, blit=False)
    
    # writergif = animation.PillowWriter(fps=30)
    ani.save(file_prefix + method + ".mp4",writer='ffmpeg')
    # ani.save(file_prefix + method + ".gif") # ,writer='ffmpeg')

def animateAllMethods(zTruthVar, covInvMispec, em_iter):
    figPrefix = "./figures/VEM_%d_%d_%d" % (em_iter, zTruthVar, covInvMispec)
    for method in ["knownPrior", "ignorant", "knownPriorSup", "gridSpan"]:
        animateEachMethod(figPrefix, method)

# test(2,0,1000)
animateAllMethods(2,0,1000)




