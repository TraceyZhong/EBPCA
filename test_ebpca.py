import numpy as np
from debug import hist_save

from ebpca.amp import ebamp_orthog
from ebpca.empbayes import NonparEB
from ebpca.empbayes import TestEB


if __name__ == "__main__":
    np.random.seed(8921)

    ntrials = 1
    niters = 3

    m = 2000
    n = 4000
    init_align = 0.5
    sigma = np.sqrt(1-init_align**2)

    sample_udenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_u_")
    sample_vdenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_v_")

    def test(W,alpha,prefix, udenoiser, vdenoiser):
        ustar = np.random.binomial(1,0.5,size=m)*2-1
        vstar = np.random.binomial(1,0.5,size=n)*2-1
        X = alpha/n * np.outer(ustar,vstar) + W
        hist_save(np.linalg.svd(X)[1], '%s_singvals.png' % prefix)
        u = ustar * init_align + sigma * np.random.normal(size=m)
        (U,V) = ebamp_orthog(X,u,init_align,iters=niters,udenoiser = udenoiser, vdenoiser = vdenoiser)
        Unormsq = np.diag(np.transpose(U).dot(U))
        Vnormsq = np.diag(np.transpose(V).dot(V))
        Ualigns = np.transpose(U).dot(ustar) / np.sqrt(Unormsq*m)
        Valigns = np.transpose(V).dot(vstar) / np.sqrt(Vnormsq*n)
        print(Ualigns)
        print(Valigns)

    # # Gaussian noise
    # alpha = 2
    # print('Gaussian noise, alpha = {}'.format(alpha))
    # for i in range(ntrials):
    #     prefix='figures/rect_gaussian_%d' % i
    #     W = np.random.normal(size=(m,n))/np.sqrt(n)
    #     udenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_gaus_u_")
    #     vdenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_gaus_v_")
    #     test(W,alpha,prefix,udenoiser,vdenoiser)

    alpha = float(2)
    print('Noise with all singular values +1, alpha = 2')
    for i in range(ntrials):
        prefix = "figures/uniform_%d" % i 
        W = np.random.normal(size=(m,m))
        O = np.linalg.qr(W)[0]
        W = np.random.normal(size=(n,n))
        Q = np.linalg.qr(W)[0]
        D = np.zeros((m,n))
        for i in range(min(m,n)): D[i,i] = 1
        W = O.dot(D).dot(Q)
        udenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_unif_u_")
        vdenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_unif_v_")
        test(W,alpha,prefix,udenoiser,vdenoiser)

    alpha = float(3)
    print('Noise with beta-distributed singular values, alpha = 3')
    for i in range(ntrials):
        prefix = "figures/beta_%d" % i 
        W = np.random.normal(size=(m,m))
        O = np.linalg.qr(W)[0]
        W = np.random.normal(size=(n,n))
        Q = np.linalg.qr(W)[0]
        D = np.zeros((m,n))
        d = np.random.beta(1,2,size=min(m,n))
        d *= np.sqrt(min(m,n))/np.linalg.norm(d)
        for i in range(min(m,n)): D[i,i] = d[i]
        W = np.dot(np.dot(O,D),Q)
        udenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_beta_u_")
        vdenoiser = NonparEB(em_iter = 1000, to_save = True, to_show = False, fig_prefix = "nopareb_beta_v_")
        test(W,alpha,prefix,udenoiser,vdenoiser)