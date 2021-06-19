import numpy as np 
import matplotlib.pyplot as plt

def write_to_record(signal_strength, algo, noise_type, aligns):
    fpath = "records/" + algo + noise_type + str(signal_strength) + ".csv"
    aligns =  ",".join(map(str, aligns))
    with open(fpath, 'a') as the_file:
        the_file.write(aligns + '\n')


def pdf_mu_sigmasq(x, mu,sigmasq):
    return 1/(2*np.sqrt(2*np.pi*sigmasq)) * np.exp(-(x-mu)**2/(2*sigmasq)) + 1/(2*np.sqrt(2*np.pi*sigmasq)) * np.exp(-(x+mu)**2/(2*sigmasq)) 

def plot_save(f, mu, sigmasq, t, figtitle = None):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7*1, 5))
    ax.hist(f, bins = 50, density = True)
    start = np.quantile(f, 0.05)*1.5
    end = np.quantile(f,0.95)*1.5
    xgrid = np.linspace(start, end, 50)
    ax.plot(xgrid, pdf_mu_sigmasq(xgrid, mu, sigmasq), color = 'orange')
    ax.vlines(x = mu, ymin = 0, ymax = pdf_mu_sigmasq(mu, mu, sigmasq), color = 'orange')
    ax.vlines(x = -mu, ymin = 0, ymax = pdf_mu_sigmasq(-mu, mu, sigmasq), color = 'orange')     
    ax.set_title("{} at step {} mu {:.4f} sigmasq {:.4f}".format(figtitle, t, mu, sigmasq))
    fig.savefig("figures/{}_at_iter_{}.png".format(figtitle,t))
    plt.close()

## --- Experiment --- ##


def get_first_k_mul(vec,k, start = 0):
    mul = 1
    for i in range(start, k):
        mul *= vec[i]
    return mul 

def getRademacherW(n):
    W = np.random.normal(size=(n,n))
    O = np.linalg.qr(W)[0]
    D = np.diag(np.random.binomial(1,0.5,size=n)*2-1)
    W = np.dot(np.dot(O,D),np.transpose(O))
    return W 


def getCenteredBetaW(n):
    W = np.random.normal(size=(n,n))
    O = np.linalg.qr(W)[0]
    d = np.random.beta(1,2,size=n)
    d -= np.mean(d)
    d *= np.sqrt(n)/np.linalg.norm(d)
    D = np.diag(d)
    W = np.dot(np.dot(O,D),np.transpose(O))
    return W 

def getGaussianW(n):
    W = np.random.normal(size=(n,n))
    W = (W+np.transpose(W))/np.sqrt(2*n)
    return W

def get_noise(n, noise_type):
    if noise_type == "Rademacher":
        return getRademacherW(n)
    if noise_type == "CenteredBeta":
        return getCenteredBetaW(n)
    if noise_type == "Gaussian":
        return getGaussianW(n)


def supervise(func):
    def supervised(*args, **kwargs):
        result = func(*args, **kwargs)
        name = func.__name__
        print(name)
        print(result)
        return result
    return supervised

def get_alignment_of_each_dim(U,V,rank = None):
    '''some random distance measurement
    U,V: ndarray (n,rank)
    '''
    if len(U.shape) == 1:
        U = U.reshape((-1,1))
    if len(V.shape) == 1:
        V = V.reshape((-1,1))
    
    if rank is None:
        assert U.shape == V.shape, "When rank is not specified, two subspaces \
            should have the same dimension"
    else:
        U = U[:, :rank]
        V = V[:, :rank]

    U = U/np.sqrt((U**2).sum(axis = 0))
    V = V/np.sqrt((V**2).sum(axis = 0))
    COR = np.abs(np.transpose(U) @ V)
    return np.diag(COR)


def get_dist_of_subspaces(U,V,rank = None):
    '''some random distance measurement
    U,V: ndarray (n,rank)
    '''
    if len(U.shape) == 1:
        U = U.reshape((-1,1))
    if len(V.shape) == 1:
        V = V.reshape((-1,1))
    
    if rank is None:
        assert U.shape == V.shape, "When rank is not specified, two subspaces \
            should have the same dimension"
    else:
        U = U[:, :rank]
        V = V[:, :rank]

    Qu, _ = np.linalg.qr(U, mode = "reduced")
    Qv, _ = np.linalg.qr(V, mode = "reduced")
    _, s, _ = np.linalg.svd(Qu.T.dot(Qv))
    return np.sqrt(1 - np.min(s)**2)
