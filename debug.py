import numpy as np
import matplotlib.pyplot as plt

# Plotting functions, for debugging
def two_point_normal_pdf(x,mu,sigmasq):
  pdf_plus = 1/np.sqrt(2*np.pi*sigmasq) * np.exp(-(x-mu)**2/(2*sigmasq))
  pdf_minus = 1/np.sqrt(2*np.pi*sigmasq) * np.exp(-(x+mu)**2/(2*sigmasq))
  return 0.5*pdf_plus+0.5*pdf_minus

def plot_save(f, mu, sigmasq, filename, bins=50):
  fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,5))
  ax.hist(f, bins = bins, density = True)
  xgrid = np.linspace(min(f)*1.1,max(f)*1.1,1000)
  ygrid = two_point_normal_pdf(xgrid, mu, sigmasq)
  ax.plot(xgrid, ygrid, color = 'orange')
  ax.vlines(x = mu, ymin = 0, ymax = max(ygrid), color = 'orange')
  ax.vlines(x = -mu, ymin = 0, ymax = max(ygrid), color = 'orange')
  ax.set_title(filename)
  fig.savefig(filename)
  plt.close()

def hist_save(eigs, filename, ylim=(0,100), bins=50):
  fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,5))
  ax.hist(eigs, bins = bins)
  ax.set_ylim(ylim[0],ylim[1])
  ax.set_title(filename)
  fig.savefig(filename)
  plt.close()

