""" WHAM equation solver using BFGS algorithm as described by Zhu and Hummer:

    Zhu, F., & Hummer, G. (2012). Convergence and error estimation in free
    energy calculations using the weighted histogram analysis method. Journal
    of Computational Chemistry, 33(4), 453-465.
    http://doi.org/10.1002/jcc.21989

This code also closely follows the notation of Zhu and Hummer (2012).
"""

from scipy.optimize import minimize
import numpy as np
import pandas as pd
from glob import glob

# Temporary dependency to load test data from Gromacs
import gmx_wrapper
gmx = gmx_wrapper.Gromacs()

def A(dg, N, M, c):
    """Negative log likelihood, to be minimized"""
    d = np.sum(N*c*np.exp(np.concatenate(([0], np.cumsum(dg)))), axis=1)
    return -np.sum(N[1:]*np.cumsum(dg)) - np.sum(M*np.log(M/d))

def dAdg(dg, N, M, c):
    """Jacobian (gradient) of negative log likelihood"""
    d = np.sum(N*c*np.exp(np.concatenate(([0], np.cumsum(dg)))), axis=1)
    return np.cumsum((N[1:]*(np.exp(np.cumsum(dg))*np.sum(M*c[:, 1:].T/d, axis=1)-1))[::-1])[::-1]

def xvg_spring_iter(xvgs):
    """ Given the list of xvg files from the data set, read the spring
    constants and positions from the corresponding tpr files and yield back
    xvg, k, x0.
    """
    for xvg in xvgs:
        tpr = xvg.replace('pullx-', '').replace('.xvg', '.tpr')
        k = float(gmx.dump_key(tpr, 'k')[0])/2.49434
        x0 = float(gmx.dump_key(tpr, 'init')[0])
        yield xvg, k, x0

# load position data from xvg files
xvgs = glob('data/pullx*xvg')
dfs = []
for xvg in xvgs:
    # remove lines starting with # or @
    data = pd.read_csv(xvg, sep='\t', names=['t', 'x'], comment='#')
    try:
        data = data[~data['t'].str.contains('@')]
    except AttributeError:
        pass
    dfs.append(data)
df = pd.concat(dfs, keys=xvgs)

# add up N
N = []
k = []
x0 = []
f = []
# From each simulation, get number of points, spring constant, and spring pos
for xvg, k_, x0_ in xvg_spring_iter(set(df.index.get_level_values(0))):
    x = df.xs(xvg)['x']
    N.append(len(x))
    k.append(k_)
    x0.append(x0_)
    f.append(-np.mean(k_*(x-x0_)))
N = np.array([len(df.xs(xvg)) for xvg in set(df.index.get_level_values(0))])

# bin points -> M
bins = np.arange(1.35, 8.71, 0.01)
M, _ = np.histogram(df['x'], bins)

# compute c, assuming harmonic biasing potentials
bin_centers = (bins[1:]+bins[:-1])/2.
c = np.exp(-np.array(k)/2.*np.subtract.outer(bin_centers, x0)**2.)

dg = np.zeros(len(N) - 1)
result = minimize(A, dg, args=(N, M, c), method='BFGS', jac=dAdg)
print(result)
