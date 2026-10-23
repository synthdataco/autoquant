"""Stationary Gaussian return dependence with unchanged per-step variance."""
import numpy as np
from scipy.signal import lfilter

def correlated_normals(normal,rho):
    if not -1<rho<1:raise ValueError('Stationary correlation must be between -1 and1')
    if rho==0:return normal
    innovations=np.asarray(normal)*np.sqrt(1-rho*rho)
    innovations[:,0]=normal[:,0]
    return lfilter([1.],[1.,-rho],innovations,axis=1)
