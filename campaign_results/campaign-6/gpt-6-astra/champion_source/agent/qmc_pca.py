"""Exact Gaussian increment representation ordered by cumulative covariance variance."""
import numpy as np

def pca_normals(normal, sig):
    normal=np.asarray(normal,dtype=float)
    sig=np.asarray(sig,dtype=float)
    if normal.shape[1]!=len(sig) or np.any(sig<=0):
        raise ValueError('PCA requires matching positive per-step scales')
    clock=np.cumsum(sig*sig)
    covariance=np.minimum.outer(clock,clock)
    values,vectors=np.linalg.eigh(covariance)
    values=values[::-1];vectors=vectors[:,::-1]
    pivot=np.argmax(np.abs(vectors),axis=0)
    vectors*=np.where(vectors[pivot,np.arange(len(sig))]<0,-1.,1.)
    path=normal@(vectors*np.sqrt(np.maximum(values,0.))).T
    return np.diff(path,axis=1,prepend=np.zeros((len(normal),1)))/sig
