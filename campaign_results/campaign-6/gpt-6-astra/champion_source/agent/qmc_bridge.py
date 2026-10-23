"""Linear Brownian bridge on an arbitrary deterministic variance clock."""
from collections import deque
import numpy as np

def bridge_normals(normal, sig):
    normal=np.asarray(normal,dtype=float)
    sig=np.asarray(sig,dtype=float)
    steps=len(sig)
    if normal.shape[1]!=steps or np.any(sig<=0):raise ValueError('Bridge requires positive per-step scales and matching normals')
    clock=np.r_[0.,np.cumsum(sig*sig)]
    path=np.zeros((normal.shape[0],steps+1))
    path[:,-1]=np.sqrt(clock[-1])*normal[:,0]
    pending=deque([(0,steps)]);coordinate=1
    while pending:
        left,right=pending.popleft()
        if right-left<=1:continue
        middle=(left+right)//2
        before=clock[middle]-clock[left];after=clock[right]-clock[middle];total=before+after
        path[:,middle]=(after*path[:,left]+before*path[:,right])/total+np.sqrt(before*after/total)*normal[:,coordinate]
        coordinate+=1
        pending.extend([(left,middle),(middle,right)])
    assert coordinate==steps
    return np.diff(path,axis=1)/sig
