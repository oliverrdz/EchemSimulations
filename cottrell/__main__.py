from .solver import PlanarDiffusionSolver
from time import time
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    geoms = ['planar','spherical','thin_layer','rde','cylindrical','microband','rce']
    results={}
    for g in geoms:
        s=PlanarDiffusionSolver(1e-5,1.0,1.0,method='explicit',geometry=g)
        t0=time(); s.solve(); t1=time()
        results[g]=(s.t_s,s.i_t,t1-t0)
        print(g,'time',t1-t0)

    plt.figure()
    for g,(t,i,_) in results.items():
        plt.plot(t,i,label=g)
    plt.legend(); plt.xlabel('t'); plt.ylabel('i'); plt.show()
