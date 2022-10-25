import numpy as np
import matplotlib.pyplot as plt
import softpotato as sp


class Species:
    '''
    '''
    def __init__(self, cOb=0, cRb=1e-6, DO=1e-5, DR=1e-5, k0=1e8, alpha=0.5):
        self.cOb = cOb
        self.cRb = cRb
        self.DO = DO
        self.DR = DR
        self.k0 = k0
        self.alpha = alpha


class XGrid:
    '''
    '''
    def __init__(self, Ageo=1):
        self.Ageo = Ageo
        self.lamb = 0.45

    def define(self, tgrid, spc):
        self.Xmax = 6*np.sqrt(tgrid.nT*self.lamb)
        self.dX = np.sqrt(tgrid.dT/self.lamb)
        self.nX = int(self.Xmax/self.dX)
        self.X = np.linspace(0, self.Xmax, self.nX)


class TGrid:
    '''
    '''
    def __init__(self):
        pass

    def define(self, wf):
        self.t = wf.t
        self.E = wf.E
        self.nT = np.size(self.t)
        self.dT = 1/self.nT


class Simulate:
    '''
    '''
    def __init__(self, wf, xgrid, tgrid, spc):
        self.wf = wf
        self. xgrid = xgrid
        self.tgrid = tgrid
        self.spc = spc
        sim.i = 0
        self.tgrid.define(self.wf)
        self.xgrid.define(self.tgrid, self.spc) 


if __name__ == '__main__':
    print('Running from main')

    wf = sp.technique.Sweep()
    E_spc = Species.E()
    xgrid = XGrid()
    tgrid = TGrid()
    sim = Simulate(wf, xgrid, tgrid, E_spc)


    # Plotting
    sp.plotting.plot(wf.t, wf.E, xlab='$t$ / s', ylab='$E$ / V', fig=1, show=1)
