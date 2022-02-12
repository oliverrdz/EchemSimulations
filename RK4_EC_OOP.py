import numpy as np
from scipy.sparse import diags

## Electrochemistry constants
F = 96485 # C/mol, Faraday constant    
R = 8.315 # J/mol K, Gas constant
T = 298 # K, Temperature
FRT = F/(R*T)

class E:
    '''
        Defines an E species
    '''
    def __init__(self, n=1, DO=1e-5, DR=1e-5, cOb=0, cRb=1e-6, E0=0, ks=1e8, 
                 alpha=0.5):
        self.n = n
        self.DO = DO
        self.DR = DR
        self.DOR = DO/DR
        self.cOb = cOb
        self.cRb = cRb
        self.E0 = E0
        self.ks = ks
        self.alpha = alpha


class C:
    '''
        Defines a C species  
    '''
    def __init__(self, DP=1e-5, cPb=0, kc=1e-2):
        self.DP = DP
        self.cPb = cPb
        self.kc = kc


class TGrid:
    '''
        Defines the grid in time
    '''
    def __init__(self, twf, Ewf):
        self.t = twf
        self.E = Ewf
        self.nT = np.size(self.t) # number of time elements
        self.dT = 1/self.nT # adimensional step time

        
class XGrid:
    '''
        Defines the grid in space
    '''
    def __init__(self, species, tgrid, Ageo=1):
        self.lamb = 0.45 # For the algorithm to be stable, lamb = dT/dX^2 < 0.5
        self.Xmax = 6*np.sqrt(tgrid.nT*self.lamb) # Infinite distance
        self.dX = np.sqrt(tgrid.dT/self.lamb) # distance increment
        self.nX = int(self.Xmax/self.dX) # number of distance elements
        self.X = np.linspace(0, self.Xmax, self.nX) # Discretisation of distance
        self.Ageo = Ageo

        for x in species:
            if isinstance(x, E):
                ## Discretisation of variables and initialisation
                if x.cRb == 0: # In case only O present in solution
                    x.CR = np.zeros([tgrid.nT, self.nX])
                    x.CO = np.ones([tgrid.nT, self.nX])
                else:
                    x.CR = np.ones([tgrid.nT, self.nX])

                x.eps = (tgrid.E-x.E0)*x.n*FRT # adimensional potential waveform
                x.delta = np.sqrt(x.DR*tgrid.t[-1]) # cm, diffusion layer thickness
                x.Ke = x.ks*x.delta/x.DR # Normalised standard rate constant
                x.CO = np.ones([tgrid.nT, self.nX])*x.cOb/x.cRb
                # Construct matrix:
                Cb = np.ones(self.nX-1) # Cbefore
                Cp = -2*np.ones(self.nX) # Cpresent
                Ca = np.ones(self.nX-1) # Cafter
                x.A = diags([Cb,Cp,Ca], [-1,0,1]).toarray()/(self.dX**2) 
                x.A[0,:] = np.zeros(self.nX)
                x.A[0,0] = 1 # Initial condition
            else:
                x.CP = np.zeros([tgrid.nT, self.nX])


class Simulate:
    '''
    '''
    def __init__(self, species, tgrid, xgrid):
        self.species = species
        self.tgrid = tgrid
        self.xgrid = xgrid

        self.join(species)

    def sim(self):
        for k in range(1, tgrid.nT):
            for s in self.species:
                if isinstance(s, E):
                    # Boundary condition, Butler-Volmer:
                    CR1kb = s.CR[k-1,1]
                    CO1kb = s.CO[k-1,1]
                    s.CR[k,0] = (CR1kb + xgrid.dX*s.Ke*np.exp(-s.alpha*s.eps[k]
                                 )*(CO1kb + CR1kb/s.DOR))/(1 + xgrid.dX*s.Ke*(
                                 np.exp((1-s.alpha)*s.eps[k])+np.exp(
                                 -s.alpha*s.eps[k])/s.DOR))
                    s.CO[k,0] = CO1kb + (CR1kb - s.CR[k,0])/s.DOR
                    # Runge-Kutta 4:
                    s.CR[k,1:-1] = self.RK4(s.CR[k-1,:], 'E', s)[1:-1]
                    s.CO[k,1:-1] = self.RK4(s.CO[k-1,:], 'E', s)[1:-1]
        for s in self.species:
            if isinstance(s, E):
                # Denormalising:
                if s.cRb:
                    I = -s.CR[:,2] + 4*s.CR[:,1] - 3*s.CR[:,0]
                    D = s.DR
                    c = s.cRb
                else: # In case only O present in solution
                    I = s.CO[:,2] - 4*s.CO[:,1] + 3*s.CO[:,0]
                    D = s.DO
                    c = s.cOb
                self.i = s.n*F*self.xgrid.Ageo*D*c*I/(2*self.xgrid.dX*s.delta)

                s.cR = s.CR*s.cRb
                s.cO = s.CO*s.cOb
                self.x = self.xgrid.X*s.delta

    def join(self, species):
        for s in species:
            print(s)


    def fun(self, y, mech, species):
        if mech == 'E':
            return np.dot(species.A,y) 
        elif mech == 'EC':
            return np.dot(species.A,y) - self.tgrid.dT*species.Kc*y

    def RK4(self, y, mech, A):
        dT = self.tgrid.dT
        k1 = self.fun(y, mech, A)
        k2 = self.fun(y+dT*k1/2, mech, A)
        k3 = self.fun(y+dT*k2/2, mech, A)
        k4 = self.fun(y+dT*k3, mech, A)
        return y + (dT/6)*(k1 + 2*k2 + 2*k3 + k4)

           



if __name__ == '__main__':
    import waveforms as wf
    import plots as p
    twf, Ewf = wf.sweep()
    e = E(ks=1e-3)
    c = C()
    tgrid = TGrid(twf, Ewf)
    xgrid = XGrid([e,c], tgrid)
    sim = Simulate([e,c], tgrid, xgrid)
    sim.sim()
    p.plot(Ewf, sim.i, xlab='$E$ / V', ylab='$i$ / A')
