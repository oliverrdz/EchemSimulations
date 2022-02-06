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
    def __init__(self, DO=1e-5, DR=1e-5, cOb=0, cRb=1e-6, ks=1e8, alpha=0.5):
        self.DO = DO
        self.DR = DR
        self.DOR = DO/DR
        self.cOb = cOb
        self.cRb = cRb
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
    def __init__(self, species, tgrid):
        self.lamb = 0.45 # For the algorithm to be stable, lamb = dT/dX^2 < 0.5
        self.Xmax = 6*np.sqrt(tgrid.nT*self.lamb) # Infinite distance
        self.dX = np.sqrt(tgrid.dT/self.lamb) # distance increment
        self.nX = int(self.Xmax/self.dX) # number of distance elements
        self.X = np.linspace(0, self.Xmax, self.nX) # Discretisation of distance

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
            print(type(x))


class Simulate:
    '''
    '''
    def __init__(self, species, tgrid, xgrid):
        
        for k in range(1,nT):
            for x in species:
                if isinstance(x, E):
                    # Boundary condition, Butler-Volmer:
                    CR1kb = x.CR[k-1,1]
                    CO1kb = x.CO[k-1,1]
                    x.CR[k,0] = (CR1kb + xgrid.dX*x.Ke*np.exp(-x.alpha*x.eps[k])*(CO1kb + 
                                 CR1kb/x.DOR))/(1 + x.dX*x.Ke*(np.exp((1-x.alpha)*x.eps[k]+
                                 np.exp(-x.alpha*x.eps[k])/x.DOR))
                    x.CO[k,0] = CO1kb + (CR1kb - x.CR[k,0])/x.DOR
                    # Runge-Kutta 4:
                    x.CR[k,1:-1] = RK4(CR[k-1,:], 'E')[1:-1]
                    x.CO[k,1:-1] = RK4(CO[k-1,:], 'E')[1:-1]

        # Denormalising:
        if cRb:
            I = -CR[:,2] + 4*CR[:,1] - 3*CR[:,0]
            D = DR
            c = cRb
        else: # In case only O present in solution
            I = CO[:,2] - 4*CO[:,1] + 3*CO[:,0]
            D = DO
            c = cOb
        i = n*F*Ageo*D*c*I/(2*dX*delta)

        cR = CR*cRb
        cO = CO*cOb
        x = X*delta

    def fun(y, mech='E'):
        if mech == 'E':
            return np.dot(A,y) 
        else:
            return np.dot(A,y) - dT*Kc*y

    def RK4(y, mech):
        k1 = fun(y, mech)
        k2 = fun(y+dT*k1/2, mech)
        k3 = fun(y+dT*k2/2, mech)
        k4 = fun(y+dT*k3, mech)
        return y + (dT/6)*(k1 + 2*k2 + 2*k3 + k4)

           



if __name__ == '__main__':
    import waveforms as wf
    t, _ = wf.sweep()
    e = E()
    c = C()
    tgrid = TGrid(t)
    xgrid = XGrid([e,c], tgrid)
    #sim = Simulate([e,c])
