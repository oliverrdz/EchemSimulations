import numpy as np
from scipy.sparse import diags
import plots as p
import waveforms as wf
import time

start = time.time()

## Electrochemistry constants
F = 96485 # C/mol, Faraday constant    
R = 8.315 # J/mol K, Gas constant
T = 298 # K, Temperature
FRT = F/(R*T)

n=1
Ageo=1
cOb=0
cRb=1e-6
DO=1e-5
DR=1e-5
ks=1e8
alpha=0.5

DP = 1e-5
kc = 1e1 # less tan 1e-1 to converge

# Potential waveform
E0 = 0  # V, standard potential
Eini = -0.5 # V, initial potential
Efin = 0.5 # V, final potential vertex
sr = 1 # V/s, scan rate
ns = 2 # number of sweeps
dE = 0.005 # V, potential increment. This value has to be small for BI to approximate the circuit properly

t, E = wf.sweep(Eini=Eini, Efin=Efin, dE=dE, sr=sr, ns=ns)

DOR = DO/DR
DPR = DP/DR

#%% Simulation parameters
nT = np.size(t) # number of time elements
dT = 1/nT # adimensional step time
lamb = 0.45 # For the algorithm to be stable, lamb = dT/dX^2 < 0.5
Xmax = 6*np.sqrt(nT*lamb) # Infinite distance
dX = np.sqrt(dT/lamb) # distance increment
nX = int(Xmax/dX) # number of distance elements

## Discretisation of variables and initialisation
if cRb == 0: # In case only O present in solution
    CR = np.zeros([nT,nX])
    CO = np.ones([nT,nX])
else:
    CR = np.ones([nT,nX])
    CO = np.ones([nT,nX])*cOb/cRb

CP = np.zeros([nT,nX])

X = np.linspace(0,Xmax,nX) # Discretisation of distance
eps = (E-E0)*n*FRT # adimensional potential waveform
delta = np.sqrt(DR*t[-1]) # cm, diffusion layer thickness
Ke = ks*delta/DR # Normalised standard rate constant
Kc = kc*delta/DR


Cb = np.ones(nX-1) # Cbefore
Cp = -2*np.ones(nX) # Cpresent
Ca = np.ones(nX-1) # Cafter
A = diags([Cb,Cp,Ca], [-1,0,1]).toarray()/(dX**2) # - dT*K*Cp
A[0,:] = np.zeros(nX)
A[0,0] = 1 # Initial condition
print(A)

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

#%% Simulation
for k in range(1,nT):
    # Boundary condition, Butler-Volmer:
    CR1kb = CR[k-1,1]
    CO1kb = CO[k-1,1]
    CR[k,0] = (CR1kb + dX*Ke*np.exp(-alpha*eps[k])*(CO1kb + CR1kb/DOR))/(
               1 + dX*Ke*(np.exp((1-alpha)*eps[k]) + np.exp(-alpha*eps[k])/DOR))
    CO[k,0] = CO1kb + (CR1kb - CR[k,0])/DOR
    # Runge-Kutta 4:
    CR[k,1:-1] = RK4(CR[k-1,:], 'E')[1:-1]
    CO[k,1:-1] = RK4(CO[k-1,:], 'EC')[1:-1]

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

end = time.time()
print(end-start)

#%% Plot
p.plot(E, i*1e3, "$E$ / V", "$i$ / mA")
