import numpy as np
import plots as p
import waveforms as wf

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

# Potential waveform
E0 = 0  # V, standard potential
Eini = -0.5 # V, initial potential
Efin = 0.5 # V, final potential vertex
sr = 1 # V/s, scan rate
ns = 2 # number of sweeps
dE = 0.005 # V, potential increment. This value has to be small for BI to approximate the circuit properly

t, E = wf.sweep(Eini=Eini, Efin=Efin, dE=dE, sr=sr, ns=ns)

DOR = DO/DR

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


X = np.linspace(0,Xmax,nX) # Discretisation of distance
eps = (E-E0)*n*FRT # adimensional potential waveform
delta = np.sqrt(DR*t[-1]) # cm, diffusion layer thickness
K0 = ks*delta/DR # Normalised standard rate constant

#%% Simulation
for k in range(1,nT):
    # Boundary condition, Butler-Volmer:
    CR1kb = CR[k-1,1]
    CO1kb = CO[k-1,1]
    CR[k,0] = (CR1kb + dX*K0*np.exp(-alpha*eps[k])*(CO1kb + CR1kb/DOR))/(
               1 + dX*K0*(np.exp((1-alpha)*eps[k]) + np.exp(-alpha*eps[k])/DOR))
    CO[k,0] = CO1kb + (CR1kb - CR[k,0])/DOR

    # Solving finite differences:
    for j in range(1,nX-1):
        CR[k,j] = CR[k-1,j] + lamb*(CR[k-1,j+1] - 2*CR[k-1,j] + CR[k-1,j-1])
        CO[k,j] = CO[k-1,j] + DOR*lamb*(CO[k-1,j+1] - 2*CO[k-1,j] + CO[k-1,j-1])

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

#%% Plot
p.plot(E, i, "$E$ / V", "$i$ / A")
