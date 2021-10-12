'''
    Copyright (C) 2020 Oliver Rodriguez
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Created on Wed Jul 22 15:07:06 2020
    * R - e- -> O
    * Diffusion, E mechanism
    * Butler Volmer
    * Runge Kutta 4

    @author: oliverrdz
    https://oliverrdz.xyz
'''

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

#%% Parameters

n = 1 # number of electrons
cB = 1e-6 # mol/cm3, bulk concentration of R
D = 1e-5 # cm2/s, diffusion coefficient of R
Ageo = 1 # cm2, geometrical area
r = np.sqrt(Ageo/np.pi) # cm, radius of electrode
ks = 1e-3 # cm/s, standard rate constant
alpha = 0.5 # transfer coefficient

# Potential waveform
E0 = 0  # V, standard potential
Eini = -0.5 # V, initial potential
Efin = 0.5 # V, final potential vertex
sr = 1 # V/s, scan rate
ns = 2 # number of sweeps
dE = 0.01 # V, potential increment. This value has to be small for BI to approximate the circuit properly

t, E = wf.sweep(Eini=Eini, Efin=Efin, dE=dE, sr=sr, ns=ns) # Creates waveform

#%% Simulation parameters
nT = np.size(t) # number of time elements
dT = 1/nT # adimensional step time
lamb = 0.45 # For the algorithm to be stable, lamb = dT/dX^2 < 0.5
Xmax = 6*np.sqrt(nT*lamb) # Infinite distance
dX = np.sqrt(dT/lamb) # distance increment
nX = int(Xmax/dX) # number of distance elements

## Discretisation of variables and initialisation
C = np.ones([nT,nX]) # Initial condition for R
X = np.linspace(0,Xmax,nX) # Discretisation of distance
eps = (E-E0)*n*FRT # adimensional potential waveform
delta = np.sqrt(D*t[-1]) # cm, diffusion layer thickness
K0 = ks*delta/D # Normalised standard rate constant

Cb = np.ones(nX-1) # Cbefore
Cp = -2*np.ones(nX) # Cpresent
Ca = np.ones(nX-1) # Cafter
A = diags([Cb,Cp,Ca], [-1,0,1]).toarray()/(dX**2)
A[0,:] = np.zeros(nX)
A[0,0] = 1 # Initial condition

def RK4(y):
    k1 = fun(y)
    k2 = fun(y+dT*k1/2)
    k3 = fun(y+dT*k2/2)
    k4 = fun(y+dT*k3)
    return y + (dT/6)*(k1 + 2*k2 + 2*k3 + k4)

def fun(y):
    return np.dot(A,y)

#%% Simulation
for k in range(1,nT):
    #print(nT-k)
    # Boundary condition, Butler-Volmer:
    C[k,0] = (C[k-1,1] + dX*K0*np.exp(-alpha*eps[k]))/(1+dX*K0*(np.exp((1-alpha)*eps[k])+np.exp(-alpha*eps[k])))
    C[k,1:-1] = RK4(C[k-1,:])[1:-1]#, A[:-1,:-1])
 
# Denormalising:
i = n*F*Ageo*D*cB*(-C[:,2] + 4*C[:,1] - 3*C[:,0])/(2*dX*delta)
cR = C*cB
cO = cB - cR
x = X*delta
end = time.time()
print(end-start)

#%% Plot
p.plot(E, i, "$E$ / V", "$i$ / A")
#p.plot2(x, cR[-1,:]*1e6, x, cO[-1,:]*1e6, "[R]", "[O]", "x / cm", "c($t_{end}$,$x$=0) / mM")
