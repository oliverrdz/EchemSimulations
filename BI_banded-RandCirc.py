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


    Created on Wed Jul 22 10:07:18 2020
    * R - e- -> O
    * Diffusion with Randless circuit
    * Butler Volmer
    * Backwards implicit with banded matrix

    Simulation time: 2 s, with:
    * dE = 0.0001 (# time elements: 20K)
    * dX = 2e-3 (# distance elements: 3K)

    @author: oliverrdz
    https://oliverrdz.xyz
'''

import numpy as np
from scipy.linalg import solve_banded
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
Cb = 1e-6 # mol/cm3, bulk concentration of R
D = 1e-5 # cm2/s, diffusion coefficient of R
Ageo = 1 # cm2, geometrical area
r = np.sqrt(Ageo/np.pi) # cm, radius of electrode
ks = 1e-3 # cm/s, standard rate constant
alpha = 0.5 # transfer coefficient

Rf = 5 # Roughness factor
C = 20e-6 # F/cm2, specific capacitance
x = 0.1 # cm, Luggin electrode distance
kapa = 0.0632 # Ohm-1 cm-1, conductivity for 0.5 M NaCl
Cdl = Ageo*Rf*C # F, double layer capacitance
Ru = x/(kapa*Ageo) # Ohms, solution resistance

# Potential waveform
E0 = 0  # V, standard potential
Eini = -0.5 # V, initial potential
Efin = 0.5 # V, final potential vertex
sr = 1 # V/s, scan rate
ns = 2 # number of sweeps
dE = 0.0001 # V, potential increment. This value has to be small for BI to approximate the circuit properly

t, E = wf.sweep(Eini=Eini, Efin=Efin, dE=dE, sr=sr, ns=ns) # Creates waveform

#%% Simulation parameters
delta = np.sqrt(D*t[-1]) # cm, diffusion layer thickness
maxT = 1 # Time normalised by total time
dt = t[1] # t[1] - t[0]; t[0] = 0
dT = dt/t[-1] # normalised time increment
nT = np.size(t) # number of time elements

maxX = 6*np.sqrt(maxT) # Normalised maximum distance
dX = 2e-3 # normalised distance increment
nX = int(maxX/dX) # number of distance elements
X = np.linspace(0,maxX,nX) # normalised distance array

K0 = ks*delta/D # Normalised standard rate constant
lamb = dT/dX**2

# Thomas coefficients
a = -lamb
b = 1 + 2*lamb
g = -lamb

C = np.ones([nT,nX]) # Initial condition for C
V = np.zeros(nT+1)
iF = np.zeros(nT)

# Constructing ab to use in solve_banded:
ab = np.zeros([3,nX])
ab[0,2:] = g
ab[1,:] = b
ab[2,:-2] = a
ab[1,0] = 1
ab[1,-1] = 1

# Initial condition for V
V[0] = E[0]

#%% Simulation
for k in range(0,nT-1):
    eps = FRT*(V[k] - E0)
    
    # Butler-Volmer:
    b0 = -(1 +dX*K0*(np.exp((1-alpha)*eps) + np.exp(-alpha*eps)))
    g0 = 1
    
    # Updating ab with the new values
    ab[0,1] = g0
    ab[1,0] = b0
    
    #Boundary conditions:
    C[k,0] = -dX*K0*np.exp(-alpha*eps)
    C[k,-1] = 1
    
    C[k+1,:] = solve_banded((1,1), ab, C[k,:])
    
    # Obtaining faradaic current and solving voltage drop
    iF[k] = n*F*Ageo*D*Cb*(-C[k+1,2] + 4*C[k+1,1] - 3*C[k+1,0])/(2*dX*delta)
    V[k+1] = (V[k] + (t[1]/Cdl)*(E[k]/Ru -iF[k]))/(1 + t[1]/(Cdl*Ru))
    
i = (E-V[:-1])/Ru

# Denormalising:
c = C*Cb
x = X*delta
end = time.time()
print(end-start)

#%% Plot
p.plot(E, i, "$E$ / V", "$i$ / A")
p.plot(x, c[-1,:]*1e6, "x / cm", "c($t_{end}$,$x$=0) / mM")
