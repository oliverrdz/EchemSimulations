#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:07:18 2020

R - e- -> O
Diffusion with Randless circuit
Butler Volmer
Backwards implicit with banded matrix

Simulation time: 1.8 s
dE = 0.0001
dX = 2e-3

@author: oliver
"""

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

n = 1
Cb = 1e-6 # mol/cm3
D = 1e-5 # cm2/s
Ageo = 1 # cm2
r = np.sqrt(Ageo/np.pi)
ks = 1e-3 # cm/s
alpha = 0.5

Rf = 5 # Roughness factor
C = 20e-6 # F/cm2, specific capacitance
x = 0.1 # cm, Luggin electrode distance
kapa = 0.0632 # Ohm-1 cm-1, conductivity for 0.5 M NaCl
Cdl = Ageo*Rf*C # F, double layer capacitance
Ru = x/(kapa*Ageo) # Ohms, solution resistance

# Potential waveform
E0 = 0 
Eini = -0.5
Efin = 0.5
sr = 1
ns = 2
dE = 0.0001 # This value has to be small for BI to approximate the circuit properly

t, E = wf.sweep(Eini=Eini, Efin=Efin, dE=dE, sr=sr, ns=ns)

#%% Simulation parameters
maxT = 1 # Time normalised by total time
dt = t[1] # t[1] - t[0]; t[0] = 0
dT = dt/t[-1]
nT = np.size(t)

maxX = 6*np.sqrt(maxT)
dX = 2e-3
nX = int(maxX/dX)
X = np.linspace(0,maxX,nX)
delta = np.sqrt(D*t[-1])

K0 = ks*delta/D
lamb = dT/dX**2

# Thomas coefficients
a = -lamb
b = 1 + 2*lamb
g = -lamb

g_mod = np.zeros(nX)
C = np.ones(nX) # Initial condition for C
V = np.zeros(nT+1)
jF = np.zeros(nT)

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
for k in range(0,nT):
    eps = FRT*(V[k] - E0)
    
    # Butler-Volmer:
    b0 = -(1 +dX*K0*(np.exp((1-alpha)*eps) + np.exp(-alpha*eps)))
    g0 = 1
    
    # Updating ab with the new values
    ab[0,1] = g0
    ab[1,0] = b0
    
    #Boundary conditions:
    C[0] = -dX*K0*np.exp(-alpha*eps)
    C[-1] = 1
    
    C = solve_banded((1,1), ab, C)
    
    # Obtaining faradaic current and solving voltage drop
    jF[k] = n*F*Ageo*D*Cb*(-C[2] + 4*C[1] - 3*C[0])/(2*dX*delta)
    V[k+1] = (V[k] + (t[1]/Cdl)*(E[k]/Ru -jF[k]))/(1 + t[1]/(Cdl*Ru))
    
i = (E-V[:-1])/Ru
end = time.time()
print(end-start)

#%% Plot
p.plot(E, i, "$E$ / V", "$i$ / A")