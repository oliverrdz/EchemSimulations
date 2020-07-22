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

    Created on Wed Jul 22 16:44:49 2020
    * R - e- -> O
    * Surface bound species
    * Butler Volmer
    * Backwards implicit

    Simulation time: 0.12 s, with:
    * dE = 0.0001 (# time elements: 2K)

    @author: oliverrdz
    https://oliverrdz.xyz
'''

import numpy as np
from scipy.integrate import cumtrapz
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
Q0 = 210e-6 # C/cm2, charge density for one monolayer
D = 1e-5 # cm2/s, diffusion coefficient of R
Ageo = 1 # cm2, geometrical area
r = np.sqrt(Ageo/np.pi) # cm, radius of electrode
ks = 1e0 # cm/s, standard rate constant
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

g0 = Q0/F # mol/cm2, maximum coverage for 1 monolayer

# Simulation parameters
nt = np.size(t)
dt = t[1]

Th = np.ones(nt)
V = np.zeros(nt)
V[0] = E[0]

#%% Simulation
for j in range(1,nt):
    
    # iR drop makes the applied potential to be V, not E:
    eps = n*FRT*(V[j-1]-E0)
    kf = ks*np.exp(alpha*eps)
    kb = ks*np.exp(-(1-alpha)*eps)
    
    # Backwards implicit:
    Th[j] = (Th[j-1] + dt*kb)/(1 + dt*(kf+kb))
    V[j] = (V[j-1] + (dt/Cdl)*(E[j]/Ru +n*F*Ageo*g0*(kb-(kf+kb)*Th[j])))/(1 + dt/(Cdl*Ru))

# Denormalisation
i = (E - V)/Ru # The current is obtained from the Randles circuit
Q = cumtrapz(i, t, initial=0)
end = time.time()
print(end-start)

#%% Plot
p.plot(E, i, "$E$ / V", "$i$ / A")
p.plot(E, Q*1e6, "$E$ / V", "$Q$ / $\mu$C cm$^{-2}$")