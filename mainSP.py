#!/usr/bin/python

from softpotato.waveform import Sweep, Step
from softpotato.simulation import FD, BI
import matplotlib.pyplot as plt

## Create waveform object
wf1 = Sweep(dE = 0.005)

# Simulate
sim_FD = FD(wf1)
sim_BI = BI(wf1)


print("\nSimulation finished without errors!\n")


plt.figure(1)
plt.plot(sim_FD.E, sim_FD.i*1e3, label="FD")
plt.plot(sim_BI.E, sim_BI.i*1e3, label="BE")
plt.xlabel("$E$ / V")
plt.ylabel("$i$ / mA")
plt.legend()
plt.grid()
plt.show()
