# EchemScripts
Collection of python scripts to simulate electrochemical problems, some of these are (or will be) used in [Soft Potato](https://oliverrdz.xyz/soft-potato). At the top of the scripts I added the total simulation time with specific parameters and with my setup (see below). In general, the simulation time goes: backwards implicit banded << backwards implicit << explicit finite differences. For learning purposes I recommend using the explicit finite differences scheme, while for more advanced users, the backwards implicit method banded should be used. In all cases, the current is approximated with three points, as opposed to the generally used 2-point approximation.

### Assumptions
* Cyclic voltammetry (chronoamperometry can be simulated by using the appropriate function from waveforms.py)
* Butler-Volmer kinetics unless otherwise stated
* Oxidation: only R present at t = 0
* Equal diffusion coefficients

### Requirements
* Python 3
* Numpy
* Scipy
* Matplotlib

### My setup
* Operating system: PopOS 20.04 LTS x86_64
* IDE: Spyder
* CPU: Intel i5-8265U (8) @ 3.900GHz
* Mem: 8 GB

# List of scripts
### General
* waveforms.py: functions to generate potential waveforms (potential sweep, potential step, current step)
* plots.py: functions for easy plotting

### Electrochemistry simulations
* Explicit finite differences:
  * FD-E.py: Simulates an E mechanism with finite differences with only R in solution
  * FD-E_OR.py: Simulates an E mechanism with finite differences with O and R in solution
  * FD-ECIrrev_ORY.py: Simulates an EC mechanism with finite differences with O and R in solution
* Runge-Kutta 4:
  * RK4-E.py: Simulates an E mechanism with Runge-Kutta 4 with only R in solution
  * RK4-EC.py: Simulates an EC mechanism with Runge-Kutta 4 with O and R in solution
* Backwards implicit method:
  * BI-ads.py: Surface bound species
  * BI-ads_RandCirc.py: Surface bound species with the Randles circuit
  * BI_banded-E.py: Simulates an E mechanism with scipy.linalg.solve_banded()
  * BI-E_RandCirc.py: Solves the Randles circuit for an E mechanism
  * BI_banded-E_RandCirc.py: Solves the Randles circuit for an E mechanism with scipy.linalg.solve_banded()
