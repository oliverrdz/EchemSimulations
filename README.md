# EchemScripts
Collection of python scripts to simulate electrochemical problems, some of these are (or will be) used in [Soft Potato](https://softpotato.xyz). The current is approximated with three points, as opposed to the generally used 2-point approximation.

### Assumptions
* Cyclic voltammetry (chronoamperometry can be simulated by using the appropriate function from waveforms.py)
* Butler-Volmer kinetics 
* Oxidation: only R present at t = 0 unless otherwise stated

### Requirements
* Python 3
* Numpy
* Scipy
* Matplotlib

### My setup
* Operating system: Manjaro
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
  * RK4-E.py: Simulates an E mechanism with Runge-Kutta 4 with only R in solution. Optimized with linear algebra.
  * RK4-E_OR.py: Simulates an E mechanism with Runge-Kutta 4 with O and R in solution. Optimized with linear algebra.
  * RK4-EC.py: Simulates an EC mechanism with Runge-Kutta 4 with O and R in solution.
* Backwards implicit method:
  * BI-ads.py: Surface bound species
  * BI-ads_RandCirc.py: Surface bound species with the Randles circuit
  * BI-E_RandCirc.py: Solves the Randles circuit for an E mechanism
* Solvers:
  * ODEsol-E.py: Simulates an E mechanism using scipy.integrate.solve_ivp. 
  * BI_banded-E.py: Simulates an E mechanism with scipy.linalg.solve_banded()
  * BI_banded-E_RandCirc.py: Solves the Randles circuit for an E mechanism with scipy.linalg.solve_banded()
