# EchemScripts
Collection of python scripts to simulate electrochemical problems, some of these are (or will be) used in [Soft Potato](https://github.com/oliverrdz/SoftPotato). At the top of the scripts I added the total simulation time with specific parameters and with my setup (see below). In general, the simulation time goes backwards implicit banded << backwards implicit.

### Assumptions
* Cyclic voltammetry (chronoamperometry can be simulated by using the appropriate function from waveforms.py)
* Butler-Volmer kinetics unless otherwise stated
* Oxidation: only R present in solution at t = 0
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
  * FD_E.py: Simulates E mechanism with finite differences
* Backwards implicit method:
  * BI_RandCirc.py: Solves the Randles circuit for an E mechanism with BI 
  * BI_banded-RandCirc.py: Solves the Randles circuit for an E mechanism with scipy.linalg.solve_banded()
