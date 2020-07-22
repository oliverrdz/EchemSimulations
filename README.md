# EchemScripts
My collection of python scripts to simulate electrochemical problems. Some of these are (or will be) used in Soft Potato.

### Assumptions
* Cyclic voltammetry (chronoamperometry can be simulated by using the appropriate function from waveforms.py)
* Butler-Volmer kinetics unless otherwise stated
* Oxidation: only R present in solution at t = 0.

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
### General scripts
* waveforms.py: functions to generate potential waveforms (potential sweep, potential step, current step)
* plots.py: plotting functions

### Electrochemistry simulations
* Backwards implicit method:
  * RandCirc-BI_banded.py: Solves the Randles circuit for an E mechanism with scipy.linalg.solve_banded()
