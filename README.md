# EchemScripts
My collection of python scripts to simulate electrochemical problems. Some of these are (or will be) used in Soft Potato.

### Requirements
* Python 3
* Numpy
* Scipy
* Matplotlib

# List of scripts

## Assumptions
* Cyclic voltammetry (chronoamperometry can be simulated by using the appropriate function from waveforms.py)
* Butler-Volmer kinetics unless otherwise stated
* Oxidation: only R present in solution at t = 0.


### General scripts
* waveforms.py: functions to generate potential waveforms (potential sweep, potential step, current step)
* plots.py: plotting functions

### Electrochemistry simulations
* Backwards implicit method:
  * RandCirc-BI_banded.py: Solves the Randles circuit for an E mechanism with scipy.linalg.solve_banded()
