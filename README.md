## A Project for Computing Gravitational Wave Signals for the Eta Prime.

# What do all of these files do?
1. Potential2.py: Builds a 'Potential' class for the 'Normal' LSM mesonic potential for a QCD-like theory, see e.g. https://arxiv.org/pdf/2309.16755. The 'Csaki' potential is from https://arxiv.org/abs/2307.04809.
2. GravitationalWave.py: Calculating the gravitational wave signal for the potential defined in (1).
3. InterpolatorCheck.py: Code to verify that the Gluonic potential is correctly being interpolated.
4. BadPointRemoval.py: Removes anomalous points from the Gluonic potential's 'raw' data, before interpolation.
5. Plotter.py: Data Generation.


# How to Download:
You'll want a python/conda virtual environment in python 3.X with cosmotransitions (pip install cosmotransitions), scipy, numpy, matplotlib, ...
Then in theory it should just run. :)

