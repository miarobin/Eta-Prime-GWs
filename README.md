## Project: Computing Gravitational-Wave Signals

### What do these files do?

1. **Potential2.py**  
   Builds a `Potential` class for the “Normal” LSM mesonic potential for a QCD-like theory (see e.g. https://arxiv.org/abs/2511.23467).  
   
2. **GravitationalWave.py**  
   Computes the gravitational-wave signal for the potentials defined in (1).

3. **InterpolatorCheck.py**  
   Verifies that the gluonic potential is correctly interpolated.

4. **BadPointRemoval.py**  
   Removes anomalous points from the gluonic potential’s raw data before interpolation.

5. **DressedMasses.py**  
   Calculates the dressed masses using the CJT method (see https://doi.org/10.1103/PhysRevD.10.2428).

6. **Plotter.py**  
   Generates data and plots.

7. **zoom_inScan.py**  
   Scans over parameter space to find the most sensitive points and returns the corresponding parameter ranges.

8. **config_debug_plot.py**  
   Modifies `plt.show()` across the code to generate dynamic plots on the HiPerGator cluster.

### How to download and run

You’ll want a Python 3.x environment (e.g. a conda environment) with:

- `cosmotransitions` (`pip install cosmotransitions`)
- `scipy`
- `numpy`
- `matplotlib`
- (and any other standard dependencies you use)

Once these are installed, the scripts should run in principle with the provided configuration.
