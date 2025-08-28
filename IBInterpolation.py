import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv
import Potential2



def save_arrays_to_csv(file_path, column_titles, *arrays):
    # Transpose the arrays to align them by columns
    transposed_arrays = np.array(list(zip(*arrays)))
    

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        #Writes the column titles.
        writer.writerow(column_titles)
        
        #Writes in data rows.
        for row in transposed_arrays:
            writer.writerow(row)
            

def IB(R2):
    if not isinstance(R2, (int, float)):
        raise ValueError("R2 must be a numeric value.")
    
    integrand = lambda x: (x**2 / np.sqrt(x**2 + R2)) * (1 / (np.exp(np.sqrt(x**2 + R2)) - 1))
    result, error = quad(integrand, 0, np.inf, limit=100, epsabs=1e-10, epsrel=1e-10)
    return result

#Create the interpolator for r^2 = M^2 /T^2 ~ Range[(700/100)^2, (800/200)^2] 
R2_vals = np.concatenate(([0],np.geomspace(0.001, 5000, 2000)))
print(R2_vals)
IB_vals = np.array([IB(R2) for R2 in R2_vals])


IB_interp = interp1d(R2_vals, IB_vals, kind='cubic', fill_value="extrapolate")

R2_test =80
print(f"IB({R2_test}) ≈ {IB_interp(R2_test)} (interpolated)")
print(f"IB({R2_test}) = {IB(R2_test)} (direct)")

# Interpolation vs solution
plt.plot(R2_vals, IB_vals, label='Original IB(R2)')
plt.plot(R2_vals, IB_interp(R2_vals), '--', label='Interpolated Ib')
plt.plot(R2_vals, Potential2.Jb_spline(R2_vals),'-.',label='Jb spline')
plt.xlabel('R2')
plt.ylabel('IB(R2)')
plt.legend()
plt.grid(True)
plt.show()

save_arrays_to_csv('IBData.csv',['IB','R2'],R2_vals, IB_vals)