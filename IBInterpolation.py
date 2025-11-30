import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv


'''
This file builds a numerical interpolator over the functions IB and dIB/dR^2 for use in DressedMasses.py.

Running as-is will build the files needed to run DressedMasses.py
'''


#Numerical derivative using central difference
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


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
    eps = 1e-3  # small offset to avoid singularities
    if not isinstance(R2, (int, float)):
        raise ValueError("R2 must be a numeric value.")
    
    if R2>0:
        integrand = lambda x: (x**2 / np.sqrt(x**2 + R2)) * (1 / (np.exp(np.sqrt(x**2 + R2)) - 1))
        result, error = quad(integrand, 0, np.inf, limit=100, epsabs=1e-10, epsrel=1e-10)
        return result
    if R2<0:
        integrand1 = lambda x: (x**2 / np.sqrt(x**2 + R2)) * (1 / (np.exp(np.sqrt(x**2 + R2)) - 1))
        integrand2 = lambda x: -np.sin(np.sqrt(np.abs(R2)-x**2))/(2-2*np.cos(np.sqrt(np.abs(R2)-x**2)))*(x**2/np.sqrt(np.abs(R2)-x**2))

        a = np.sqrt(np.abs(R2))
        result = quad(integrand1, a+eps, np.inf)[0] + quad(integrand2, 0, a-eps)[0]
        if result is not None:
            return result
        else:
            return 0

    if R2==0:
        #Avoiding numerical issues by taking limits properly.
        integrand = lambda x: x * (1 / (np.exp(x) - 1))
        result, error = quad(integrand, 0, np.inf, limit=100, epsabs=1e-10, epsrel=1e-10)
        return result
    
def dIB(R2):
    #Derivative of Ib wrt R2.
    eps = 1e-3 # small offset to avoid singularities
    if not isinstance(R2, (int, float)):
        raise ValueError("R2 must be a numeric value.")
    
    if R2>0:
        integrand = lambda x: (-1/2) * (x**2 / (x**2 + R2)) * (1 / (np.exp(np.sqrt(x**2 + R2)) - 1)) * (np.exp(np.sqrt(x**2 + R2))/(np.exp(np.sqrt(x**2 + R2)) - 1) + 1/np.sqrt(x**2 + R2))
  
        result, error = quad(integrand, 0.000000001, 30)
        return result
    
    if R2<0:
        a2 = np.abs(R2)
        integrand1 = lambda x: (-1/2) * (x**2 / (x**2 - a2)**(3/2)) * (1 / (np.exp(np.sqrt(x**2 - a2)) - 1))**2 * (np.exp(np.sqrt(x**2 - a2))*( 1 + np.sqrt(x**2 - a2)) - 1)
        integrand2 = lambda x: (1/4) * (x**2 / (x**2 - a2)) * (1 / (1 - np.cos(np.sqrt(a2 - x**2)))) * (1 + 1/np.sqrt(a2 - x**2) * np.sin(np.sqrt(a2 - x**2)))

        
        if np.sqrt(a2)>10*eps:
            resultA = quad(integrand1, np.sqrt(a2)+eps, 100)[0] 
            resultB = quad(integrand2, 0, np.sqrt(a2)-eps)[0] 
            result= resultA + resultB + 1/(2*eps) - np.sqrt(np.sqrt(a2)/(2*eps))/4
    
            if result is not None:
                return result
            else:
                return 0
        else:
            return 1/eps
    
    if R2==0:
        return 10**(2)

if __name__ == "__main__":
    #Set interpolator domain.
    R2_vals = np.concatenate((np.linspace(-10,-0.0001,num=1000),np.concatenate(([0],np.geomspace(0.001, 100, 2000)))))
    
    #Computing results for IB/dIB/numerical dIB across interpoloator domain.
    IB_vals = np.array([IB(R2) for R2 in R2_vals])
    dIB_vals = np.array([dIB(R2) for R2 in R2_vals])
    dIB_numerical = np.array([numerical_derivative(IB, R2) for R2 in R2_vals])

    
    #Testing the interpolator.
    IB_interp = interp1d(R2_vals, IB_vals, kind='cubic', fill_value="extrapolate")
    dIB_interp = interp1d(R2_vals, dIB_vals, kind='cubic', fill_value="extrapolate")
    
    R2_test =80
    print(f"IB({R2_test}) ≈ {IB_interp(R2_test)} (interpolated)")
    print(f"IB({R2_test}) = {IB(R2_test)} (direct)")

    # Interpolation vs solution
    plt.plot(R2_vals, IB_vals, label='Original IB(R2)')
    plt.plot(R2_vals, IB_interp(R2_vals), '--', label='Interpolated Ib')
    plt.plot(R2_vals, dIB_interp(R2_vals), '-.', label='Interpolated dIb')
    plt.plot(R2_vals, dIB_numerical, ':', label='Numerical Derivative')
    plt.scatter(R2_vals, dIB_vals)
    plt.xlabel('R2')
    plt.ylabel('IB(R2)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Temporal-Plots/Intervssol.pdf", dpi=300)
    plt.show()

    #Saves arrays as used by DressedMasses.py
    save_arrays_to_csv('IBData.csv',['R2','IB'], R2_vals, IB_vals)
    save_arrays_to_csv('dIBData.csv',['R2','dIB'], R2_vals, dIB_vals)