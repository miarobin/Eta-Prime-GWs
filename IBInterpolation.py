import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv


# Numerical derivative using central difference
#another comment

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
    eps = 1e-6  # small offset to avoid singularities
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
    eps = 1e-4  # small offset to avoid singularities
    if not isinstance(R2, (int, float)):
        raise ValueError("R2 must be a numeric value.")
    
    if R2>0:
        integrand = lambda x: (-1/2) * (x**2 / (x**2 + R2)) * (1 / (np.exp(np.sqrt(x**2 + R2)) - 1)) * (np.exp(np.sqrt(x**2 + R2))/(np.exp(np.sqrt(x**2 + R2)) - 1) + 1/np.sqrt(x**2 + R2))
        #AGREES WITH MATHEMATICA NOW
  
        result, error = quad(integrand, 0.000000001, 30)
        #print(f'result={result}')
        return result
    
    if R2<0:
        integrand1 = lambda x: (-1/2) * (x**2 / (x**2 + R2)) * (1 / (np.exp(np.sqrt(x**2 + R2)) - 1)) * (np.exp(np.sqrt(x**2 + R2))/(np.exp(np.sqrt(x**2 + R2)) - 1) + 1/np.sqrt(x**2 + R2))
        integrand2 = lambda x: (-1/4) * (x**2 / (x**2 - np.abs(R2))) * (1 / (1 - np.cos(np.sqrt(np.abs(R2) - x**2)))) * (-1 + 1/np.sqrt(np.abs(R2) - x**2) * np.sin(np.sqrt(np.abs(R2) - x**2)))

        a = np.abs(R2)
        #print(f'R2={R2}')
        #print(f'int={integrand2(0.0001)}')
        #plt.plot(np.linspace(0.001,np.sqrt(a),num=200), integrand2(np.linspace(0.001,np.sqrt(a),num=200)))
        #plt.show()
        

        result = quad(integrand1, np.sqrt(a)+eps, 30)[0]*0 + quad(integrand2, 0, np.sqrt(a)-eps)[0]+0#LOOK AT THIS LATER IT'S A LITTLE DODGY
        if result is not None:
            return result
        else:
            return 0
    
    if R2==0:
        return 10**(2)

if __name__ == "__main__":

    #Create the interpolator for r^2 = M^2 /T^2 ~ Range[(700/100)^2, (800/200)^2] 
    R2_vals = np.concatenate((np.linspace(-10,-0.001),np.concatenate(([0],np.geomspace(0.001, 100, 2000)))))



    IB_vals = np.array([IB(R2) for R2 in R2_vals])
    dIB_vals = np.array([dIB(R2) for R2 in R2_vals])
    dIB_numerical = np.array([numerical_derivative(IB, R2) for R2 in R2_vals])

    

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
    #plt.plot(R2_vals, Potential2.Jb_spline(R2_vals),'-.',label='Jb spline')
    plt.xlabel('R2')
    plt.ylabel('IB(R2)')
    plt.legend()
    plt.grid(True)
    plt.show()

    save_arrays_to_csv('IBData.csv',['R2','IB'], R2_vals, IB_vals)
    save_arrays_to_csv('dIBData.csv',['R2','dIB'], R2_vals, dIB_vals)