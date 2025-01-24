import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import interpolate



data = np.genfromtxt(f'GridDataF6N3Corrected.csv', delimiter=',', dtype=float, skip_header=1)
#Index 0 -> Temperature; index 1 -> Sigma; index 2 -> V

num = round(len(data)*.5)
linear_small = interpolate.SmoothBivariateSpline(data[:num,0],data[:num,1],data[:num,2]/1e7, kx=4,ky=3)
linear_large = interpolate.SmoothBivariateSpline(data[num:,0],data[num:,1],data[num:,2]/1e10, kx=4,ky=3)

Ts = range(15,1000,15)
sigmas = range(0,1000,15); step = 15

T_switch = data[num,0]

def _Vg(T, sig):
        # Check if input1 or input2 are single numbers (scalars)
        if np.ndim(T) == 0:
            T = [T]  # Treat as a vector of one element
        if np.ndim(sig) == 0:
            sig = [sig]  # Treat as a vector of one element

        # Convert inputs to numpy arrays for easier handling
        vector1 = np.array(T)
        vector2 = np.array(sig)
        
        # Create a matrix to store the results
        matrix = np.zeros((len(vector1), len(vector2)))
        
        # Loop through each element of vector1 and vector2, applying the function
        for i, a in enumerate(vector1):
            for j, b in enumerate(vector2):
                matrix[i, j] = _Vg_f(a, b)
    
        return np.array(matrix)
            
def _Vg_f(T, sig):
    if T<90:
        return 0
    if T<T_switch:
        if sig>1000:
            return linear_small.ev(T,1000)*1e7 
        else:
            return linear_small.ev(T,sig)*1e7
    else:
        if sig>1000:
            return linear_large.ev(T,1000)*1e10
        else:
            return linear_large.ev(T,sig)*1e10


def linear(T, sigma):
    if T<90:
        return 0
    if T<data[num,0]:
        return linear_small.ev(T,sigma)*1e7
    else:
        return linear_large.ev(T,sigma)*1e10




for T in Ts:
    temperaturepoint=[]
    for point in data:
        if round(T)==round(point[0]):
            temperaturepoint.append(point)
    print(T)
    print(np.array(temperaturepoint)[:,0])
    
    print(_Vg(T,sigmas))

    temperaturepoint=np.array(temperaturepoint)       
    plt.plot(temperaturepoint[:,1],temperaturepoint[:,2],label='real data')
    plt.plot(sigmas,(_Vg(T,sigmas)[0]),label='interpolated')
    plt.title(f'T={T}')
    plt.legend()
    plt.show()
        