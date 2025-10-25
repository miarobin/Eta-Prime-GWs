import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy import interpolate
from debug_plot import debug_plot



GLUONIC_CUTOFF = 1000
data = np.genfromtxt(f'GridDataF6N3Corrected.csv', delimiter=',', dtype=float, skip_header=1)
#Index 0 -> Temperature; index 1 -> Sigma; index 2 -> V
 
# Split by temperature midpoint
T_mid = 0.5 * (min(data[:,0]) + max(data[:,0]))
low_mask = data[:,0] < T_mid
high_mask = data[:,0] >= T_mid

data_low = data[low_mask]
data_high = data[high_mask]

# Automatic normalization
scale_low = np.max(np.abs(data_low[:,2]))
scale_high = np.max(np.abs(data_high[:,2]))


#self.num = round(len(data)/2)
#self.T_switch = data[self.num,0]
#self.linear_small = interpolate.SmoothBivariateSpline(data[:self.num,0],data[:self.num,1],data[:self.num,2]/1e7, kx=4,ky=3)
#self.linear_large = interpolate.SmoothBivariateSpline(data[self.num:,0],data[self.num:,1],data[self.num:,2]/1e10, kx=4,ky=3)
T_switch = T_mid
linear_small = interpolate.SmoothBivariateSpline(
    data_low[:,0], data_low[:,1], data_low[:,2]/scale_low, kx=4, ky=3
)
linear_large = interpolate.SmoothBivariateSpline(
    data_high[:,0], data_high[:,1], data_high[:,2]/scale_high, kx=4, ky=3
)

# Save scales for later when evaluating potential
scale_low = scale_low
scale_high = scale_high

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
        if sig>GLUONIC_CUTOFF:
            return linear_small.ev(T,GLUONIC_CUTOFF)*scale_low
        else:
            return linear_small.ev(T,sig)*scale_high
    else:
        if sig>GLUONIC_CUTOFF:
            return linear_large.ev(T,GLUONIC_CUTOFF)*scale_high
        else:
            return linear_large.ev(T,sig)*scale_high


def linear(T, sigma):
    if T<90:
        return 0
    if T<data[num,0]:
        return linear_small.ev(T,sigma)*1e7
    else:
        return linear_large.ev(T,sigma)*1e10

Ts = range(15,1000,15)
sigmas = range(0,1000,15)

 


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
    plt.plot(sigmas,(_Vg(T,sigmas)[0]),label='interpolated', linestyle = 'dashed')
    plt.title(f'T={T}')
    plt.legend()
    plt.savefig("Temporal-Plots/bunchplots.pdf")
    debug_plot(name="debug", overwrite=False)
    #plt.show()
        