import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy import interpolate
from debug_plot import debug_plot

GLUONIC_CUTOFF=1000
Tc=570

data = np.genfromtxt(f'fineGridDatamqsigma.csv', delimiter=',', dtype=float, skip_header=1)[68:]
dataNew = np.genfromtxt(f'VGluonicDataF3N3TEST.csv', delimiter=',', dtype=float, skip_header=0)
TTcsNew = np.genfromtxt(f'VGluonicDataF3N3TTcs.csv', delimiter=',', dtype=float, skip_header=0)
sigTcsNew = np.genfromtxt(f'VGluonicDataF3N3mqTcs.csv', delimiter=',', dtype=float, skip_header=0)

print(data.shape)
#Index 0 -> T/Tc; index 1 -> Sigma/Tc; index 2 -> V/Tc^4
 
# Split by temperature midpoint
T_mid = 0.5 * (min(data[:,0]) + max(data[:,0]))
low_mask = data[:,0] < T_mid
high_mask = data[:,0] >= T_mid

# Split by temperature midpoint
T_mid = 0.5 * (min(data[:,0]) + max(data[:,0]))
low_mask = data[:,0] < T_mid
high_mask = data[:,0] >= T_mid
 
data_low = data[low_mask]
data_high = data[high_mask]

print(data_low)
print(data_high)
# Automatic normalization
scale_low = np.max(np.abs(data_low[:,2]))
scale_high = np.max(np.abs(data_high[:,2]))

T_switch = T_mid
linear_small = interpolate.SmoothBivariateSpline(data_low[:,0], data_low[:,1], data_low[:,2]/scale_low, kx=4, ky=3
            )
linear_large = interpolate.SmoothBivariateSpline(
                data_high[:,0], data_high[:,1], data_high[:,2]/scale_high, kx=4, ky=3
            )
# Save scales for later when evaluating potential
scale_low = scale_low
scale_high = scale_high


def is_strictly_increasing(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            print(i)
            print(arr[i])
            print(arr[i+1])
            return False
    return True

# Example
print(is_strictly_increasing(TTcsNew))
linearNew = interpolate.RectBivariateSpline(TTcsNew,sigTcsNew,dataNew)
 

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



Ts = range(1,800,10)
sigmas = range(0,1000,15)

for T in Ts:
    print(T)
    temperaturepoint=[]
    for point in data:
        if round(T)==round(point[0]):
            temperaturepoint.append(point)
    
    #print(np.array(temperaturepoint))
    #print(_Vg(T,sigmas))
            
    temperaturepoint=np.array(temperaturepoint)    
    print(linearNew.ev(np.array(T)/Tc,np.array(10)/Tc)*Tc**4/temperaturepoint[1,2])

       
    plt.plot(temperaturepoint[:,1],temperaturepoint[:,2],label='real data')
    plt.plot(sigmas,(_Vg(T,sigmas)[0]),label='interpolated',linestyle='dashed')
    plt.plot(sigmas, linearNew.ev(np.array(T)/Tc,np.array(sigmas)/Tc)*Tc**4,label='New interpolated')

    plt.title(f'T={T}')
    plt.legend()
    plt.savefig("Temporal-Plots/bunchplots.pdf")
    debug_plot(name="debug", overwrite=False)
    #plt.show()

for i,TTc in enumerate(TTcsNew):
    plt.plot(sigTcsNew*Tc, dataNew[i,:], label='raw data')
    plt.plot(sigTcsNew*Tc, linearNew.ev(TTc,sigTcsNew), label='interpolated')
    plt.title(f'T={T}')
    plt.legend()
    plt.savefig("Temporal-Plots/bunchplots.pdf")
    debug_plot(name="debug", overwrite=False)

