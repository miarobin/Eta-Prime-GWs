import matplotlib.pyplot as plt
import numpy as np

Mpl = 2.435e18  # GeV
g = 106.75
eps = np.sqrt(30.0/(np.pi**2*g))
Saction = 140.0

#Hubble eqn2.9
def H_radiation(T):
    return T**2 / (np.sqrt(3)*Mpl*eps)

def H_vacuum(Tv):
    return Tv**2 / (np.sqrt(3)*Mpl*eps)

#Functions to evaluate

#Vacuum domination eqn2.10
'''def weight_V(T, Tv, Tp):
    Hv = H_vacuum(Tv)
    arg =  1/Hv * (1 - T/Tp)
    return  Hv * arg '''
    
def weight_V(T, Tv, Tp):
    Hv = H_vacuum(Tv)
    arg = 1/Hv * np.maximum(0, 1 - T/Tp)
    return arg

#Radiation to vacuum domination 2.11

'''def weight_RV(T, Tv, Tp):
    Hv = H_vacuum(Tv)
    arg = (1/(Tp*Hv)) * (2.0*Tv -  T - Tv**2/Tp)
    return Hv * arg '''

def weight_RV(T, Tv, Tp):
    Hv = H_vacuum(Tv)
    w_rad = np.sqrt(3)*Mpl*eps * (1/Tp) * (1/T - 1/Tp)
    w_vac = (1/(Tp*Hv)) * (2.0*Tv - T - Tv**2/Tp)
    
    if T >= 2*Tv:
        arg = w_rad
    elif T <= Tv:
        arg = w_vac
    else:
        # smooth cubic interpolation between Tv and 2Tv
        alpha = (T - Tv)/(2*Tv - Tv)
        alpha_smooth = alpha**2 * (3 - 2*alpha)  # "smoothstep" (C¹ continuous)
        arg = alpha_smooth * w_rad + (1 - alpha_smooth) * w_vac
    return Hv * arg


#eqn 2.13 Radiation domination             
def weight_R(T, Tv, Tp):
    Hv = H_vacuum(Tv)
    arg = np.sqrt(3)*Mpl*eps * (1/Tp) * (1/T - 1/Tp)
    return Hv * arg

Tv = 100.0
Tp = 5. * Tv 
Tp_vac = Tv  

T_values = np.linspace(0.01*Tv, 1.*Tp, 200)
x = T_values / Tv

#Evaluate
w_v = [weight_V(T, Tv, Tp_vac) for T in T_values]
w_RV = [weight_RV(T, Tv, Tp) for T in T_values]
w_R = [weight_R(T, Tv, Tp) for T in T_values]


#Plot
plt.figure(figsize=(8,5))
plt.loglog(x, np.abs(w_v), label=r'Vacuum-dominated (2.10)', color='purple', linestyle='--')
plt.loglog(x, np.abs(w_RV), label=r'RV (2.11)', color='green', linestyle='--')
plt.loglog(x, np.abs(w_R), label=r'Radiation-dominated (2.13)', color='orange', linestyle='--')
plt.xlabel(r'$T/T_V$')
plt.ylabel(r"$H_V\,a(T')\,r(T,T')$")
plt.legend()
plt.grid(which='both', alpha=0.3)
plt.xlim(1e-2, 10)
plt.ylim(1e-3, 1e1)
plt.tight_layout()
plt.savefig("Temporal-Plots/Ealmost.pdf", dpi=300)
plt.show()
print("save plot")


