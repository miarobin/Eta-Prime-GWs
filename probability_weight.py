import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate, integrate
import os
from mpl_toolkits.mplot3d import Axes3D


#Lets solve equation 2.8 from https://arxiv.org/pdf/1809.08242
# Volume fraction converted to true vacuum
#Diferent cases I(t)

Mpl = 2.435e18 #GeV
g = 106.75
eps = np.sqrt(30.0/(np.pi**2*g))
Saction = 140.0
R_param = 1.
rhov = 0.5
rhor = 0.5
chi = rhov/rhor
Tv = 0.05
Tc = 1.0

# numerical integration tolerances
quad_args = {'epsabs':1e-9, 'epsrel':1e-8, 'limit':200}

def H_radiation(T):
    return T**2/(np.sqrt(3)*Mpl*eps)

def H_vacuum(Tv_local = Tv):
    return Tv_local**2/(np.sqrt(3)*Mpl*eps)

def gamma_T(T):
    return 1.0/ (R_param**4) * (Saction/2*np.pi)**2*np.exp(-Saction)

def chi(T):
    rhov = 0.5
    rhor = 0.5
    chi = rhov/rhor
    return chi  # constant here, could be T-dependent later

    
    
# CASE 1:  vw~1 eqn 2.8
    
def weight_vacuum(T):
    Hv = H_vacuum()
    pref = 4*np.pi/3
    a = (1.0 + 1.0/chi(T))**(-0.5)
    
    def inner_integral(Tprime):
        """Integrate over Ttwiddle from T to Tprime."""
        def integrand_inner(Ttwiddle):
            return 1.0 / (Hv  * np.sqrt(1 + 1/chi(Tprime)))
        val, _ = integrate.quad(integrand_inner, T, Tprime, epsrel=1e-6)
        return val

    def integrand_outer(Tprime):
        inner_val = inner_integral(Tprime)
        gTp = gamma_T(Tprime)
        denom = Hv * Tprime**4 * np.sqrt(1 + 1/chi(Tprime))
        return gTp/denom * inner_val**3

    val, _ = integrate.quad(integrand_outer, T, Tc, epsrel=1e-6)
    return pref * val
        
# CASE 2: "Radiation -> Vacuum" (RV) eqn 2.12

def weight_RV(T):
    Hv = H_vacuum()
    pref = 4.0 * np.pi / (3.0 * Hv**4)

    # first term: integral from Tv to Tc 
    def integrand1(Tp):
        if Tp <= Tv:
            return 0.0
        gTp = gamma_T(Tp)
        arg = (2.0*Tv - T - (Tv**2)/Tp)
        return gTp / (Tp**6) * (Tv**2) * (arg**3)

    # second term: integral from T to Tv 
    def integrand2(Tp):
        if Tp <= T or Tp > Tv:
            return 0.0
        gTp = gamma_T(Tp)
        return gTp / Tp * (1.0 - T/Tp)**3

    val1, err1 = integrate.quad(integrand1, Tv, Tc, **quad_args) if Tv < Tc else (0.0, 0.0)
    val2, err2 = integrate.quad(integrand2, max(T, 1e-12), Tv, **quad_args) if T < Tv else (0.0, 0.0)

    return pref * (val1 + val2)

# CASE 3: Radiation-dominated (R) eqn 2.14:

def weight_R(T):
    pref = 12.0 * np.pi * (Mpl * eps)**4

    def integrand(Tp):
        if Tp <= T:
            return 0.0
        gTp = gamma_T(Tp)
        return gTp / (Tp**6) * (1.0/T - 1.0/Tp)**3

    val, err = integrate.quad(integrand, max(T, 1e-12), Tc, **quad_args)
    return pref * val


T_values = np.logspace(-3, np.log10(Tc), 200)  # temperatures spanning below Tv up to Tc

W_vac = np.array([weight_vacuum(T) for T in T_values])
W_RV  = np.array([weight_RV(T) for T in T_values])
W_R   = np.array([weight_R(T) for T in T_values])

plt.figure(figsize=(8,5))
plt.loglog(T_values, np.abs(W_vac) + 1e-300, label='Vacuum-like (approx)')
plt.loglog(T_values, np.abs(W_RV)  + 1e-300, label='Radiation->Vacuum (RV)')
plt.loglog(T_values, np.abs(W_R)   + 1e-300, label='Radiation-dominated (R)')
plt.axvline(Tv, color='gray', linestyle='--', linewidth=0.8, label=f'Tv = {Tv}')
plt.xlabel('Temperature $T$ (GeV)')
plt.ylabel('Weight / Volume factor (arb. units)')
plt.title('Comparison of three weight functions')
plt.legend()
plt.grid(which='both', alpha=0.3)
plt.tight_layout()
plt.show()



# To verify I am doing the right thing, compare with Fig1 from the paper

   
    
# CASE 1:  vw~1 eqn 2.7
def Hvar_vacuum(T):
    Hv = H_vacuum()
    pref = 4*np.pi/3
    a = (1.0 + 1.0/chi(T))**(-0.5)
    
    def inner_integral(Tprime):
        """Integrate over Ttwiddle from T to Tprime."""
        def integrand_inner(Ttwiddle):
            return 1.0 / (Hv  * np.sqrt(1 + 1/chi(Tprime)))
        val, _ = integrate.quad(integrand_inner, T, Tprime, epsrel=1e-6)
        return val

    def integrand_outer(Tprime):
        inner_val = inner_integral(Tprime)
        gTp = gamma_T(Tprime)
        denom = Hv * Tprime**4 * np.sqrt(1 + 1/chi(Tprime))
        return gTp/denom * inner_val**3

    val, _ = integrate.quad(integrand_outer, T, Tc, epsrel=1e-6)
    return pref * val
        
# CASE 2: "Radiation -> Vacuum" (RV) eqn 2.11

def weight_RV(T):
    Hv = H_vacuum()
    pref = 4.0 * np.pi / (3.0 * Hv**4)

    # first term: integral from Tv to Tc 
    def integrand1(Tp):
        if Tp <= Tv:
            return 0.0
        gTp = gamma_T(Tp)
        arg = (2.0*Tv - T - (Tv**2)/Tp)
        return gTp / (Tp**6) * (Tv**2) * (arg**3)

    # second term: integral from T to Tv 
    def integrand2(Tp):
        if Tp <= T or Tp > Tv:
            return 0.0
        gTp = gamma_T(Tp)
        return gTp / Tp * (1.0 - T/Tp)**3

    val1, err1 = integrate.quad(integrand1, Tv, Tc, **quad_args) if Tv < Tc else (0.0, 0.0)
    val2, err2 = integrate.quad(integrand2, max(T, 1e-12), Tv, **quad_args) if T < Tv else (0.0, 0.0)

    return pref * (val1 + val2)

# CASE 3: Radiation-dominated (R) eqn 2.14:

def weight_R(T):
    pref = 12.0 * np.pi * (Mpl * eps)**4

    def integrand(Tp):
        if Tp <= T:
            return 0.0
        gTp = gamma_T(Tp)
        return gTp / (Tp**6) * (1.0/T - 1.0/Tp)**3

    val, err = integrate.quad(integrand, max(T, 1e-12), Tc, **quad_args)
    return pref * val

