from cosmoTransitions.tunneling1D import SingleFieldInstanton
from cosmoTransitions.tunneling1D import PotentialError
from cosmoTransitions import helper_functions
from functools import reduce
import Potential, GravitationalWave
from scipy import interpolate, optimize
import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.integrate import solve_ivp ,simps
from scipy.optimize import root_scalar

#Calculate g_star
data_array = np.loadtxt('gstar_data.dat')
Ts = data_array[:,0]
g_stars = data_array[:,1]
_g_star = interpolate.interp1d(Ts, g_stars)
##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2


def wallVelocity(V, alpha, Tn):
    #Jouget velocity.
    vJ = (1/np.sqrt(3)) * (1 + np.sqrt(3*alpha**2 + 2*alpha))/(1+alpha)

    lhs, rhs = V.findminima(Tn)
    psi = V.dVdT(rhs,Tn)/V.dVdT(lhs,Tn)
    print(f'psi = {psi}')

    a = 0.2233; b = 1.704; p = -3.433
    smallV = np.sqrt(np.abs(0.5*(3*alpha + psi - 1)/(2-3*psi+psi**3)))
    bigV = vJ*(1-a*(1-psi)**b/alpha)

    if alpha > max((1-psi)/3, 0):
        vw = (np.abs(smallV)**p + np.abs(bigV)**p)**(1/p)
        if vw < vJ: return vw
        else: return 0
    else:
        return 0

def alpha(V, Tn, cb2):
    minima = V.findminima(Tn)
    thetaBar = lambda h,T: -T*V.dVdT(h,T)+V.Vtot(h,T) + V.Vtot(h,T)/cb2
    DThetaBar = thetaBar(minima[0],Tn) - thetaBar(minima[1],Tn)
	
    symEnthalpy = - Tn * V.dVdT(minima[0],Tn)
    return DThetaBar/( 3 * symEnthalpy )
    


def find_vJ(alN, cb2):
    return np.sqrt(cb2)*(1+np.sqrt(3*alN*(1-cb2+3*cb2*alN)))/(1+3*cb2*alN)

def get_vp(vm, al, cb2, branch=-1):
    disc = vm**4-2*cb2*vm**2*(1-6*al)+cb2**2*(1-12*vm**2*al*(1-3*al))
    return 0.5*(cb2+vm**2+branch*np.sqrt(disc))/(vm+3*cb2*vm*al)

def w_from_alpha(al,alN,nu,mu):
    return (abs((1-3*alN)*mu-nu)+1e-100)/(abs((1-3*al)*mu-nu)+1e-100)

def eqWall(al,alN,vm,nu,mu,psiN,solution=-1):
    vp = get_vp(vm,al,1/(nu-1),solution)
    ga2m,ga2p= 1/(1-vm**2),1/(1-vp**2)
    psi = psiN*w_from_alpha(al,alN,nu,mu)**(nu/mu-1)
    return vp*vm*al/(1-(nu-1)*vp*vm)-(1-3*al-(ga2p/ga2m)**(nu/2)*psi)/(3*nu)

def solve_alpha(vw,alN,cb2,cs2,psiN): 
    nu,mu = 1+1/cb2,1+1/cs2
    vm = min(np.sqrt(cb2),vw)
    vp_max = min(cs2/vw,vw)
    al_min = max((vm-vp_max)*(cb2-vm*vp_max)/(3*cb2*vm*(1-vp_max**2)),(mu-nu) /(3*mu))
    al_max = 1/3
    branch = -1
    if eqWall(al_min,alN,vm,nu,mu,psiN)*eqWall(al_max,alN,vm,nu,mu,psiN)>0:
        branch = 1
    sol = root_scalar(eqWall,(alN,vm,nu,mu,psiN,branch),bracket=(al_min,
    al_max),rtol=1e-10,xtol=1e-10) 
    if not sol.converged:
        print("WARNING: desired precision not reached in ’solve_alpha’") 
    return sol.root

def dfdv(v,X,cs2): 
    xi,w=X
    mu_xiv = (xi-v)/(1-xi*v)
    dxidv = xi*(1-v*xi)*(mu_xiv**2/cs2-1)/(2*v*(1-v**2))
    dwdv = w*(1+1/cs2)*mu_xiv/(1-v**2)
    return [dxidv ,dwdv]

def integrate_plasma(v0,vw,w0,c2,shock_wave=True):
    def event(v,X,cs2): 
        xi,w = X
        return xi*(xi-v)/(1-xi*v) - cs2 
    event.terminal = True
    sol = None
    if shock_wave:
        sol = solve_ivp(dfdv,(v0,1e-20),[vw,w0],events=event,args=(c2,),rtol =1e-10,atol=1e-10)
    else:
        sol = solve_ivp(dfdv,(v0,1e-20),[vw,w0],args=(c2,),rtol=1e-10,atol=1e-10)
    if not sol.success:
        print("WARNING: desired precision not reached in ’integrate_plasma’") 
    return sol

def shooting(vw,alN,cb2,cs2,psiN):
    nu,mu = 1+1/cb2,1+1/cs2
    vm = min(np.sqrt(cb2),vw)
    al = solve_alpha(vw, alN, cb2, cs2, psiN) 
    vp = get_vp(vm, al, cb2)
    wp = w_from_alpha(al, alN, nu, mu)
    sol = integrate_plasma((vw-vp)/(1-vw*vp), vw, wp, cs2) 
    vp_sw = sol.y[0,-1]
    vm_sw = (vp_sw-sol.t[-1])/(1-vp_sw*sol.t[-1])
    wm_sw = sol.y[1,-1]
    return vp_sw/vm_sw - ((mu-1)*wm_sw+1)/((mu-1)+wm_sw)

def find_vw(alN,cb2,cs2,psiN):
    nu,mu = 1+1/cb2,1+1/cs2
    vJ = find_vJ(alN, cb2)
    if alN < (1-psiN)/3 or alN <= (mu-nu)/(3*mu):
        print('alN too small')
        return 0
    if alN > max_al(cb2,cs2,psiN,100) or shooting(vJ,alN,cb2,cs2,psiN) < 0:
        print('alN too large')
        return 1
    sol = root_scalar(shooting ,(alN,cb2,cs2,psiN),bracket=[1e-3,vJ],rtol=1e-10,xtol=1e-10)
    return sol.root


def max_al(cb2,cs2,psiN,upper_limit=1): 
    nu,mu = 1+1/cb2,1+1/cs2
    vm = np.sqrt(cb2)
    def func(alN):
        vw = find_vJ(alN, cb2)
        vp = cs2/vw
        ga2p,ga2m = 1/(1-vp**2),1/(1-vm**2)
        wp = (vp+vw-vw*mu)/(vp+vw-vp*mu)
        psi = psiN*wp**(nu/mu-1)
        al = (mu-nu)/(3*mu)+(alN-(mu-nu)/(3*mu))/wp
        return vp*vm*al/(1-(nu-1)*vp*vm)-(1-3*al-(ga2p/ga2m)**(nu/2)*psi)/(3*nu)
    if func(upper_limit) < 0: return upper_limit
    sol = root_scalar(func,bracket=((1-psiN)/3,upper_limit),rtol=1e-10,xtol=1e-10)
    return sol.root

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

def readAndEdit(filename, g4, beta):
    delimiter = ','
    #Read into a numpy array the edited data.
    data = np.array(np.genfromtxt(filename, delimiter=delimiter, skip_header=2, dtype=None))

    Eps = []; Deltas = []; Tns = []; Alphas = []; Betas = []; Vws = []; Messages = []
    for item in data:
        Eps.append(item[0]); Deltas.append(item[1]); Tns.append(item[2]); Alphas.append(item[3]); Betas.append(item[4]); Vws.append(item[5]); Messages.append(item[6])

    Eps = np.array(Eps); Deltas = np.array(Deltas); Tns = np.array(Tns); Alphas = np.array(Alphas); Betas = np.array(Betas); Vws = np.array(Vws)

    for i in range(len(Eps)):
        if Tns[i]>1:
            V = Potential.Potential(Eps[i],g4,Potential.gammaa(Eps[i],g4,Deltas[i]),beta,higgs_corr=True,loop=True)
            minima = V.findminima(Tns[i])
            psiN = V.dVdT(minima[1],Tns[i])/V.dVdT(minima[0],Tns[i])
            cb2 = V.dVdT(minima[1],Tns[i])/(Tns[i]*V.d2VdT2(minima[1],Tns[i]))
            cs2 = V.dVdT(minima[0],Tns[i])/(Tns[i]*V.d2VdT2(minima[0],Tns[i]))
            alp = alpha(V, Tns[i], cb2)
            Vws[i] = find_vw(alp,cb2,cs2,psiN)
                #Vws[i] = wallVelocity(V, alp, Tns[i])
            print(rf'$\epsilon$ = {Eps[i]}, $\delta$ = {Deltas[i]}')
            print(rf'$c_s^2$ = {cs2}, $c_b^2$ = {cb2}')
            print(rf'$\Psi$_N$ = {psiN}, $\alpha$ = {alp}, Vw = {Vws[i]}')
            
    
    save_arrays_to_csv('EditedScans/newvwsmalleps_g4_1.6_beta_0.32.csv',['Epsilon', 'Delta', 'Tn', 'Alpha', 'Beta', 'Vw', 'Message'],Eps, Deltas, Tns, Alphas, Betas, Vws, Messages)

if __name__ == "__main__":
    '''
    eps = 0.045; g4 = 1.6; delta = -0.1; beta = np.sqrt(0.1)
    print(f"$\gamma_a$ = {Potential.gammaa(eps,g4,delta)}")
    #The Potential Object.
    V = Potential.Potential(eps,g4,Potential.gammaa(eps,g4,delta),beta,higgs_corr=True,loop=True)

    Tn, _, message = GravitationalWave.grid(V)
    print(f'Tn = {Tn} and message = {message}')
    alp = alpha(V, Tn, 1/np.sqrt(3))
    vw = wallVelocity(V,alp,Tn)
    print(f'new alpha = {alp}, formula vw = {vw}')
    

    oldalp = GravitationalWave.alpha(V, Tn)
    oldvw = GravitationalWave.wallVelocity(V, oldalp, Tn)
    print(f'old alpha = {oldalp}, vw = {oldvw}')
    
    minima = V.findminima(Tn)
    psiN = V.dVdT(minima[1],Tn)/V.dVdT(minima[0],Tn)
    print(f'Numerical Vw = {find_vw(alp,1/3,1/3,psiN)}')
    '''
    
    readAndEdit('EditedScans/smalleps_g4_1.6_beta_0.32.csv', 1.6, np.sqrt(0.1))


