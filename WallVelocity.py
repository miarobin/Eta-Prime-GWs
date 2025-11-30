import Potential2
import numpy as np
import csv
from scipy.integrate import solve_ivp ,simpson
from scipy.optimize import root_scalar


'''Computes the Bubble Wall Velocity. The first functions are taken from the code snippet in https://arxiv.org/abs/2303.10171

The two additional functions I have written are:
	1. "save_arrays_to_csv" is a generic function for saving data as a csv array.

	    save_arrays_to_csv: 
        INPUTS:  (file_path, column_titles, *arrays)
                (string, array, arrays)
        OUTPUTS: Nothing
    
    2. "readAndEdit" which reads a data file 'filename.csv' which contains Lagrangian parameters:
        ['m2Sig','m2Eta','m2X','fPI','m2','c','lambda_sigma','lambda_a','Tc'] with the GW signal 
        parameters for N colours and F flavours. 
         
        The bubble wall velocity is then calculated using the functions from the paper 2303.10171 and added to a new 
        csv file.
        
        readAndEdit:
        INPUTS: (filename, N, F, termType)
                (string, int, int, string)
        OUTPUTS: Nothing.
    
    '''
###### GLOBAL VARIABLES ######

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
    DThetaBar = thetaBar(0,Tn) - thetaBar(minima,Tn)
	
    symEnthalpy = - Tn * V.dVdT(0,Tn)
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
    vp_max = min(cs2/(vw+1e-10),vw)
    al_min = max((vm-vp_max)*(cb2-vm*vp_max)/(3*cb2*vm*(1-vp_max**2)),(mu-nu) /(3*mu))
    al_max = 1/3
    branch = -1
    if eqWall(al_min,alN,vm,nu,mu,psiN)*eqWall(al_max,alN,vm,nu,mu,psiN)>0:
        branch = 1
    sol = root_scalar(eqWall,(alN,vm,nu,mu,psiN,branch),bracket=(al_min,
    al_max),rtol=1e-10,xtol=1e-10) 
    if not sol.converged:
        print("WARNING: desired precision not reached in ’solve_alpha’")
        print(f'alN={alN},vm={vm},nu={nu},mu={mu},branch={branch}')
        print(eqWall(al_min,alN,vm,nu,mu))
        print(eqWall(al_max,alN,vm,nu,mu))
        print(sol)
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
    try:
        sol = integrate_plasma((vw-vp)/(1-vw*vp), vw, wp, cs2) 
        vp_sw = sol.y[0,-1]
        vm_sw = (vp_sw-sol.t[-1])/(1-vp_sw*sol.t[-1])
        wm_sw = sol.y[1,-1]
        return vp_sw/vm_sw - ((mu-1)*wm_sw+1)/((mu-1)+wm_sw)
    except ValueError as e:
        print(e)
        print(f'Inputs Causing Error: vw={vw} and wp={wp}. Returns large numerical value.')
        return 1e20

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

def find_kappa (alN, cb2, cs2, psiN, vw=None):
    if vw is None:
        vw = find_vw (alN, cb2, cs2, psiN)
    nu , mu = 1+1/cb2 ,1+1/cs2
    kappa, wp, vm, vp = 0, 1, 0, 0
    if vw < 1:
        vm = min (np.sqrt(cb2), vw)
        al = solve_alpha(vw, alN, cb2, cs2, psiN)
        vp = get_vp(vm, al, cb2)
        wp = w_from_alpha(al, alN, nu, mu)
        try:
            sol = integrate_plasma((vw - vp)/(1 - vw * vp), vw, wp, cs2)
            v , xi , w = sol.t, sol.y[0], sol .y[1]
            kappa += 4*simpson((xi * v)**2*w/(1 - v**2), xi)/(vw**3*alN)
        except ValueError as e:
            print(e)
            print(f'Inputs Causing Error: vw={vw},wp={wp}. Adds nothing to efficiency factor.')
            return kappa

    if vw **2 > cb2 :
        w0 = psiN * wp**(nu / mu)*((1 - vm**2)/(1 - v**2))**(nu/2) if vw < 1 else 1+6*alN/(nu-2)
        v0 = (vw - vm) /(1 - vw * vm) if vw < 1 else 3* alN /(nu-2+3*alN)
        try:
            sol = integrate_plasma(v0, vw, w0, cb2, False)
            v , xi , w = np.flip(sol.t), np.flip(sol.y[0]), np.flip(sol.y[1])
            mask = np.append(xi[1:] > xi[: -1], True)
            kappa += 4*simpson(((xi * v)**2*w/(1 - v **2))[mask], xi[mask]) /(vw**3*alN)
        except ValueError as e:
            print(e)
            print(f'Inputs Causing Error: vw={vw},wp={w0}. Adds nothing to efficiency factor.')
        
    return kappa




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

def readAndEdit(filename, N, F, termType):
    delimiter = ','
    #Read into a numpy array the edited data.
    data = np.array(np.genfromtxt(filename, delimiter=delimiter, skip_header=1, dtype=None))

    #Empty arrays to store the needed potential parameters from 'data'.
    m2Sigs = []; m2Etas = []; m2Xs = []; fPIs = []; m2s = []; cs = []; lss = []; las = []; Tcs = []; Tns = []; Alphas = []; Betas = []; Messages = []
    VwsLTE = np.zeros(len(data)); kappasLTE = np.zeros(len(data))

    for item in data:
        #Zero T particle masses
        m2Sigs.append(item[0]); m2Etas.append(item[1]); m2Xs.append(item[2]); fPIs.append(item[3]); 
        #Potential Parameters
        m2s.append(item[4]); cs.append(item[5]); lss.append(item[6]); las.append(item[7]); Tcs.append(item[8]); 
        #GW Parameters
        Tns.append(item[9]); Alphas.append(item[10]); Betas.append(item[11])

    m2Sigs = np.array(m2Sigs); m2Etas = np.array(m2Etas); m2Xs = np.array(m2Xs); fPIs = np.array(fPIs)
    m2s = np.array(m2s); cs = np.array(cs); lss = np.array(lss); las = np.array(las); Tcs=np.array(Tcs)
    Tns = np.array(Tns); Alphas = np.array(Alphas); Betas = np.array(Betas); Messages = np.array(Messages)

    #Scanning over each row in 'data' and calculating new Vw.
    for i in range(len(m2s)):
        if Tns[i]>1:
            detPow = Potential2.get_detPow(N,F,termType)
            V = Potential2.Potential(m2s[i], cs[i], lss[i], las[i], N, F, detPow)
            minima = V.findminima(Tns[i])
            psiN = V.dVdT(minima,Tns[i])/V.dVdT(0,Tns[i])
            
            cs2 = V.dVdT(0,Tns[i])/(Tns[i]*V.d2VdT2(0,Tns[i]))
            cb2 = V.dVdT(minima,Tns[i])/(Tns[i]*V.d2VdT2(minima,Tns[i]))
            alN = alpha(Tcs[i], Tns[i], cb2)
            
            VwsLTE[i] = find_vw(alN,cb2,cs2)
            kappasLTE[i] = find_kappa(alN, cb2, cs2, psiN, VwsLTE[i])

            #Potential Parameters
            print(rf'$m^2$ = {m2s[i]}, $c$ = {cs[i]}, $\lambda_\sigma$ = {lss[i]}, $\lambda_a$ = {las[i]}')
            #Sound speed
            print(rf'$c_{{\text{{sound,sym}}}}^2$ = {cs2}, $c_{{\text{{sound,b}}}}^2$ = {cb2}')
            #GW Parameters
            print(rf'$\Psi$_N$ = {0}, $\alpha_N$ = {alN}, VwLTE = {VwsLTE[i]}, kappa = {kappasLTE[i]}')
            
    save_arrays_to_csv(f'VwLTE_N{N}F{F}_{termType}.csv',
                           ['m2Sigs', 'm2Etas', 'm2X', 'fPi', 'm2', 'c', 'ls', 'la', 'Tc', 'Tn', 'Alpha', 'Beta', 'Message', 'VwsLTE','kappasLTE'], 
                            m2Sigs, m2Etas, m2Xs, fPIs, m2s, cs, lss, las, Tcs, Tns, Alphas, Betas, Messages, VwsLTE, kappasLTE)



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

    
    readAndEdit('Test_N3F3_Normal2401.csv', 3, 3, "Normal")
    '''
    F=6; N=3
    
    m2=	65016.666666666700;	c= 1E-10;ls=0.13006666666666700;la=	1.7499666666666700	
    Tc=264.3718266272440;Tn=	229.1092491831630
    detPow = Potential2.get_detPow(N,F,'Normal')
    V = Potential2.Potential(m2, c, ls, la, N, F, detPow)
    
    Ts = np.linspace(Tn, Tc)
    Vws = []
    for T in Ts:
        minima = V.findminima(T)
        psiN = V.dVdT(minima,T)/V.dVdT(0,T)
                
        cs2 = V.dVdT(0,T)/(T*V.d2VdT2(0,T))
        cb2 = V.dVdT(minima,T)/(T*V.d2VdT2(minima,T))
        alN = alpha(V, T, cb2)
                    
        Vws.append(find_vw(alN,cb2,cs2))
        
    plt.plot(Ts, Vws)
    plt.show()'''
    
    


