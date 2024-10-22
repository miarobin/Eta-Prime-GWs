import Potential
import GravitationalWave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate, optimize
import csv
from multiprocessing import Pool


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



def populate(xi, muSig, lmb, kappa, m2Sig, N, F):
    fPi = 50.2
	#Lambda, Kappa, m^2_Sigma, Mu_Sig, Xi.
    V = Potential.Potential(xi, muSig, lmb, kappa, m2Sig*fPi**(4-4/N), N, F)
    fSig = V.findminima(0) 

    if fSig is None or (N==1 and (kappa+4*lmb<muSig)):
        #Error message #15:
        print('Potential does not have 3 real solutions')
        return (0, 0, 0, 0, 15)

    massRatio = abs(V.mSq['Phi'][0](fSig,0)/V.mSq['Eta'][0](fSig,0))
    
    Tn, grd, message = GravitationalWave.grid(V,prnt=True,plot=False)

    if Tn is not None:
        alpha = abs(GravitationalWave.alpha(V,Tn)); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = 1
        print(f"\mu_\Sigma = {muSig}, \lambda={lmb}, \kappa={kappa}, m^2_\Sigma={m2Sig}")
        print(f"Tn = {Tn}, alpha = {alpha}, betaH = {betaH}, massRatio = {massRatio}")
        return (Tn, alpha, betaH, massRatio, message)
    else:
        print('CT Returned None')
        return (0, 0, 0, massRatio, message)
	



def parallelScan(muSigs, lmbs, kappas, m2Sigs, N, F):
	#MAKE THE ARRAY
    result=[]
	
    points = [(0, muSig, lmb, kappa, m2Sig, N, F) for muSig in muSigs for lmb in lmbs for kappa in kappas for m2Sig in m2Sigs]
	#Multithreading with 16 cores.
    with Pool(8) as p:
        result = p.starmap(populate, points)
		
    points=np.array(points)
    MuSigs = points[:,1]; Lmbs = points[:,2]; Kappas = points[:,3]; M2Sigs = points[:,4]; Tns = []; Alphas = []; Betas = []; MassRatios = []; Messages = []
    for item in result:
        Tns.append(item[0]); Alphas.append(item[1]); Betas.append(item[2]); MassRatios.append(item[3]); Messages.append(item[4])

    column_titles = ['mu_Sigma', 'Lambda', 'Kappa', 'm^2_Sigma', 'Tn', 'Alpha', 'Beta', 'MassRatio', 'Message']
	# File path to save the CSV
    file_path = f'Scan_N{N}F{F}.csv'
    save_arrays_to_csv(file_path, column_titles, MuSigs, Lmbs, Kappas, M2Sigs, Tns, Alphas, Betas, MassRatios, Messages)

    #MAKE THE PLOT
    _Lmbs, _Kappas = np.meshgrid(sorted(Lmbs), sorted(Kappas))
    _Tns = interpolate.griddata((Lmbs, Kappas), Tns, (_Lmbs, _Kappas), method='linear')

    fig, ax = plt.subplots(2,2)
    im0 = ax[0,0].pcolormesh(_Lmbs, _Kappas, _Tns)
    cbar = plt.colorbar(im0)
    cbar.set_label(r"$T_n$")
    ax[0,0].set_xlabel(r"$\lambda$")
    ax[0,0].set_ylabel(r"$\kappa$")

    _MuSigs, _M2Sigs = np.meshgrid(sorted(MuSigs), sorted(M2Sigs))
    _Tns = interpolate.griddata((MuSigs, M2Sigs), Tns, (_MuSigs, _M2Sigs), method='linear')

    im1 = ax[0,1].pcolormesh(_MuSigs, _M2Sigs, _Tns)
    cbar = plt.colorbar(im1)
    cbar.set_label(r"$T_n$")
    ax[0,1].set_xlabel(r"$\mu_\Sigma$")
    ax[0,1].set_ylabel(r"$M^2_\Sigma$")
	
    _Kappas, _M2Sigs = np.meshgrid(sorted(Kappas), sorted(M2Sigs))
    _Tns = interpolate.griddata((Kappas, M2Sigs), Tns, (_Kappas, _M2Sigs), method='linear')

    im2 = ax[1,0].pcolormesh(_Kappas, _M2Sigs, _Tns)
    cbar = plt.colorbar(im2)
    cbar.set_label(r"$T_n$")
    ax[1,0].set_xlabel(r"$\kappa$")
    ax[1,0].set_ylabel(r"$M^2_\Sigma$")
    
    _Lmbs, _MuSigs = np.meshgrid(sorted(Lmbs), sorted(MuSigs))
    _Tns = interpolate.griddata((Lmbs, MuSigs), Tns, (_Lmbs, _MuSigs), method='linear')

    im3 = ax[1,1].pcolormesh(_Lmbs, _MuSigs, _Tns)
    cbar = plt.colorbar(im3)
    cbar.set_label(r"$T_n$")
    ax[1,1].set_xlabel(r"$\lambda$")
    ax[1,1].set_ylabel(r"$\mu_\Sigma$")

    plt.show()


if __name__ == "__main__":
    print('hello')
    
    F=4; N=2
    def fSig1_function(muSig, lmb, kappa, m2Sig): 
        if F==4 and N==1: return 2*(2*m2Sig)**0.5 / (kappa + 4*lmb - muSig)**0.5
        elif F==4 and N>1: return 2*(2*m2Sig)**0.5 / (kappa + 4*lmb)**0.5

    MuSigs = [0]
    Lmbs = [10, ]
    Kappas = [0]
    M2Sigs = [10**2, 50**2, 100**2,1000**2, 5000**2]
    
    fPi = 95
    #parallelScan(MuSigs, Lmbs, Kappas, M2Sigs, 1, 4)
    parallelScan(fPi**2 * MuSigs, Lmbs, Kappas, M2Sigs, 2, 4)
    


    