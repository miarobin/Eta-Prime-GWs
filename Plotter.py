import Potential2
import GravitationalWave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import csv
from multiprocessing import Pool
import DressedMasses

'''
	Please see the dockstring in "Potential2.py" for a parameter dictionary!!
	
	This file does the following:
	1. "save_arrays_to_csv" is a generic function for saving data as a csv array.

	    save_arrays_to_csv: 
        INPUTS:  (file_path, column_titles, *arrays)
                (string, array, arrays)
        OUTPUTS: Nothing
		
	2. "populate" is a function to generate gravitational wave data from a single set of potential parameters. The input detPow can be used to
		change the potential to a modified potential from https://arxiv.org/abs/2307.04809. 
		Returns (0,0,0,0, (error code), 0, 0, 0, 0) if no phase transition. NB the 0's are used to avoid crashes later on.
	
		populate:
		INPUTS: (mSq, c, lambdas, lambdaa, N, F, detPow, plot=False)
				(float, float, float, float, int, int, float, bool)
		OUTPUT: (Tn, alpha, betaH, vW, message, _m2Sig, _m2Eta, _m2Pi, _m2X)
				(float, float, float, float, int, float, float, float, float)
				
	3. "plotDifference" allows you to see, for the large N (half-transparency) and normal (full-transparency) cases, 
		the difference in GW data for cases with the same zero-temp masses. These are indicated by having the same marker. 
		However, this function can only handle up to 22 data points or it does not have enough markers.
		
		TLDR large N IS HALF TRANSPARENCY; NORMAL IS FULL TRANSPARENCY. CAN ONLY HANDLE 22 DATA POINTS.
		
		plotDifference:
		INPUTS: (reslN, resN)
				(np.array, np.array)
		
				
	4. "populateN" and "populatelN" are wrapper functions for "populate" for the normal and large N case respectively. Does not allow "plot" in populate.
	
		populateN/ln:
		INPUTS: (mSq, c, lambdas, lambdaa, N, F, detPow, polyakov=False)
				(float, float, float, float, int, int, bool, bool)
		OUTPUT: (Tn, alpha, betaH, vW, message, _m2Sig, _m2Eta, _m2Pi, _m2X)
				(float, float, float, float, int, float, float, float, float)
				
	5. "parallelScan" is a function to perform a parallelised scan over a set of zero-temp particle masses:
			a Loops over each array sequentially and builds a new list of valid Lagrangian inputs. These are stored in lN_data and N_data.
			b Runs populateN and populateC over the N_data and lN_data arrays respectively, in parallel over, say, 8, cores.
			c Stores the run results in two csv arrays. 
			d Attempts to run plotDifference (though note that if there are more than 22 non-zero data points this will crash.
		
		parallelScan:
		INPUTS: (m2Sig, m2Eta, m2Pi, m2X, N, F)
				(np.array, np.array, np.array, np.array, int, int)
				
	6. "getTcs" does something.
	
		getTcs:
		INPUTS (m2Sig, m2Eta, m2X, fPI, N, F)
				(np.array, np.array, np.array, np.array, int, int)
		
	NOTE The code at the end of the file only runs when this file is run directly. Adjust the scan ranges as necessary.
'''
##GLOBAL VARIABLES##
CORES=1


def plotV(V, Ts):
	for T in Ts:
		plt.plot(np.linspace(-5,V.fSigma()*1.25,num=100)/V.fSigma(),V.Vtot(np.linspace(-5,V.fSigma()*1.25,num=100),T)/V.fSigma()**4-V.Vtot(0,T)/V.fSigma()**4,label=f"T={T}")


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


def populate(mSq, c, lambdas, lambdaa, N, F, detPow, Polyakov=True, plot=False):
	
	#Building the potential...
	try:
		V = Potential2.Potential(mSq, c, lambdas, lambdaa, N, F, detPow, Polyakov=Polyakov)
	except Potential2.InvalidPotential as e:
		print(e)
		return (0, 0, 0, 0, 0, 16) #Dressed mass calculation has failed for this.
	
	#Calculating the zero temperature, tree level, analytic minimum.
	fSig = V.fSigma()
	print(f'fSigma={fSig}')
	print(f'm2_sig={V.mSq['Sig'][0](fSig)}, m2Eta={V.mSq['Eta'][0](fSig)}, m2X={V.mSq['X'][0](fSig)}, m2Pi={V.mSq['Pi'][0](fSig)}')
	
	if plot:
		#Plotting the dressed masses
		DressedMasses.SolveMasses(V,plot=True)
		#Plots the potential as a function of temperature
		def plotV(V, Ts):
			for T in Ts:
				plt.plot(np.linspace(-5,fSig*1.2,num=300),V.Vtot(np.linspace(-5,fSig*1.2,num=300),T)-V.Vtot(0,T),label=f"T={T}")
			plt.legend()
			plt.show()
			
		#Do feel free to change this list of temperatures to something more sensible.
		plotV(V,[0,100,150,200,225,250,400,450,500,510,fSig])
		
	if fSig == None:
		#If fSig does not exist, then the potential does not have enough solutions for a tunneling. Return None.
		return (0, 0, 0, 0, 0, 15)
	
	#Grid function computes:
	#	a) Nucleation temperature Tn,
	#	b) An interpolated function grd of action over temperature w/ temperature, 
	#	c) and an error code.
	Tn, grd, tc, message = GravitationalWave.grid(V,prnt=True,plot=plot)
	
	if Tn is not None:
		#Bubbles nucleate before BBN! Yay!
		
		#Calculating wave parameters.
		alpha = abs(GravitationalWave.alpha(V,Tn)); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = GravitationalWave.wallVelocity(V, alpha, Tn)
		print(f"Tn = {Tn}, alpha = {alpha}, betaH = {betaH}")
		
		#Returning wave parameters and zero-temperature particle masses.
		return (Tn, alpha, betaH, 1, tc, message)
	
	else:
		#If Tn is none, bubbles do not nucleate in time.
		print(f'CT Returned None with message {message}')
		
		#Returns the failure state, the associated failure code, and the associated zero-temperature particle masses.
		return (0, 0, 0, 0, tc, message)
	

def plotDifference(reslN, resN):

	#Fixed list of markers to track each set of zero-temp masses.
	markers = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","D","d","|"]
	
	colormap = plt.cm.plasma 
	#normalising the colour map to between the min and max values of _mSig
	normalize = matplotlib.colors.Normalize(vmin=min(np.ravel(reslN[:,5]+resN[:,5])), vmax=max(np.ravel(reslN[:,5]+resN[:,5])))
	

	for i in range(len(reslN)):
		if reslN[i,2]!=0 and resN[i,2]!=0:
			#large N term has half-transparency.
			plt.scatter(reslN[i,2], reslN[i,1], c = reslN[i,5], alpha=1/(2), marker=markers[-1], cmap=colormap, norm=normalize)
			#Normal term has full-transparency.
			plt.scatter(resN[i,2], resN[i,1], c = resN[i,5], alpha=1/(1), marker=markers[-1], cmap=colormap, norm=normalize)
			
			#Pops off the last marker to move onto the next one.
			markers.pop()

			
	plt.xscale("log")
	plt.yscale("log")
	plt.colorbar(label=f'$m^2_\sigma$')
	plt.xlabel(f'beta/H')
	plt.ylabel(f'alpha')
	plt.show()
	
	
def populateN(mSq, c, ls, la, N, F, Polyakov=True,plot=True):
	#Wrapper function for normal case.
	detPow = Potential2.get_detPow(N,F,"Normal")
	print(f'Normal: m2={mSq},c={c},ls={ls},la={la},N={N},F={F},p={detPow}')
	return populate(mSq, c, ls, la, N, F, detPow, Polyakov=Polyakov, plot=plot)
def populatelN(mSq, c, ls, la, N, F, Polyakov=True,plot=True):
	#Wrapper for the largeN case.
	detPow = Potential2.get_detPow(N,F,"largeN")
	print(f'largeN: m2={mSq},c={c},ls={ls},la={la},N={N},F={F},p={detPow}')
	return populate(mSq, c, ls, la, N, F, detPow, Polyakov=Polyakov, plot=plot)



def parallelScan(m2Sig,m2Eta,m2X, fPI, N, F):
	
	#MAKE THE ARRAY
	data = []
	for i in m2Sig:
		for j in m2Eta:
			for k in m2X:
				for l in fPI:
					if k>j and k>i:
						data.append([i,j,k,l,N,F])
	#Arrays to store the Lagrangian inputs
	lN_LInputs = []; N_LInputs = []
	#Arrays to store the zero-temperature masses
	lN_Masses = []; N_Masses = []
	for i in range(len(data)):
		point = data[i]
		print(point)
		#Calculating the Lagrangian inputs. See appendix D of draft.
		lN_Linput = [*Potential2.masses_to_lagrangian(*point,Potential2.get_detPow(N,F,"largeN")),N,F,"largeN"]
		N_Linput = [*Potential2.masses_to_lagrangian(*point,Potential2.get_detPow(N,F,"Normal")),N,F,"Normal"]
		#Only keeping point if BOTH largeN and Normal are valid. <---- May want to change this later.
		if (lN_Linput[0] is not None) and (N_Linput[0] is not None): 
			lN_LInputs.append(lN_Linput)
			N_LInputs.append(N_Linput)
			lN_Masses.append(point)
			N_Masses.append(point)
			
	#Cropping the data for now.
	crop=40
	lN_LInputs=lN_LInputs[:crop]
	N_LInputs=N_LInputs[:crop]
	lN_Masses=lN_Masses[:crop]
	N_Masses=N_Masses[:crop]
	
        
	#Multithreading with X cores.
	with Pool(CORES) as p:
		#Populating the result arrays.
		resN = p.starmap(populateN, N_LInputs)
		reslN = p.starmap(populatelN, lN_LInputs)
	
	#MAKE THE FILE WRITER
	#Column Titles
	N_LInputs = np.array(N_LInputs); lN_LInputs = np.array(lN_LInputs); lN_Masses=np.array(lN_Masses); N_Masses=np.array(N_Masses); resN=np.array(resN); reslN=np.array(reslN)
	column_titles = ['m2Sig','m2Eta','m2X','fPI', 'm2', 'c', 'lambda_sigma', 'lambda_a', 'Tc', 'Tn', 'Alpha', 'Beta', 'Vw']
	# File path to save the CSV
	file_path = f'Test_N{N}F{F}_Normal.csv'
	save_arrays_to_csv(file_path, column_titles, 
					N_Masses[:,0],N_Masses[:,1],N_Masses[:,2],N_Masses[:,3],
					N_LInputs[:,0],N_LInputs[:,1],N_LInputs[:,2],N_LInputs[:,3],
					resN[:,4],resN[:,0],resN[:,1],resN[:,2]
					)
	file_path = f'Test_N{N}F{F}_largeN.csv'
	save_arrays_to_csv(file_path, column_titles, 
					lN_Masses[:,0],lN_Masses[:,1],lN_Masses[:,2],lN_Masses[:,3],
					lN_LInputs[:,0],lN_LInputs[:,1],lN_LInputs[:,2],lN_LInputs[:,3],
					reslN[:,4],reslN[:,0],reslN[:,1],reslN[:,2]
					)

	plotDifference(reslN, resN)
	
def getTcs_ChiralPT(m2Sig,m2Eta,m2X, fPI, N, F, num=30):
	#Gets critical temperature associated with ChiralPT NOT incl. confinement.
	data = []
	for i in m2Sig:
		for j in m2Eta:
			for k in m2X:
				for l in fPI:
					if k>i:
						data.append([i,j,k,l,N,F])

	data=np.array(data)
	data = data[np.array(np.random.randint(0, high=len(data),size=num))]
	
	reslN = []; resNormal = []
	for point in data:
		try:
			#largeN Term
			detPowlN = Potential2.get_detPow(N,F,"largeN")
			VlN=Potential2.Potential(*Potential2.masses_to_lagrangian(*point,detPowlN),N,F,detPowlN,Polyakov=False)
			if num==1:
				tclN = VlN.criticalT(prnt=True)
			else:
				tclN = VlN.criticalT(prnt=False)
			
			if tclN is not None:
				orderlN = 1 if VlN.findminima(tclN) else 2
			else: orderlN = 0
			
			reslN.append([*Potential2.masses_to_lagrangian(*point, detPowlN)]+ [tclN, orderlN])
			print(f'Large N: Tc={tclN}, fPi={point[3]}, Transition Order = {orderlN}')
			
		except (Potential2.InvalidPotential):
			print('Large N term threw invalid potential.')
			tclN=None
			detPowlN = Potential2.get_detPow(N,F,"largeN")
			reslN.append([*Potential2.masses_to_lagrangian(*point, detPowlN)]+ [None, 0])
				
		try:
			#Normal Term
			detPowN=Potential2.get_detPow(N,F,"Normal")
			VNormal=Potential2.Potential(*Potential2.masses_to_lagrangian(*point,detPowN),N,F,detPowN,Polyakov=False)
			if num==1:
				tcNormal = VNormal.criticalT(prnt=True)
			else:
				tcNormal = VNormal.criticalT(prnt=False)
			
			if tcNormal is not None:
				orderNormal = 1 if VNormal.findminima(tcNormal) else 2
			else: orderNormal = 0
			

			resNormal.append([*Potential2.masses_to_lagrangian(*point,detPowN)] + [tcNormal, orderNormal])
			print(f'Normal: Tc={tcNormal}, fPi={point[3]}, Transition Order = {orderNormal}')
					
		except (Potential2.InvalidPotential):
			print('Normal term threw invalid potential.')
			tcNormal=None
			detPowN=Potential2.get_detPow(N,F,"Normal")
			resNormal.append([*Potential2.masses_to_lagrangian(*point,detPowN)] + [None, 0])

		plot=False
		if num==1: plot=True
		if tclN != None and plot:
			reslN=np.array(reslN)
			print(f'\nPotential Parameters for Large N Case:')
			print(f'Input Params')
			print(f'm2Sig={data[0,0]},m2Eta={data[0,1]},m2X={data[0,2]},fPI={data[0,3]}')
			print(f'Lagrangian Params')
			print(f"m2={reslN[0,0]},ls={reslN[0,2]},la={reslN[0,3]},c={reslN[0,1]}")
			print(f'PT Params')
			print(f'Tc={reslN[0,4]}, Transition Order {reslN[0,5]}')
			print(f"Tree Level Formula fPI={VlN.fSigma()}\n")
			
			
			sigmas = np.linspace(-5,VlN.fSigma()*1.25,num=100)
			plt.plot(sigmas,VlN.mSq['Sig'][0](sigmas,0),label='Sig')
			plt.plot(sigmas,VlN.mSq['Eta'][0](sigmas,0),label='Eta')
			plt.plot(sigmas,VlN.mSq['X'][0](sigmas,0),label='X')
			plt.plot(sigmas,VlN.mSq['Pi'][0](sigmas,0),label='Pi')

			plt.title('largeN at T=0 GeV')
			plt.legend()
			plt.show()
			

		if tcNormal != None and plot:
			resNormal=np.array(resNormal)
			
			print(f'\nPotential Parameters for Normal Case')
			print(f'Input Params')
			print(f'm2Sig={data[0,0]},m2Eta={data[0,1]},m2X={data[0,2]},fPI={data[0,3]}')
			print(f'Lagrangian Params')
			print(f'm2={resNormal[0,0]},ls={resNormal[0,2]},la={resNormal[0,3]},c={resNormal[0,1]}')
			print(f'PT Params')
			print(f'Tc={resNormal[0,4]},Transition Order {resNormal[0,5]}')
			print(f"Tree Level Formula fPI={VNormal.fSigma()}")
	
			fPiN=VNormal.fSigma()
			xs = np.linspace(-5,fPiN*1.5,num=100)
			plt.plot(xs, VNormal.V(xs)/fPiN**4-VNormal.V(0)/fPiN**4, label='$V_{{Tree}}$')
			plt.plot(xs, VNormal.V1_cutoff(xs)/fPiN**4-VNormal.V1_cutoff(0)/fPiN**4, label='$V_{{1-loop}}$')
			plt.plot(xs, VNormal.V1T(xs,tcNormal)/fPiN**4-VNormal.V1T(0,tcNormal)/fPiN**4, label='$V_{{thermal}}$ at $T=T_c$')
			plt.plot(xs, VNormal.Vtot(xs,0)/fPiN**4-VNormal.Vtot(0,0)/fPiN**4, label='$V_{{tot}}$ at $T=0$')
			plt.plot(xs, VNormal.Vtot(xs,tcNormal)/fPiN**4-VNormal.Vtot(0,tcNormal)/fPiN**4, label='$V_{{tot}}$ at $T=T_c$')
			plt.plot(xs,VNormal.VIm(xs,tcNormal)/fPiN**4-VNormal.VIm(0,tcNormal)/fPiN**4,label=f"Imaginary at $T=T_c$")
			plt.title('Normal')
			plt.xlabel(rf'$\sigma$')
			plt.ylabel(rf'$V/f_{{pi}}^4$')
			plt.legend()
			plt.show()
			
			sigmas = np.linspace(-5,VNormal.fSigma()*10,num=100)
			plt.plot(sigmas,VNormal.mSq['Sig'][0](sigmas,0),label='Sig')
			plt.plot(sigmas,VNormal.mSq['Eta'][0](sigmas,0),label='Eta')
			plt.plot(sigmas,VNormal.mSq['X'][0](sigmas,0),label='X')
			plt.plot(sigmas,VNormal.mSq['Pi'][0](sigmas,0),label='Pi')
			plt.title('Normal at T=0 GeV')
			plt.legend()
			plt.show()


	if num>1:
		data = np.array(data); reslN = np.array(reslN); resNormal = np.array(resNormal)
		save_arrays_to_csv(f'TclN_N{N}F{F}.csv', ['m2Sig','m2Eta','m2X','fPI','m2', 'c', 'lambda_sigma', 'lambda_a', 'Tc','TransitionOrder'], 
						data[:,0], data[:,1], data[:,2], data[:,3], reslN[:,0], reslN[:,1], reslN[:,2], reslN[:,3], reslN[:,4], reslN[:,5])

		save_arrays_to_csv(f'TcNormal_N{N}F{F}.csv', ['m2Sig','m2Eta','m2X','fPI','m2', 'c', 'lambda_sigma', 'lambda_a','Tc','TransitionOrder'], 
						data[:,0], data[:,1], data[:,2], data[:,3], resNormal[:,0], resNormal[:,1], resNormal[:,2], resNormal[:,3], resNormal[:,4], resNormal[:,5])

	
if __name__ == "__main__":
	print('hello')
	

	N=3; F=6

	m2Sig = np.linspace(3E2**2, 4E3**2/np.sqrt(2), num=5)
	m2Eta = np.linspace(1E2**2, 0.5E3**2, num=10)
	
	fPi = np.linspace(800, 1000, num=3)
	m2X = np.linspace(500**2, 2000**2, num=5)
	
	#getTcs_ChiralPT(m2Sig,m2Eta,m2X, fPi, N, F, num=30)
	


	#getTcs_ChiralPT(m2Sig,m2Eta,m2X, fPi, N, F, num=2)
		
	parallelScan(m2Sig,m2Eta,m2X,fPi,3,6)
	
	#m2Sig = 90000.0; m2X = 250000.0; fPI = 900.0
	m2Sig = 90000.0; m2Eta = 22500.0; m2X=	4000000.0;	fPI=850.0
	#m2Sig = 90000.0; m2Eta = 239722.22222222200; m2X=2750000.0; fPI=833.3333333333330
	#m2Sig = 90000.0; m2Eta = 239722.22222222200; m2X = 250000.0; fPI=833.3333333333330
	

	#Large N 
	#m2Eta = 8.19444444444445E-09 * fPI**4 * (F/N)**2
	lN_Linput = [*Potential2.masses_to_lagrangian(m2Sig,m2Eta,m2X,fPI,N,F,Potential2.get_detPow(N,F,"largeN"))]
	
	#NORMAL (fixed c = 8.19444444444445E-09)
	#m2Eta = 131111.11111111100
	N_Linput = [*Potential2.masses_to_lagrangian(m2Sig,m2Eta,m2X,fPI,N,F,Potential2.get_detPow(N,F,"Normal"))]

	
	print(populateN(*N_Linput, N, F, Polyakov=True,plot=True))
	print(populatelN(*lN_Linput, N, F, Polyakov=True,plot=True))
	

	#VAN DER WOUDE COMPARISON
	#m2 = -4209; ls = 16.8; la = 12.9; c = 2369; F=3; N=3
	
	#print(populateN(m2,c,ls,la, N, F, Polyakov=False,plot=True))