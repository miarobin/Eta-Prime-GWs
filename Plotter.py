import Potential2
import GravitationalWave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate, optimize
import csv
from multiprocessing import Pool

##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2

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


def S3T(V,*T):
	#Produce a log plot of 3D action against temperature.
	if not T:
		T = np.linspace(v/1.8,v/2., num=25)
		
	highTmin = optimize.minimize(lambda h:V.Vtot(h,400)/v**4,-246,method="Nelder-Mead")
	print(highTmin)
	zeroTmax = optimize.minimize(lambda h:-V.Vtot(h,0)/v**4, -246,method="Nelder-Mead")
	print(zeroTmax)
	print(f"High T Minimum: {highTmin.x[0]}; Zero T Local Max = {zeroTmax.x[0]}")
	Tn,grd,message = GravitationalWave.grid(V,prnt=True,plot=True)
	print(f"message = {message}")
	print(f"alpha = {GravitationalWave.alpha(V,Tn)}, Beta/H = {GravitationalWave.beta_over_H(V,Tn,grd)}, Wall Velocity = {GravitationalWave.wallVelocity(V,GravitationalWave.alpha(V,Tn), Tn)}")
		
	plt.plot(v/T, 140*np.ones((25)))
	plt.plot(v/T, grd(T))
	plt.xlabel(f"$v/T$")
	plt.ylabel(f"Log($S_3/T$)")
	
	return (Tn, GravitationalWave.alpha(V,Tn), GravitationalWave.beta_over_H(V,Tn,grd), GravitationalWave.wallVelocity(V,GravitationalWave.alpha(V,Tn), Tn), message)

def populate(mSq, c, lambdas, lambdaa, N, F,plot=False, CsakiTerm=False):
	#Lambda, Kappa, m^2_Sigma, Mu_Sig, Xi.
	V = Potential2.Potential(mSq, c, lambdas, lambdaa, N, F, CsakiTerm=CsakiTerm)
	fSig = V.fSigma()
	print(f'fSigma={fSig}')
	
	if plot:
		#Plots the potential as a function of temperature
		def plotV(V, Ts):
			for T in Ts:
				plt.plot(np.linspace(-5,fSig*1.2,num=300),V.Vtot(np.linspace(-5,fSig*1.2,num=300),T)-V.Vtot(0,T),label=f"T={T}")
			plt.legend()
			plt.show()
		plotV(V,[0,100,150,200,225,250,400,450,500,510,fSig])
	if fSig == None:
		return (0, 0, 0, 0, 15)
	

	Tn, grd, message = GravitationalWave.grid(V,prnt=True,plot=plot)
	
	if Tn is not None:
		alpha = abs(GravitationalWave.alpha(V,Tn)); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = GravitationalWave.wallVelocity(V, alpha, Tn)
		print(f"Tn = {Tn}, alpha = {alpha}, betaH = {betaH}")
		return (Tn, alpha, betaH, 1, message, V.mSq['Sig'][0](fSig), V.mSq['Eta'][0](fSig), V.mSq['X'][0](fSig), V.mSq['Pi'][0](fSig))
	else:
		print(f'CT Returned None with message {message}')
		return (0, 0, 0, 1, message, V.mSq['Sig'][0](fSig), V.mSq['Eta'][0](fSig), V.mSq['X'][0](fSig), V.mSq['Pi'][0](fSig))
	

def plotDifference(resC, resN):
	#So you can see exactly where the points have moved to
	markers = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","D","d","|"]
	
	colormap = plt.cm.plasma #or any other colormap
	normalize = matplotlib.colors.Normalize(vmin=min(np.ravel(resC[:,5]+resN[:,5])), vmax=max(np.ravel(resC[:,5]+resN[:,5])))
	#For the colour map

	for i in range(len(resC)):
		if resC[i,2]!=0 and resN[i,2]!=0:
			plt.scatter(resC[i,2], resC[i,1], c = resC[i,5], alpha=1/(2), marker=markers[-1], cmap=colormap, norm=normalize)
			plt.scatter(resN[i,2], resN[i,1], c = resN[i,5], alpha=1/(1), marker=markers[-1], cmap=colormap, norm=normalize)
		markers.pop()

			
	plt.xscale("log")
	plt.yscale("log")
	plt.colorbar(label=f'$m^2_\sigma$')
	plt.xlabel(f'beta/H')
	plt.ylabel(f'alpha')
	plt.show()
	
def populateN(mSq, c, lambdas, lambdaa, N, F):
	return populate(mSq, c, lambdas, lambdaa, N, F, CsakiTerm=False, plot=False)
def populateC(mSq, c, lambdas, lambdaa, N, F):
	return populate(mSq, c, lambdas, lambdaa, N, F, CsakiTerm=True, plot=False)


def parallelScan(m2Sig,m2Eta,m2Pi,m2X, N, F):
	#MAKE THE ARRAY
	
	data = []
	for i in m2Sig:
		for j in m2Eta:
			for k in m2X:
				for l in m2Pi:
					data.append([i,j,k,l,N,F])
					
	C_data = []; N_data = []
	for i in range(len(data)):
		point = data[i]
		inputsC = [*Potential2.masses_to_lagrangian_Csaki(*point),N,F]
		inputsN = [*Potential2.masses_to_lagrangian_Normal(*point),N,F]
		if (inputsC[0] is not None) and (inputsN[0] is not None):
			C_data.append(inputsC)
			N_data.append(inputsN)
			
	
	C_data = C_data[:30]
	N_data = N_data[:30]
	

        
	#Multithreading with 16 cores.
	with Pool(8) as p:
		resN = p.starmap(populateN, N_data)
		resC = p.starmap(populateC, C_data)
	
	#MAKE THE FILE WRITER
	#Column Titles
	N_data = np.array(N_data); C_data = np.array(C_data); resN=np.array(resN); resC=np.array(resC)
	column_titles = ['m2', 'c', 'lambda_sigma', 'lambda_a', 'Tn', 'Alpha', 'Beta', 'Vw','m2Sig','m2Eta','m2Pi','m2X']
	# File path to save the CSV
	file_path = f'Test_N{3}F{6}_Normal.csv'
	save_arrays_to_csv(file_path, column_titles, N_data[:,0], N_data[:,1], N_data[:,2], N_data[:,3], 
												resN[:,0], resN[:,1], resN[:,2], resN[:,3], resN[:,4], resN[:,6], resN[:,7], resN[:,8])
	file_path = f'Test_N{3}F{6}_Csaki.csv'
	save_arrays_to_csv(file_path, column_titles, C_data[:,0], C_data[:,1], C_data[:,2], C_data[:,3], 
												resC[:,0], resC[:,1], resC[:,2], resC[:,3], resC[:,4], resC[:,6], resC[:,7], resC[:,8])


	plotDifference(resC, resN)

	
if __name__ == "__main__":
	print('hello')
	
	N=3; F=6
	m2Sig = np.linspace(10E3/np.sqrt(3), 2*10E4, num=15)
	m2Eta = np.linspace(10E3, 2*10E4, num=10)
	m2X = np.linspace(8*10E2,10E4, num=10)
	m2Pi = np.linspace(8*10E2,10E4, num=10)
	
	parallelScan(m2Sig,m2Eta,m2X,m2Pi,3,6)
	
	'''

	data = []
	for i in m2Sig:
		for j in m2Eta:
			for k in m2X:
				for l in m2Pi:
					data.append([i,j,k,l])
	
	C_data = []; N_data = []
	for i in range(len(data)):
		point = data[i]
		inputsC = Potential2.masses_to_lagrangian_Csaki(*point, N, F)
		inputsN = Potential2.masses_to_lagrangian_Normal(*point, N, F)
		if (inputsC[0] is not None) and (inputsN[0] is not None):
			C_data.append(inputsC)
			N_data.append(inputsN)
			
	
	C_data = C_data[:20]
	N_data = N_data[:20]


	
	
	resC=[];resN=[]

	for i in range(len(C_data)):
		#Normal Term
		rowN = N_data[i]
		resN.append(populate(rowN[0],rowN[1],rowN[2],rowN[3]**2, N=3, F=6, plot=False, CsakiTerm=False))
		print(f'Message (normal) = {resN[-1][-1]}')
		#Csaki Term
		rowC = C_data[i]
		resC.append(populate(rowC[0],rowC[1],rowC[2],rowC[3]**2, N=3, F=6, plot=False, CsakiTerm=True))
		print(f'Message (csaki) = {resN[-1][-1]}')

	resC = np.array(resC); resN=np.array(resN)

	data = np.array(data)
	N_data = np.array(N_data); C_data = np.array(C_data)
	column_titles = ['m2', 'c', 'lambda_sigma', 'lambda_a', 'Tn', 'Alpha', 'Beta', 'MassRatio']
	# File path to save the CSV
	file_path = f'Test_N{3}F{6}_Normal.csv'
	save_arrays_to_csv(file_path, column_titles, N_data[:,0], N_data[:,1], N_data[:,2], N_data[:,3], resN[:,0], resN[:,1], resN[:,2], resN[:,3])
	file_path = f'Test_N{3}F{6}_Csaki.csv'
	save_arrays_to_csv(file_path, column_titles, C_data[:,0], C_data[:,1], C_data[:,2], C_data[:,3], resC[:,0], resC[:,1], resC[:,2], resC[:,3])


	plotDifference(resC, resN)'''


	


