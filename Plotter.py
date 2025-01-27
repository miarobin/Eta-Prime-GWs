import Potential2
import GravitationalWave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import csv
from multiprocessing import Pool

'''
	Please see the dockstring in "Potential2.py" for a parameter dictionary!!
	
	This file does the following:
	1. "save_arrays_to_csv" is a generic function for saving data as a csv array.

	    save_arrays_to_csv: 
        INPUTS:  (file_path, column_titles, *arrays)
                (string, array, arrays)
        OUTPUTS: Nothing
		
	2. "populate" is a function to generate gravitational wave data from a single set of potential parameters. The input CsakiTerm can be used to
		change the potential to a Csaki-style potential. 
		Returns (0,0,0,0, (error code), 0, 0, 0, 0) if no phase transition. NB the 0's are used to avoid crashes later on.
	
		populate:
		INPUTS: (mSq, c, lambdas, lambdaa, N, F,plot=False, CsakiTerm=False)
				(float, float, float, float, int, int, bool, bool)
		OUTPUT: (Tn, alpha, betaH, vW, message, _m2Sig, _m2Eta, _m2Pi, _m2X)
				(float, float, float, float, int, float, float, float, float)
				
	3. "plotDifference" allows you to see, for the Csaki (half-transparency) and normal (full-transparency) cases, 
		the difference in GW data for cases with the same zero-temp masses. These are indicated by having the same marker. 
		However, this function can only handle up to 22 data points or it does not have enough markers.
		
		TLDR CSAKI IS HALF TRANSPARENCY; NORMAL IS FULL TRANSPARENCY. CAN ONLY HANDLE 22 DATA POINTS.
		
		plotDifference:
		INPUTS: (resC, resN)
				(np.array, np.array)
		
				
	4. "populateN" and "populateC" are wrapper functions for "populate" for the normal and Csaki case respectively.
	
		populateN/C:
		INPUTS: (mSq, c, lambdas, lambdaa, N, F,plot=False, CsakiTerm=False/True)
				(float, float, float, float, int, int, bool, bool)
		OUTPUT: (Tn, alpha, betaH, vW, message, _m2Sig, _m2Eta, _m2Pi, _m2X)
				(float, float, float, float, int, float, float, float, float)
				
	5. "parallelScan" is a function to perform a parallelised scan over a set of zero-temp particle masses:
			a Loops over each array sequentially and builds a new list of valid Lagrangian inputs. These are stored in C_data and N_data.
			b Runs populateN and populateC over the N_data and C_data arrays respectively, in parallel over, say, 8, cores.
			c Stores the run results in two csv arrays. 
			d Attempts to run plotDifference (though note that if there are more than 22 non-zero data points this will crash.
		
		parallelScan:
		INPUTS: (m2Sig, m2Eta, m2Pi, m2X, N, F)
				(np.array, np.array, np.array, np.array, int, int)
				
		
	NOTE The code at the end of the file only runs when this file is run directly. Adjust the scan ranges as necessary.
'''

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


def populate(mSq, c, lambdas, lambdaa, N, F,plot=False, CsakiTerm=False):
	
	#Building the potential...
	V = Potential2.Potential(mSq, c, lambdas, lambdaa, N, F, CsakiTerm=CsakiTerm)
	
	#Calculating the zero temperature, tree level, analytic minimum.
	fSig = V.fSigma()
	print(f'fSigma={fSig}')
	
	if plot:
		
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
		return (0, 0, 0, 0, 15)
	
	#Grid function computes:
	#	a) Nucleation temperature Tn,
	#	b) An interpolated function grd of action over temperature w/ temperature, 
	#	c) and an error code.
	Tn, grd, message = GravitationalWave.grid(V,prnt=True,plot=plot)
	
	if Tn is not None:
		#Bubbles nucleate before BBN! Yay!
		
		#Calculating wave parameters.
		alpha = abs(GravitationalWave.alpha(V,Tn)); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = GravitationalWave.wallVelocity(V, alpha, Tn)
		print(f"Tn = {Tn}, alpha = {alpha}, betaH = {betaH}")
		
		#Returning wave parameters and zero-temperature particle masses.
		return (Tn, alpha, betaH, 1, message, V.mSq['Sig'][0](fSig), V.mSq['Eta'][0](fSig), V.mSq['X'][0](fSig), V.mSq['Pi'][0](fSig))
	
	else:
		#If Tn is none, bubbles do not nucleate in time.
		print(f'CT Returned None with message {message}')
		
		#Returns the failure state, the associated failure code, and the associated zero-temperature particle masses.
		return (0, 0, 0, 0, message, V.mSq['Sig'][0](fSig), V.mSq['Eta'][0](fSig), V.mSq['X'][0](fSig), V.mSq['Pi'][0](fSig))
	

def plotDifference(resC, resN):

	#Fixed list of markers to track each set of zero-temp masses.
	markers = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","D","d","|"]
	
	colormap = plt.cm.plasma 
	#normalising the colour map to between the min and max values of _mSig
	normalize = matplotlib.colors.Normalize(vmin=min(np.ravel(resC[:,5]+resN[:,5])), vmax=max(np.ravel(resC[:,5]+resN[:,5])))
	

	for i in range(len(resC)):
		if resC[i,2]!=0 and resN[i,2]!=0:
			#Csaki term has half-transparency.
			plt.scatter(resC[i,2], resC[i,1], c = resC[i,5], alpha=1/(2), marker=markers[-1], cmap=colormap, norm=normalize)
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
	
def populateN(mSq, c, ls, la, N, F):
	#Wrapper function for normal case.
	return populate(mSq, c, ls, la, N, F, CsakiTerm=False, plot=False)
def populateC(mSq, c, ls, la, N, F):
	#Wrapper for the Csaki case.
	return populate(mSq, c, ls, la, N, F, CsakiTerm=True, plot=False)



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
		#Calculating the Lagrangian inputs. See appendix D of draft.
		inputsC = [*Potential2.masses_to_lagrangian_Csaki(*point),N,F]
		inputsN = [*Potential2.masses_to_lagrangian_Normal(*point),N,F]
		if (inputsC[0] is not None) and (inputsN[0] is not None):
			C_data.append(inputsC)
			N_data.append(inputsN)
			
	#Cropping the data for now.
	C_data = C_data[:30]
	N_data = N_data[:30]
	

        
	#Multithreading with X cores.
	with Pool(8) as p:
		#Populating the result arrays.
		resN = p.starmap(populateN, N_data)
		resC = p.starmap(populateC, C_data)
	
	#MAKE THE FILE WRITER
	#Column Titles
	N_data = np.array(N_data); C_data = np.array(C_data); resN=np.array(resN); resC=np.array(resC)
	column_titles = ['m2', 'c', 'lambda_sigma', 'lambda_a', 'Tn', 'Alpha', 'Beta', 'Vw','m2Sig','m2Eta','m2Pi','m2X']
	# File path to save the CSV
	file_path = f'Test_N{N}F{F}_Normal.csv'
	save_arrays_to_csv(file_path, column_titles, N_data[:,0], N_data[:,1], N_data[:,2], N_data[:,3], 
												resN[:,0], resN[:,1], resN[:,2], resN[:,3], resN[:,4], resN[:,6], resN[:,7], resN[:,8])
	file_path = f'Test_N{N}F{F}_Csaki.csv'
	save_arrays_to_csv(file_path, column_titles, C_data[:,0], C_data[:,1], C_data[:,2], C_data[:,3], 
												resC[:,0], resC[:,1], resC[:,2], resC[:,3], resC[:,4], resC[:,6], resC[:,7], resC[:,8])


	plotDifference(resC, resN)

	
if __name__ == "__main__":
	print('hello')
	
	N=3; F=6
	#See appendix D of draft for why I've chosen these!
	m2Sig = np.linspace(10E3/np.sqrt(3), 2*10E4, num=15)
	m2Eta = np.linspace(10E3, 2*10E4, num=10)
	m2X = np.linspace(8*10E2,10E4, num=10)
	m2Pi = np.linspace(8*10E2,10E4, num=10)
	
	parallelScan(m2Sig,m2Eta,m2X,m2Pi,3,6)