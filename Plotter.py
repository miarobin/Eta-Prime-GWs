import Potential2
import GravitationalWave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import csv
from multiprocessing import Pool
import DressedMasses
import os
from debug_plot import debug_plot

import WallVelocity
import WallVelocityLargeN

if not Potential2.PLOT_RUN:
	matplotlib.use('Agg') 

# Get number of CPUs allocated by SLURM
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
CORES = 10  # default to 1 if not set
print(f"Using {CORES} cores")


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
#CORES=1


def plotV(V, Ts):
	for T in Ts:
		plt.plot(np.linspace(-5,V.fSIGMA*Potential2.SIGMULT,num=100)/V.fSIGMA,V.Vtot(np.linspace(-5,V.fSIGMA*Potential2.SIGMULT,num=100),T)/V.fSIGMA**4-V.Vtot(0,T)/V.fSIGMA**4,label=f"T={T}")


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



def populate(mSq, c, lambdas, lambdaa, N, F, detPow, Polyakov=False, xi=1, plot=True, fSIGMA=None):
	#Building the potential...
	
	try:
		V = Potential2.Potential(mSq, c, lambdas, lambdaa, N, F, detPow, Polyakov=Polyakov, xi=xi, fSIGMA=fSIGMA)
	except Potential2.InvalidPotential as e:
		print(e)
		return (0, 0, 0, 0, 16, 0, 0, 0, 0) #Dressed mass calculation has failed for this.
	except Potential2.BadDressedMassConvergence as e:
		return (0, 0, 0, 0, 23, 0, 0, 0, 0)
	
	
	#Calculating the zero temperature, tree level, analytic minimum.
	fSig = V.fSIGMA
	print(f'fSigma={fSig}')
	print(f'Masses: m2_sig={V.mSq['Sig'][0](fSig)}, m2Eta={V.mSq['Eta'][0](fSig)}, m2X={V.mSq['X'][0](fSig)}, m2Pi={V.mSq['Pi'][0](fSig)}')


	if plot:
		#Plotting the interpolated dressed masses
		DressedMasses.plotInterpMasses(V)
		#Plots the potential as a function of temperature
		def plotV(V, Ts):
			plt.figure()
			for T in Ts:
				plt.plot(np.linspace(-5,fSig*1.2,num=300),V.Vtot(np.linspace(-5,fSig*1.2,num=300),T)-V.Vtot(0,T),label=f"T={T}")
			plt.legend()
			debug_plot(name="debug", overwrite=False)
			#plt.show()
			
		#Do feel free to change this list of temperatures to something more sensible.
		plotV(V,[0,100,150,200,225,250,400,450,500,510,fSig])
		
	if fSig == None:
		#If fSig does not exist, then the potential does not have enough solutions for a tunneling. Return None.
		return (0, 0, 0, 0, 15, 0, 0, 0, 0)

	#Grid function computes:
	#   a) Nucleation temperature Tn,
	#   b) An interpolated function grd of action over temperature w/ temperature,
	#   c) and an error code.
	Tn, grd, tc, message = GravitationalWave.grid(V,prnt=True,plot=plot,ext_minT=V.minT)
	


	if Tn is not None:
		#I'm not even sure how this is an error but anyway:
		if Tn<tc/10:
			return (0, 0, 0, tc, 18, 0, 0, 0, 0)
		
		#Bubbles nucleate before BBN! Yay!
		
		#Calculating wave parameters.
		alpha = abs(GravitationalWave.alpha(V,Tn)); betaH = GravitationalWave.beta_over_H(V,Tn,grd)
		print(f"Tn = {Tn}, alpha = {alpha}, betaH = {betaH}, message = {message}")
		
		#Wall Velocity 2303.10171:
		minima = V.findminima(Tn)
		psiN = V.dVdT(minima,Tn)/V.dVdT(0,Tn)
            
		cs2 = V.dVdT(0,Tn)/(Tn*V.d2VdT2(0,Tn))
		cb2 = V.dVdT(minima,Tn)/(Tn*V.d2VdT2(minima,Tn))
		alN = WallVelocity.alpha(V, Tn, cb2)
            
		vwLTE = WallVelocity.find_vw(alN,cb2,cs2, psiN)
		kappaLTE = WallVelocity.find_kappa(alN, cb2, cs2, psiN, vw=vwLTE)
		

		#Wall Velocity 2312.09964. Large N refers to number of degrees of freedom here!
		#NOTE for this to be valid, DoFBroken << DoFSym.
		DoFSym = (7/2*V.F*V.N) + 2*(V.N**2-1) + Potential2._g_starSM(Tn)
		DoFBroken = 2*V.F**2 + Potential2._g_starSM(Tn)
		
		if DoFBroken<DoFSym:#Maybe make harsher! DoFBroken needs to be negligible 
			alNLN = WallVelocityLargeN.find_alphaN(tc, Tn, cb2, DoFSym)
			vwLN = WallVelocityLargeN.find_vw(alNLN,cb2,cs2)
			kappaLN = WallVelocityLargeN.find_kappa(alNLN, cb2, cs2, psiN, vw=vwLN)
		else:
			vwLN = None
			kappaLN = None
			
		print(f"vwLTE = {vwLTE}, kappaLTE = {kappaLTE}, vwLN = {vwLN}, kappaLN = {kappaLN}")
		
		#Returning wave parameters and zero-temperature particle masses.
		return (Tn, alpha, betaH, tc, message, vwLTE, kappaLTE, vwLN, kappaLN)

	else:
		#If Tn is none, bubbles do not nucleate in time.
		print(f'CT Returned None with message {message}')
		
		#Returns the failure state, the associated failure code, and the associated zero-temperature particle masses.
		return (0, 0, 0, tc, message, 0, 0, 0, 0)

# --- safe wrapper around your existing populate() ---
def populate_safe(*args, **kwargs):
    """
    Calls the original populate(), but converts all outputs
    to plain Python objects (numbers or lists of numbers) for multiprocessing.
    """
    raw_results = populate(*args, **kwargs)
    
    safe_results = []
    for r in raw_results:
        if isinstance(r, (int, float)):
            safe_results.append(r)
        elif hasattr(r, "__iter__"):
            safe_results.append([float(x) for x in r])
        else:
            # fallback for unpicklable objects
            safe_results.append(0.0)
    
    return safe_results

	
	
	
def populateN(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=False, xi=1, plot=False):
	#Wrapper function for normal case.
	detPow = Potential2.get_detPow(N,F,"Normal")
	
	try:
		mSq, c, ls, la = Potential2.masses_to_lagrangian(m2Sig,m2Eta,m2X,fPI,N,F,detPow)
	except Potential2.NonUnitary as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 0, 20)
	except Potential2.NonTunnelling as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 0, 21)
	except Potential2.BoundedFromBelow as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 0, 22)

	
	print(f'Normal: m2={mSq},c={c},ls={ls},la={la},N={N},F={F},p={detPow}')
	return [mSq, c, ls, la, *populate_safe(mSq, c, ls, la, N, F, detPow, Polyakov=Polyakov, xi=xi, plot=plot, fSIGMA=fPI)]


def populatelN(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=True, xi=1, plot=False):
	#Wrapper for the largeN case.
	detPow = Potential2.get_detPow(N,F,"largeN")
	
	try:
		mSq, c, ls, la = Potential2.masses_to_lagrangian(m2Sig,m2Eta,m2X,fPI,N,F,detPow)
	except Potential2.NonUnitary as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 0, 20)
	except Potential2.NonTunnelling as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 0, 21)
	except Potential2.BoundedFromBelow as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 0, 22)
	


	print(f'largeN: m2={mSq},c={c},ls={ls},la={la},N={N},F={F},p={detPow}')
	#return [mSq, c, ls, la, *populate(mSq, c, ls, la, N, F, detPow, Polyakov=Polyakov, plot=plot, fSIGMA=fPI)]
	return [mSq, c, ls, la, *populate_safe(mSq, c, ls, la, N, F, detPow, Polyakov=Polyakov, plot=plot, fSIGMA=fPI)]


def populate_safe_wrapperN(*args):
    try:
        out = populateN(*args)
        return [float(x) if isinstance(x, (int, float, np.floating)) else 0.0 for x in np.ravel(out)]
    except Exception as e:
        print(f"populateN failed: {e}")
        return [0.0]*10  # same number of outputs you expect
        

def populate_safe_wrapperlN(*args):
    try:
        out = populatelN(*args)
        return [float(x) if isinstance(x, (int, float, np.floating)) else 0.0 for x in np.ravel(out)]
    except Exception as e:
        print(f"populatelN failed: {e}")
        return [0.0]*10


def parallelScan(m2Sig,m2Eta,m2X, fPI, N, F, crop=3):
	
	#MAKE THE ARRAY
	data = []
	for i in m2Sig:
		for j in m2Eta:
			for k in m2X:
				for l in fPI:
					if k>j and k>i:
						data.append([i,j,k,l,N,F])
						
	#Cropping the data.
	data = np.array(data)
	data = data[:crop]
        
	#Multithreading with X cores.
	with Pool(CORES) as p:
		#Populating the result arrays.
		resN = p.starmap(populate_safe_wrapperN, data)
		reslN = p.starmap(populate_safe_wrapperlN, data)


	resN=np.array(resN); reslN=np.array(reslN)


	#MAKE THE FILE WRITER
	#Column Titles
	column_titles = ['m2Sig','m2Eta','m2X','fPI', 'm2', 'c', 'lambda_sigma', 'lambda_a', 'Tc', 'Tn', 'Alpha', 'Beta', 'Message', 'vwLTE', 'kappaLTE', 'vwLN', 'kappaLN']
	# File path to save the CSV
	file_path = f'Test_N{N}F{F}_Normal.csv'
	save_arrays_to_csv(file_path, column_titles, 
					data[:,0],data[:,1],data[:,2],data[:,3],
					resN[:,0],resN[:,1],resN[:,2],resN[:,3],
					resN[:,7],resN[:,4],resN[:,5],resN[:,6],resN[:,8],
					resN[:,9], resN[:,10], resN[:,11], resN[:,12]
					)
	file_path = f'Test_N{N}F{F}_largeN.csv'
	save_arrays_to_csv(file_path, column_titles, 
					data[:,0],data[:,1],data[:,2],data[:,3],
					reslN[:,0],reslN[:,1],reslN[:,2],reslN[:,3],
					reslN[:,7],reslN[:,4],reslN[:,5],reslN[:,6],reslN[:,8],
					reslN[:,9], reslN[:,10], reslN[:,11], reslN[:,12]
					)

	print('Scan Finished')

def parallelScanNorm(m2Sig,m2Eta,m2X, fPI, N, F, crop=None):
    
	#MAKE THE ARRAY
	data = []
	for i in m2Sig:
		for j in m2Eta:
			for k in m2X:
				for l in fPI:
					data.append([i,j,k,l,N,F])
						
	#Cropping the data.
	data = np.array(data)
	if crop and crop<len(data):
		data = data[:crop]
		
	#Multithreading with X cores.
	with Pool(CORES) as p:
		#Populating the result arrays.
		#resN = p.starmap(populateN, data)
		resN = p.starmap(populate_safe_wrapperN, data)
  
	resN=np.array(resN)

	#MAKE THE FILE WRITER
	#Column Titles
	column_titles = ['m2Sig','m2Eta','m2X','fPI', 'm2', 'c', 'lambda_sigma', 'lambda_a', 'Tc', 'Tn', 'Alpha', 'Beta', 'Message', 'vwLTE', 'kappaLTE', 'vwLN', 'kappaLN']
	# File path to save the CSV
	file_path = f'NoPoly_N{N}F{F}_Normal.csv'
	save_arrays_to_csv(file_path, column_titles, 
					data[:,0],data[:,1],data[:,2],data[:,3],
					resN[:,0],resN[:,1],resN[:,2],resN[:,3],
					resN[:,7],resN[:,4],resN[:,5],resN[:,6],resN[:,8],
					resN[:,9], resN[:,10], resN[:,11], resN[:,12]
					)
					

	print('Scan Finished')
 
	
if __name__ == "__main__":

	###LARGE SCAN###s
	N=3; F=3

	m2Sig = np.linspace(1., 25., num=3)*1000**2
	m2Eta = np.linspace(1, 25., num=3)*1000**2
	m2X = np.linspace(1., 25., num=3)*1000**2
 
	fPi = np.linspace(0.5,1.5,num=3)*1000*np.sqrt(F/2)
	
	#comment out parallelscan norm to plot
	Potential2.PLOTRUN=False
	#parallelScanNorm(m2Sig,m2Eta,m2X,fPi,N,F)
	
	
	###SINGLE POINT FROM SCAN###
	N=3; F=3
	Potential2.PLOTRUN=True
	POINT_OF_INTEREST=12

	filename = 'Test_N3F3_Normal100hou1.csv'; delimiter = ','
	data = np.array(np.genfromtxt(filename, delimiter=delimiter, skip_header=1, dtype=None))

	m2Sig, m2Eta, m2X, fPI, m2, c, ls, la, Tc, Tn, alpha, beta,_ = data[POINT_OF_INTEREST-2]

	print(f'm2Sig = {m2Sig}, m2Eta = {m2Eta}, m2X = {m2X}, fPI = {fPI}')
	print(f'm2 = {m2}, c = {c}, ls = {ls}, la = {la}')
	print(f'Tc = {Tc}, Tn = {Tn}, alpha = {alpha}, beta = {beta}')

	populateN(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=True, xi=2, plot=True)

