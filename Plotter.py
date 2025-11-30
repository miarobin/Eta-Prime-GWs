import config
import Potential2
import GravitationalWave
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import csv
from multiprocessing import Pool
import DressedMasses
import os
from config_debug_plot import debug_plot
from functools import partial
import WallVelocity
import WallVelocityLargeN
from datetime import datetime


# Get number of CPUs allocated by SLURM
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
CORES = 9  # default to 1 if not set

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


def unwrap_populate(args, N, F, Polyakov, xi, detType):	
	return populate_safe_wrapper(*args, N, F, Polyakov, xi, detType)


def plotV(V, Ts):
	for T in Ts:
		plt.plot(np.linspace(-5,V.fSIGMA*config.SIGMULT,num=100)/V.fSIGMA,V.Vtot(np.linspace(-5,V.fSIGMA*config.SIGMULT,num=100),T)/V.fSIGMA**4-V.Vtot(0,T)/V.fSIGMA**4,label=f"T={T}")


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
	plot = False #force plot false for scan
 
	try:
		V = Potential2.Potential(mSq, c, lambdas, lambdaa, N, F, detPow, Polyakov=Polyakov, xi=xi, fSIGMA=fSIGMA)
	except Potential2.InvalidPotential as e:
		return (0, 0, 0, 0, 16., 0, 0, 0, 0) 
	except Potential2.BadDressedMassConvergence as e:
		return (0, 0, 0, 0, 23., 0, 0, 0, 0)
	
	
	
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
			
		#List of temeperatures to plot
		plotV(V,[0,100,150,200,225,250,400,450,500,510,fSig])
		
	if fSig == None:
		#If fSig does not exist, then the potential does not have enough solutions for a tunneling. Return None.
		return (0, 0, 0, 0, 15., 0, 0, 0, 0)

	#Grid function computes:
	#   a) Nucleation temperature Tn,
	#   b) An interpolated function grd of action over temperature w/ temperature,
	#   c) and an error code.
	Tn, grd, tc, message = GravitationalWave.grid(V,ext_minT=V.minT)
	V.Tn = Tn
	


	if Tn is not None:
		#Fixing box edges (Nan Values)
		if Tn<tc/10:
			return (0, 0, 0, tc, 18., 0, 0, 0, 0)
					
		#Calculating wave parameters.
		alpha = abs(GravitationalWave.alpha(V,Tn)); betaH = GravitationalWave.beta_over_H(V,Tn,grd)
		print(f"Tn = {Tn}, alpha = {alpha}, betaH = {betaH}, message = {message}")
		
		#Wall Velocity 2303.10171:
		minima = V.findminima(Tn)
		psiN = V.dVdT(minima,Tn)/V.dVdT(0,Tn)
            
		cs2 = V.dVdT(0,Tn)/(Tn*V.d2VdT2(0,Tn))
		cb2 = V.dVdT(minima,Tn)/(Tn*V.d2VdT2(minima,Tn))

		print(f'cs2={cs2}, cb2={cb2}')

		if 0.1<cb2<2/3 and 0.1<cs2<2/3 and 0.5<psiN<0.99:
			alN = WallVelocity.alpha(V, Tn, cb2)
			print(f'alN={alN}, psiN={psiN}')
			
			vwLTE = WallVelocity.find_vw(alN,cb2,cs2, psiN)
			kappaLTE = WallVelocity.find_kappa(alN, cb2, cs2, psiN, vw=vwLTE)			

			#Wall Velocity 2312.09964. Large N refers to number of degrees of freedom here!
			#NOTE for this to be valid, DoFBroken << DoFSym.
   
			DoFSym = (7/2*V.F*V.N) + 2*(V.N**2-1) + Potential2._g_starSM(Tn)
			DoFBroken = 2*V.F**2 + Potential2._g_starSM(Tn)
			
			if DoFBroken<DoFSym:
				alNLN = WallVelocityLargeN.find_alphaN(tc, Tn, cb2, DoFSym)
				print(f'alNLN={alNLN}')
				vwLN = WallVelocityLargeN.find_vw(alNLN,cb2,cs2)
				kappaLN = WallVelocityLargeN.find_kappa(alNLN, cb2, cs2, psiN, vw=vwLN)
			else:
				vwLN = None
				kappaLN = None
		else:
			vwLTE=None
			kappaLTE=None
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

# safe wrapper around populate() 
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


def populateWrapper(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov, xi, detType, plot=False):
	#Wrapper function for normal case.
	detPow = Potential2.get_detPow(N,F,detType)
	
	try:
		mSq, c, ls, la = Potential2.masses_to_lagrangian(m2Sig,m2Eta,m2X,fPI,N,F,detPow)
	except Potential2.NonUnitary as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 20., 0, 0, 0, 0)
	except Potential2.NonTunnelling as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 21., 0, 0, 0, 0)
	except Potential2.BoundedFromBelow as e:
		return (0, 0, 0, 0, 0, 0, 0, 0, 22., 0, 0, 0, 0)

	
	print(f'{detType}: m2={mSq},c={c},ls={ls},la={la},N={N},F={F},p={detPow}')
	return [mSq, c, ls, la, *populate_safe(mSq, c, ls, la, N, F, detPow, Polyakov=Polyakov, xi=xi, plot=plot, fSIGMA=fPI)]



def populate_safe_wrapper(*args):
    try:
        out = populateWrapper(*args)
        return [float(x) if isinstance(x, (int, float, np.floating)) else 0.0 for x in np.ravel(out)]
    except Exception as e:
        print(f"populate failed: {e}")
        return [0.0]*13  # same number of outputs you expect
        


#Make sure to delete the old file before running (otherwise will overwritte)
def parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=False, xi=1, detType='Normal', crop=None, filename=None):
    
    if filename is None:
        today = datetime.today()
        day = today.day
        hour = today.hour

        if Polyakov:
            filename = f'F{F}/N{N}/N{N}F{F}xi{xi}_{detType}_{day}Nov{hour}hr.csv'
        if not Polyakov and detType == 'Normal':
            filename = f'F{F}/N{N}F{F}_{detType}_{day}Nov{hour}hr.csv'
        elif not Polyakov:
            filename = f'F{F}/N{N}/N{N}F{F}_{detType}_{day}Nov{hour}hr.csv'

    # Build full parameter list
    data = []
    for i in m2Sig:
        for j in m2Eta:
            for k in m2X:
                for l in fPI:
                    data.append([i, j, k, l])
    data = np.array(data)

    if crop and crop < len(data):
        data = data[:crop]

    # Check if checkpoint file exists
    done = set()
    if os.path.exists(filename):
        print(f"[Checkpoint] Resuming from {filename}")
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if row:
                    done.add(tuple(map(float, row[:4])))

    else:
        print(f"[Checkpoint] Creating new file: {filename}")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['m2Sig','m2Eta','m2X','fPI','m2','c','lambda_sigma',
                             'lambda_a','Tc','Tn','Alpha','Beta','Message',
                             'vwLTE', 'kappaLTE', 'vwLN', 'kappaLN'])

    # Filter out completed points
    todo = [params for params in data if tuple(params[:4]) not in done]
    print(f"[Checkpoint] Total points={len(data)}, remaining={len(todo)}")

    if not todo:
        print("[Checkpoint] All points already completed.")
        return

    unwrap_populate_static = partial(unwrap_populate, N=N, F=F, Polyakov=Polyakov, xi=xi, detType=detType)

    # Run in parallel and write results one-by-one
    with Pool(CORES) as p, open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for params, result in zip(todo, p.imap(unwrap_populate_static, todo)):
            writer.writerow(list(params[:4]) +  [ result[0],  result[1],  result[2],  result[3], result[7], 
												result[4], result[5], result[6],  result[8], 
                                                result[9], result[10], result[11], result[12] ])
            f.flush()
            print(f"[Saved] {params[:4]} → result saved.")


    print("Scan Finished")
    


def parallelScan_refill(N, F, Polyakov, xi, detType, day, hour):
    
	
    if Polyakov:
        filename = f'F{F}/N{N}/N{N}F{F}xi{xi}_{detType}_{day}Nov{hour}hr.csv'
        refill_filename = f'F{F}/N{N}/N{N}F{F}xi{xi}_{detType}_{day}Nov{hour}hr_refill.csv'
        new_filename = f'F{F}/N{N}/N{N}F{F}xi{xi}_{detType}_{day}Nov{hour}hr_toppedup.csv'
    if not Polyakov and detType == 'Normal':
        filename = f'F{F}/F{F}_{detType}_{day}Nov{hour}hr.csv'
        refill_filename = f'F{F}/F{F}_{detType}_{day}Nov{hour}hr_refill.csv'
        new_filename = f'F{F}/F{F}_{detType}_{day}Nov{hour}hr_toppedup.csv'
    elif not Polyakov:
        filename = f'F{F}/N{N}/N{N}F{F}_{detType}_{day}Nov{hour}hr.csv'
        refill_filename = f'F{F}/N{N}/N{N}F{F}_{detType}_{day}Nov{hour}hr_refill.csv'
        new_filename = f'F{F}/N{N}/N{N}F{F}_{detType}_{day}Nov{hour}hr_toppedup.csv'
        
    if Polyakov:
        filename = f'N{N}F{F}xi{xi}_{detType}_{day}Nov{hour}hr.csv'
        refill_filename = f'N{N}F{F}xi{xi}_{detType}_{day}Nov{hour}hr_refill.csv'
        new_filename = f'N{N}F{F}xi{xi}_{detType}_{day}Nov{hour}hr_toppedup.csv'
    if not Polyakov and detType == 'Normal':
        filename = f'N{N}F{F}_{detType}_{day}Nov{hour}hr.csv'
        refill_filename = f'N{N}F{F}_{detType}_{day}Nov{hour}hr_refill.csv'
        new_filename = f'N{N}F{F}_{detType}_{day}Nov{hour}hr_toppedup.csv'
    elif not Polyakov:
        filename = f'N{N}F{F}_{detType}_{day}Nov{hour}hr.csv'
        refill_filename = f'N{N}F{F}_{detType}_{day}Nov{hour}hr_refill.csv'
        new_filename = f'N{N}F{F}_{detType}_{day}Nov{hour}hr_toppedup.csv'
    
	#OPERATIONAL TEST
    #filename = 'RefillTestArray_F6N3_AMSB.csv'
    #refill_filename = 'RefillTestArray_refill.csv'
    #new_filename = 'RefillTestArray_toppedup.csv'

	
    # Check if checkpoint file exists
    todo = set()
    if os.path.exists(filename):
        print(f"[Checkpoint] Resuming from {filename}")
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if row:
                    if float(row[9])>0:
                        todo.add(tuple(map(float, row[:4])))
    else:
        print(f'File with name {filename} does not exist')

    # Filter out completed points
    print(f"[Checkpoint] Remaining={len(todo)}")

    if not todo:
        print("[Checkpoint] All points already correct.")
        return

    with open(refill_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['m2Sig','m2Eta','m2X','fPI','m2','c','lambda_sigma',
                             'lambda_a','Tc','Tn','Alpha','Beta','Message',
                             'vwLTE', 'kappaLTE', 'vwLN', 'kappaLN'])

	#Fills in static arguments for populate.
    unwrap_populate_static = partial(unwrap_populate, N=N, F=F, Polyakov=Polyakov, xi=xi, detType=detType)

    # Run in parallel and write results one-by-one
    with Pool(CORES) as p, open(refill_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for params, result in zip(todo, p.imap(unwrap_populate_static, todo)):
            print(f'Summary: m2Sig={params[0]}, m2Eta={params[1]}, m2X={params[2]}, fPI={params[3]}')
            print(f'm2 = {result[0]}, c={result[1]}, ls={result[2]}, la={result[3]}')
            print(f'Tc={result[7]}, Tn={result[4]}, alpha={result[5]}, beta/H={result[6]}, message={result[8]}')
            print(f'vwLTE={result[9]}, kappaLTE={result[10]}, vwLN={result[11]}, kappaLN={result[12]}')
            
            writer.writerow(list(params[:4]) +  [ result[0],  result[1],  result[2],  result[3], result[7], 
												result[4], result[5], result[6],  result[8], 
                                                result[9], result[10], result[11], result[12] ])
            f.flush()
            print(f"[Saved] {params[:4]} → result saved.")
            
    print("Scan Finished")
    
    #Now refill the original dataset.
    refill(filename,refill_filename,new_filename)
         
    
def refill(original_filename, refill_filename, new_filename):
	data = np.array(np.genfromtxt(original_filename, delimiter=',', skip_header=1, dtype=float))
	refill_data = np.array(np.genfromtxt(refill_filename, delimiter=',', skip_header=1, dtype=float))

	for row in refill_data:
		mask = np.all(np.isclose(data[:, :4], row[:4]),axis=1)
		data[mask] = row

	# Save the array to a CSV file
	with open(new_filename, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['m2Sig','m2Eta','m2X','fPI','m2','c','lambda_sigma',
							'lambda_a','Tc','Tn','Alpha','Beta','Message',
							'vwLTE', 'kappaLTE', 'vwLN', 'kappaLN'])

	# Run in parallel and write results one-by-one
	with open(new_filename, 'a', newline='') as f:
		writer = csv.writer(f)
		for row in data:
			writer.writerow(list(row))
			f.flush()


	return data


if __name__ == "__main__":
    
    #LARGE SCANS
    N=4; F=6; detType='Normal'; 
    num=6

    detPow = Potential2.get_detPow(N,F,detType)
    m2Sig = np.linspace(1., 10., num=num)*1000**2

    if F*detPow>4:
        maxm2Eta = (16*np.pi/3) * 1.5**2 * (F*detPow)**3 / (16*(4*np.pi)**(F*detPow-4) * 25) 
        minm2Eta = maxm2Eta/25 #Arbitrary.
        m2Eta = np.linspace(minm2Eta, maxm2Eta, num=num)*1000**2 
    else:
        m2Eta = np.linspace(1., 25., num=num)*1000**2
    m2X = np.linspace(1., 25., num=num)*1000**2
    fPi = np.linspace(0.5,1.5,num=num)*1000*np.sqrt(F*detPow/2)


    parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPi, N, F, detType=detType, Polyakov=False,xi=1)

    #parallelScan_refill(N, F, False, 1, 'Normal', 13, 0)
    Potential2.PRNT_RUN=False
	
    '''
    filename = 'F6/N4/N4F6xi1_AMSB_13Nov.csv'; delimiter = ','

    N=4; F=6; detType='Normal'; 
    num=3
    
    detPow = Potential2.get_detPow(N,F,detType)

	#LARGE SCANS 
	
	N=3; F=3; detType='Normal'; 
	num=6

	detPow = Potential2.get_detPow(N,F,detType)

	#m2Sig = np.linspace(1., 10., num=num)*1000**2
	m2Sig = np.array([1.])*1000**2
	if F*detPow>4:
		maxm2Eta = (16*np.pi/3) * 1.5**2 * (F*detPow)**3 / (16*(4*np.pi)**(F*detPow-4) * 25) 
		minm2Eta = maxm2Eta/25 #Arbitrary.
		m2Eta = np.linspace(minm2Eta, maxm2Eta, num=num)*1000**2 
	else:
		m2Eta = np.linspace(1., 25., num=num)*1000**2
	m2X = np.linspace(1., 25., num=num)*1000**2
	fPi = np.linspace(0.5,1.5,num=num)*1000*np.sqrt(F*detPow/2)#CHANGE

	parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPi, N, F, detType=detType, Polyakov=True,xi=5)
	
	#REFILL
	N=3; F=3; detType='Normal'; Day=19; Hour=23; Polyakov=True; xi=1

	parallelScan_refill(N, F, Polyakov, xi, detType, Day, Hour)

	# SINGLE POINT FROM SCAN
	POINT_OF_INTEREST=13
	parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPi, N, F, detType=detType, Polyakov=False,xi=1)

	#REFILL
	#N=3; F=3; detType='Normal'; Day=19; Hour=23; Polyakov=True; xi=1
	#parallelScan_refill(N, F, Polyakov, xi, detType, Day, Hour)
 
	Potential2.PRNT_RUN=False
	
	'''
	#REFILL TEST (You have to go into the function to manually change the test filenames)
    #config.PRNT_RUN=True
    #parallelScan_refill(N, F, False, None, 'AMSB', None, None)
    
    #original_filename = 'RefillTestArray_F6N3_AMSB.csv'
    #refill_filename = 'RefillTestArray_refill.csv'
    #new_filename = 'RefillTestArray_toppedup.csv'
    #refill(original_filename, refill_filename, new_filename) 
    
    filename = 'F6/N4/N4F6xi1_AMSB_13Nov.csv'; delimiter = ','
    data = np.array(np.genfromtxt(filename, delimiter=delimiter, skip_header=1, dtype=None))

	#comment out parallelscan norm to plot
    N=4; F=4; detType = 'AMSB'; 
    num=6
    detPow = Potential2.get_detPow(N,F,detType)  
    parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPi, N, F, detType=detType, Polyakov=True,xi=5)
    print(populateWrapper(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=True, xi=1, detType='AMSB', plot=True))
	
    '''
	# SINGLE POINT FROM SCAN
	POINT_OF_INTEREST=10

	N=3; F=3; detType='Normal'; Polyakov=True; xi=1


	filename = 'F3/N3/N3F3xi1_Normal_24Nov19hr.csv'; delimiter = ','
	data = np.array(np.genfromtxt(filename, delimiter=delimiter, skip_header=1, dtype=None))

	m2Sig, m2Eta, m2X, fPI, m2, c, ls, la, Tc, Tn, alpha, beta,message,vwLTE,kappaLTE,vwLN,kappaLN = data[POINT_OF_INTEREST-2]

	print(f'm2Sig = {m2Sig}, m2Eta = {m2Eta}, m2X = {m2X}, fPI = {fPI}')
	print(f'm2 = {m2}, c = {c}, ls = {ls}, la = {la}')
	print(f'Tc = {Tc}, Tn = {Tn}, alpha = {alpha}, beta = {beta}')
	print(populateWrapper(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=Polyakov, xi=xi, detType=detType, plot=True))
	'''
