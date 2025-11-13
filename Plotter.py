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
from functools import partial
import cProfile
<<<<<<< HEAD
import pstats
=======

>>>>>>> upstream/main
import WallVelocity
import WallVelocityLargeN

matplotlib.use('Agg') 

# Get number of CPUs allocated by SLURM
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
<<<<<<< HEAD
CORES = 36  # default to 1 if not set
=======
CORES = 8  # default to 1 if not set
>>>>>>> upstream/main
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
		
	The code at the end of the file only runs when this file is run directly. Adjust the scan ranges as necessary.
'''


def unwrap_populate(args, N, F, Polyakov, xi, detType):	
	return populate_safe_wrapper(*args, N, F, Polyakov, xi, detType)


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
	plot = False #force plot false for scan
 
	try:
		V = Potential2.Potential(mSq, c, lambdas, lambdaa, N, F, detPow, Polyakov=Polyakov, xi=xi, fSIGMA=fSIGMA)
	except Potential2.InvalidPotential as e:
		print(e)
		return (0, 0, 0, 0, 16., 0, 0, 0, 0) #Dressed mass calculation has failed for this.
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
			#plt.show()
			
		#Do feel free to change this list of temperatures to something more sensible.
		plotV(V,[0,100,150,200,225,250,400,450,500,510,fSig])
		
	if fSig == None:
		#If fSig does not exist, then the potential does not have enough solutions for a tunneling. Return None.
		return (0, 0, 0, 0, 15., 0, 0, 0, 0)

	#Grid function computes:
	#   a) Nucleation temperature Tn,
	#   b) An interpolated function grd of action over temperature w/ temperature,
	#   c) and an error code.
	Tn, grd, tc, message = GravitationalWave.grid(V,prnt=True,plot=plot,ext_minT=V.minT)
	


	if Tn is not None:
		#I'm not even sure how this is an error but anyway:
		if Tn<tc/10:
			return (0, 0, 0, tc, 18., 0, 0, 0, 0)
					
		
		#Bubbles nucleate before BBN! Yay!
		
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
			
			if DoFBroken<DoFSym:#Maybe make harsher! DoFBroken needs to be negligible 
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
		print(e)
		return (0, 0, 0, 0, 0, 0, 0, 0, 20., 0, 0, 0, 0)
	except Potential2.NonTunnelling as e:
		print(e)
		return (0, 0, 0, 0, 0, 0, 0, 0, 21., 0, 0, 0, 0)
	except Potential2.BoundedFromBelow as e:
		print(e)
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
        

<<<<<<< HEAD
def populate_safe_wrapperlN(*args):
    try:
        out = populatelN(*args)
        return [float(x) if isinstance(x, (int, float, np.floating)) else 0.0 for x in np.ravel(out)]
    except Exception as e:
        print(f"populatelN failed: {e}")
        return [0.0]*13
 
 
#Just make sure to delete the old file before running   
def parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=False, xi=1, crop=None):
   
    data = []
    for i in m2Sig:
        for j in m2Eta:
            for k in m2X:
                for l in fPI:
                    if k > j and k > i:  
                        data.append([i, j, k, l, N, F])
    data = np.array(data)

    if crop and crop < len(data):
        data = data[:crop]

    normal_file = f"Test_N{N}F{F}_Normal.csv"
    largeN_file = f"Test_N{N}F{F}_largeN.csv"

    def init_csv(fname):
        if not os.path.exists(fname):
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['m2Sig','m2Eta','m2X','fPI','m2','c','lambda_sigma',
                             'lambda_a','Tc','Tn','Alpha','Beta','Message',
                             'vwLTE', 'kappaLTE', 'vwLN', 'kappaLN'])

    init_csv(normal_file)
    init_csv(largeN_file)

    #  Load finished parameter points to skip them
    def load_done(fname):
        done = set()
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if row:
                        done.add(tuple(map(float, row[:4])))
        return done

    done_normal = load_done(normal_file)
    done_largeN = load_done(largeN_file)

    # Only compute points not already done in BOTH files
    todo = [pt for pt in data if (tuple(pt[:4]) not in done_normal) or (tuple(pt[:4]) not in done_largeN)]

    print(f"[Checkpoint] Total points = {len(data)}, still to do = {len(todo)}")

    if not todo:
        print("[Checkpoint] All points are already done.")
        return

    #  Run in parallel, but save as soon as each result comes back
    with Pool(CORES) as p, \
        open(normal_file, 'a', newline='') as fn, \
        open(largeN_file, 'a', newline='') as fl:

        writer_norm = csv.writer(fn)
        writer_lN   = csv.writer(fl)

        for params, results in zip(
            todo,
            p.starmap(
                # This calls both wrappers on each point
                lambda m2Sig, m2Eta, m2X, fPI, N, F: (
                    populate_safe_wrapperN(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=Polyakov, xi=xi),
                    populate_safe_wrapperlN(m2Sig, m2Eta, m2X, fPI, N, F,  Polyakov=Polyakov, xi=xi)
                ),
                [tuple(pt) for pt in todo]
            )
        ):
            resN, reslN = results

            #  Save Normal 

            writer_norm.writerow(list(params[:4]) +  [ resN[0], resN[1], resN[2], resN[3], resN[7],
												resN[4], resN[5], resN[6],  resN[8], 
                                                resN[9], resN[10], resN[11], resN[12] ])
            fn.flush()

            #  Save large-N 
            writer_lN.writerow(list(params[:4]) +  [ reslN[0], reslN[1], reslN[2], reslN[3], reslN[7],
												reslN[4], reslN[5], reslN[6],  reslN[8], 
                                                reslN[9], reslN[10], reslN[11], reslN[12] ])
            fl.flush()

            print(f"[Saved] {params[:4]}")

    print("Scan Finished")

=======
>>>>>>> upstream/main

#Just make sure to delete the old file before running
def parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=False, xi=1, detType='Normal', crop=None, filename=None):
    
    if filename is None and Polyakov:
        filename = f'PolyakovComp_N{N}F{F}xi{xi}_{detType}.csv'
    if filename is None and not Polyakov:
        filename = f'PolyakovComp_N{N}F{F}_{detType}.csv'

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
            print(f'Summary: m2Sig={params[0]}, m2Eta={params[1]}, m2X={params[2]}, fPI={params[3]}')
            print(f'm2 = {result[0]}, c={result[1]}, ls={result[2]}, la={result[3]}')
            print(f'Tc={result[7]}, Tn={result[4]}, alpha={result[5]}, beta/H={result[6]}, message={result[8]}')
            print(f'vwLTE={result[9]}, kappaLTE={result[10]}, vwLN={result[11]}, kappaLN={result[12]}')
            #for params, result in zip(todo, p.imap_unordered(unwrap_populate, todo)): #we can go to unordered once we did all checks, decreasing speed code by 3%
            writer.writerow(list(params[:4]) +  [ result[0],  result[1],  result[2],  result[3], result[7], 
												result[4], result[5], result[6],  result[8], 
                                                result[9], result[10], result[11], result[12] ])
            f.flush()
            print(f"[Saved] {params[:4]} → result saved.")


    print("Scan Finished")



if __name__ == "__main__":

    #LARGE SCANS
    N=3; F=6

    m2Sig = np.linspace(1., 25., num=3)*1000**2
    #m2Eta = np.linspace(0.01, 0.5, num=3)*1000**2 #for N3F5 N3F6 
    m2Eta = np.linspace(1., 25., num=3)*1000**2
    m2X = np.linspace(1., 25., num=3)*1000**2

<<<<<<< HEAD
    fPi = np.linspace(0.5,1.5,num=5)*1000*np.sqrt(F/2)

    #comment out parallelscan norm to plot
    parallelScanNorm_checkpoint(m2Sig, m2Eta, m2X, fPi, N, F, Polyakov=True,xi=5)
=======
    fPi = np.linspace(0.5,1.5,num=3)*1000*np.sqrt(F/2)

    #comment out parallelscan norm to plot
    #parallelScan_checkpoint(m2Sig, m2Eta, m2X, fPi, N, F, detType='Normal', Polyakov=False,xi=1)
>>>>>>> upstream/main
	

    
    # SINGLE POINT FROM SCAN
    
'''
    Potential2.PLOT_RUN=True
    POINT_OF_INTEREST=7

	

    filename = 'PolyakovComp_N3F6xi1_AMSB.csv'; delimiter = ','
    data = np.array(np.genfromtxt(filename, delimiter=delimiter, skip_header=1, dtype=None))

    m2Sig, m2Eta, m2X, fPI, m2, c, ls, la, Tc, Tn, alpha, beta,message,vwLTE,kappaLTE,vwLN,kappaLN = data[POINT_OF_INTEREST-2]

    print(f'm2Sig = {m2Sig}, m2Eta = {m2Eta}, m2X = {m2X}, fPI = {fPI}')
    print(f'm2 = {m2}, c = {c}, ls = {ls}, la = {la}')
    print(f'Tc = {Tc}, Tn = {Tn}, alpha = {alpha}, beta = {beta}')

<<<<<<< HEAD
    print(populateN(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=True, xi=1, plot=True)) '''
=======
    print(populateWrapper(m2Sig, m2Eta, m2X, fPI, N, F, Polyakov=True, xi=1, detType='AMSB', plot=True))
>>>>>>> upstream/main

