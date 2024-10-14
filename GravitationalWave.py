from cosmoTransitions.tunneling1D import SingleFieldInstanton
from cosmoTransitions.tunneling1D import PotentialError
from cosmoTransitions import helper_functions
from functools import reduce
import Potential
from scipy import interpolate, optimize
import numpy as np
import matplotlib.pyplot as plt
import csv
import TestCode

#Terminal Colour Escape Sequences
RESET = "\033[0m"  # Reset all formatting
RED = "\033[31m"   # Red color
GREEN = "\033[32m" # Green color
CYAN = "\033[36m" # Cyan color

#Set g_star
_g_star = 114
##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2


#Calculate the action S3 for a given temperature T using CosmoTransitions SingleFieldInstanton class.
def action(V,T,prnt=False):
	#First find & classify the true & false minima
	vT = V.findminima(T)
	
	#If rhs is none at this point it means there's only one minima. Return action as None.
	if vT == None: return None
	
	true_min = min([0,vT], key=lambda phi: V.Vtot(phi,T))
	false_min = max([0,vT], key=lambda phi: V.Vtot(phi,T))

	try:
		#Initialise instanton in CosmoTransitions.
		Instanton = SingleFieldInstanton(true_min, false_min, lambda phi: V.Vtot(phi,T), alpha=2)
		Profile = Instanton.findProfile()
		
		#Find the action & return it.
		action = Instanton.findAction(Profile)
		if action < 0 and prnt:
			print('negative action')
			return None
		return action
		
	
	#Sometimes CosmoTransitions throws these errors (not entirely sure why). Might be worth investigating these later. For now, just returning None.
	except helper_functions.IntegrationError:
		if prnt: print(RED + "CosmoTransitions has returned IntegrationError" + RESET)
		return None
	except ValueError:
		if prnt: print(RED + "CosmoTransitions has returned ValueError" + RESET)
		return None
	except PotentialError:
		if prnt: print(RED + "CosmoTransitions has returned PotentialError" + RESET)
		return None
	except AssertionError:
		if prnt: print(RED + "CosmoTransitions has returned PotentialError" + RESET)
		return None
	
	
#Plots the potential as a function of temperature
def plotV(V, Ts):
	for T in Ts:
		plt.plot(np.linspace(-5,V.fSigmaApprox()*1.25,num=100),V.Vtot(np.linspace(-5,V.fSigmaApprox()*1.25,num=100),T)-V.Vtot(0,T),label=f"T={T}")
	plt.legend()
	plt.show()
	
#Plots actions as a function of T
def plotAs(As, Ts):
	_Ts = [T for T,A in sorted(zip(Ts,As))]
	_As = [A for T,A in sorted(zip(Ts,As))]
	plt.plot(_Ts, np.array(_As)/np.array(_Ts))
	plt.plot(_Ts, np.ones((len(_Ts)))*140)
	plt.show()



#Finds an interpolated function of 3D Action/T as a function of T, and Tn if it exists.
	#NOTE Print just shows you all of the individual broken points in detail so it can be manually fixed.
def grid(V, tc=None, prnt=True, plot=False):
	#Range of T's to consider.
	if tc==None:
		tc = V.criticalT(prnt=plot)
		if prnt: print(f"Tc = {tc}")
		if tc == None:
			return None, None, 1

	maxT = tc-1.5

	#Set up for the scan of S3 against T:
	stepSize = .2; bounds = []; flare = []; guess = None
	#First point to find if we go up or down
	T1 = maxT; T2 = maxT-stepSize
	A1=action(V,T1); A2=action(V,T2)
	if prnt: print(f"A1 = {A1}, A2 = {A2}, T1 = {T1}, T2 = {T2}")
	if A1 is None or A2 is None:
		if A1 is None:
			T1+=0.5; A1=action(V,T1,prnt=plot)
		if A2 is None:
			T2+=0.5; A2=action(V,T2,prnt=plot)
		if A1 is None or A2 is None:
			if plot: plotV(V, [T1,T2])
			return None, None, 2
	#Bank of Ts and corresponding As
	Ts = [T1, T2]
	As = [A1, A2]
	
	#ASCEND
	if A1/T1<=140:
		#A variable stepsize...
		miniStepSize = (tc-T1)*0.05
		
		Aa = A1; Ta = T1
		Ab = action(V,T1+miniStepSize); Tb = T1+miniStepSize
		if prnt: print(f"Aa = {Aa}, Ab = {Ab}, Ta = {Ta}, Tb = {Tb}")
		if Ab is None:
			return None, None, 4
		As.append(Ab); Ts.append(Tb)
		#Ascend in temperature by small steps until we go above 140.
		while Ab/Tb <= 140:
			Aa = Ab; Ta = Tb
			#Update the backstep size so we move towards Tc
			miniStepSize=(tc-Tb)*0.1
			#If it gets too small we're probably in trouble.?
			if miniStepSize < 0.00001: 
				return None, None, 3
						
			Ab = action(V,Tb+miniStepSize); Tb += miniStepSize
			if prnt: print(f"Aa = {Aa}, Ab = {Ab}, Ta = {Ta}, Tb = {Tb}")
			if Ab is None:
				#Investigate individual point.
				if plot: 
					plotV(V, Ts+[Tb,tc])
					print(f"tc = {tc}")
					plotAs(As, Ts)
					Ab = action(V,Tb+.5); Tb = Tb+.5
					
					if Ab is None:
						return None, None, 4
					else:
						print(f"Ab = {Ab}, Ab/Tb = {Ab/Tb}")
						As.append(Ab); Ts.append(Tb)
				else:
					return None, None, 4
			elif Ab<Aa:
				#Difference in minima might be too small.
				if plot: 
					plotV(V, Ts+[Tb,0,400])
					plotAs(As+[Ab],Ts+[Tb])
				return None,None, 5 
			else:
				As.append(Ab); Ts.append(Tb)
					
		#Ascending stopping criteria achieved if code exits while loop. Flare around temperature to more accurately calculate gradient.
		bounds = [(Ta*0.99,Tb*1.01)]; guess = Ta + miniStepSize/2
		flare = [Tb + (tc - Tb)*adj for adj in [-3,-2,-1.5,-1.1,-0.65,-0.5,-0.2,-0.1,-0.05,-0.025,-0.01,0.01,0.025,0.05] if Tb + (tc - Tb)*adj < tc] 
		
	
	#DESCEND
	elif A1/T1 >= 140 and A2/T2 >= 140: 
	#Fixed Stepsize since we don't need to sneak this time
		while A2/T2>=140 and A1/T1>=140:
			#Use newton's method to descend the curve
			m = (A1/T1 - A2/T2)/(T1-T2); c = (A2*(T1/T2) - A1*(T2/T1))/(T1-T2)
			T1 = (140-c)/m; T2=T1-stepSize
			
			if T1<10 or T2<10: #Basically no way there's a PT now.
				if plot: 
					print(f'T1 = {T1}, T2 = {T2}')
					plotAs(As,Ts)
				return None, None,6
			
			A1=action(V,T1,prnt=plot)
			A2=action(V,T2,prnt=plot)
			
			if prnt: print(f"A1 = {A1}, A2 = {A2}, T1 = {T1}, T2 = {T2}")

		
			#First check you can find an action...
			if A1 is not None and A2 is not None:
				#Add findings to the 'bank'
				Ts = Ts + [T1,T2]
				As = As + [A1,A2]
			
			else:
				if A1 is None:
					A1=action(V,T1+0.5,prnt=plot); T1+=0.5
				if A2 is None:
					A2=action(V,T2+0.5,prnt=plot); T2+=0.5
				if A1 is None or A2 is None:
					if plot: 
						plotV(V, Ts+[T1,T2,tc])
						plotAs(As,Ts)
					return None, None,7
				else:
					Ts = Ts + [T1,T2]
					As = As + [A1,A2]
					if prnt: print(f"A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
		
			if (A1/T1) < (A2/T2) and A1/T1>=140:	
				#We have passed the minima of the S3/T at this point and not reached S3/T=140 so return none found

				if plot: 
					print(f"A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
					plotAs(As + [A1,A2],Ts + [T1,T2])
				return None, None, 0
			


		#If we found Tc by descending, we'll want to do flare around descent values
		flare = [T1 + adj*stepSize for adj in [-0.5,-0.75,-1.,-1.2,-2,-2.5,-3,-4,-6,-10]]
		if A1/T1>=140 and A2/T2<=140:
			bounds = [(T2,T1)]; guess = T2 + stepSize/2
		else:
			bounds = [(T1*.99,Ts[-3]*1.01)]; guess = T1 + abs(T1 - Ts[-3])/2
		
	elif A1/T1 >= 140 and A2/T2 <= 140: 
		#Happened to find crossing point with our first guess
		flare = [T1 + adj*stepSize for adj in [(tc-T1)*0.1/stepSize,(tc-T1)*0.05/stepSize,-1/3,-1/6,-0.5,-2/3,-1.5,-2,-3]]
		bounds = [(T2,T1)]; guess = T2 + stepSize/2

	else:
		if prnt: print(f"A1/T1 = {A1/T1}, A2/T2 = {A2/T2}")
		return None, None, 8
	
	#For some reason T>Tc was getting through
	As = [A for i,A in enumerate(As) if Ts[i]<tc]
	Ts = [T for T in Ts if T < tc]

	Ts = Ts + flare
	As = As + [action(V,t) for t in flare]
	
	if not all(item is not None for item in As):
		if plot:
			
			plotAs([As[i] for i in range(len(Ts)) if As[i] is not None], [Ts[i] for i in range(len(Ts)) if As[i] is not None])
			length = len(Ts)
			Ts = [Ts[i] for i in range(length) if As[i] is not None]; As = [As[i] for i in range(length) if As[i] is not None]
			
		else:
			Ts = [T for i,T in enumerate(Ts) if As[i] is not None]
			As = [A for A in As if A is not None]
			
			if len(As)<5:
				return None, None, 11

	_Ts = [T for T,A in sorted(zip(Ts,As))]
	_As = [A for T,A in sorted(zip(Ts,As))]

	interpolator = interpolate.UnivariateSpline(_Ts,np.array(_As)/np.array(_Ts),k=3, s=len(_Ts)+np.sqrt(2*len(_Ts))/2)

	if prnt: print(f"Bounds = {bounds}, guess = {guess}")
	res = 0
	try:
		res = optimize.minimize(lambda T: abs(interpolator(T) - 140.), guess, bounds=bounds,method='L-BFGS-B',tol=1e-2)
	except ValueError:
		return None, None, 12
	
	if res.success and res.fun <=5:
		if plot: 
			print(f"Tn = {res.x[0]}, Nruns = {len(Ts)}")
			plotAs(As,Ts)
		return res.x[0], interpolator, 0
	else:
		res = optimize.minimize(lambda T: abs(interpolator(T) - 140.), guess, bounds=bounds,method='Nelder-Mead')
		if res.success and res.fun <=5:
			if plot: 
				print(f"Tn = {res.x[0]}, Nruns = {len(Ts)}")
				plotAs(As,Ts)
				print(res)
			return res.x[0], interpolator, 0
		
		print(res)
		if plot:
			plt.plot(_Ts, np.array(_As)/np.array(_Ts))
			plt.plot(np.linspace(_Ts[0],_Ts[-1]), (interpolator(np.linspace(_Ts[0],_Ts[-1]))))
			plt.plot(_Ts, np.ones((len(_Ts)))*140)
			plt.show()
		return None, None, 9
	
	#If you reach this point something's gone totally wrong...
	return None,None,10

			
	
	
def checkProfile(V, Instanton, Profile, T):
	#Have a look at how well CosmoTransitions has solved the tunnelling ODE.
	rhs = []; lhs = []
	try:
		Phi = interpolate.UnivariateSpline(Profile.R,Profile.Phi)
		for r in Profile.R[5:]:
			rhs.append(Phi.derivatives(r)[2]+(2/r)*Phi.derivatives(r)[1])
			lhs.append(V.dVdh(r, T))
			
		plt.plot(Profile.R[5:], lhs, label='lhs')
		plt.plot(Profile.R[5:], rhs, label='rhs')
		plt.xlabel('r')
		plt.legend()
		plt.show()
	
	except ValueError as e:
		print(e)


	
#Compute beta/H for gravitational wave signals.
def beta_over_H(V, Tn, act):
	return act.derivatives(Tn)[1]*Tn
	
def alpha(V, Tn):
	minima = V.findminima(Tn)
	delV = abs(V.Vtot(0,Tn) - V.Vtot(minima,Tn))
	ddelVdT = abs(V.dVdT(0,Tn) - V.dVdT(minima,Tn))
	#Note g_star is defined at the top of the document.
	return (30/(_g_star*np.pi**2*Tn**4)) * (-delV + Tn*ddelVdT/4)
	
	
def gravitationalWave(V):
	#This method is slow so we only want to run it once, then just pass through the interpolated function to everything else.
	Tn, grd, message = grid(V,prnt=False)
	
	#Calculate nucleation temperature
	if Tn == None:
		#No transition occurs
		return None
	#Calculate beta/H
	beta_H = beta_over_H(V, Tn, grd)
	#Calculate alpha
	alp = alpha(V, Tn)
	
	return(Tn, alp, beta_H)


def wallVelocity(V, alpha, T):
	#Calculating the wall velocity at temperature T. Assumes a constant temperature.
	
	return 1
		
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



if __name__ == "__main__":
	print('hello')
	test = TestCode.TestPotential()
	'''
	test.test_F3(0)
	test.test_F3(1)
	test.test_F3(2)
	'''
	
	#test.test_F4(0)
	#test.test_F4(2)
	#test.test_F4(2)
	
	test.testN_F4(1, 2)
	#test.testSymmRestoration(1,3)
	
	#xi, muSig, lmb, kappa, m2Sig, N, F
	V=Potential.Potential(0,*[0.2, 2, 1, 100**2],1,4)
	
	def fSig1_function(muSig, lmb, kappa, m2Sig): return 2*(2*m2Sig)**0.5 / (kappa + 4*lmb - muSig)**0.5
	fSig1 = fSig1_function(0.2, 2, 1, 100**2)	
	#print(fSig1)
	print(V.V1T(fSig1, 500))
	print(V.mSq['Phi'][0](fSig1,10))
	print(V.mSq['Eta'][0](fSig1,10))
	print(V.mSq['X8'][0](fSig1,10))
	print(V.mSq['Pi8'][0](fSig1,10))

	

	for T in range(300,500):
		plt.scatter(T, V.Vtot(fSig1,T))
	plt.show()
	

