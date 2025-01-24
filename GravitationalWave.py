from cosmoTransitions.tunneling1D import SingleFieldInstanton
from cosmoTransitions.tunneling1D import PotentialError
from cosmoTransitions import helper_functions
from functools import reduce
import Potential2
from scipy import interpolate, optimize, integrate
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
_g_star = 106.75; MPl = 2.435E18;  



#Calculate the action S3 for a given temperature T using CosmoTransitions SingleFieldInstanton class.
def action(V,T,prnt=True):
	#First find & classify the true & false minima
	vT = V.findminima(T)
	
	#If rhs is none at this point it means there's only one minima. Return action as None.
	if vT == None: 
		print('No vT found')
		return None
	
	true_min = min([0,vT], key=lambda phi: V.Vtot(phi,T))
	false_min = max([0,vT], key=lambda phi: V.Vtot(phi,T))
	

	if false_min > true_min:
		if prnt:
			print('Attempting to calculate tunnelling in the wrong direction.')
		return None

	try:
		#Initialise instanton in CosmoTransitions.
		Instanton = SingleFieldInstanton(true_min, false_min, lambda phi: V.Vtot(phi,T), alpha=2)
		Profile = Instanton.findProfile()
		
		#Find the action & return it.
		action = Instanton.findAction(Profile)
		if action < 0 and prnt:
			print('negative action')
			return None
		elif action < 0:
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




#Finds an interpolated function of 3D Action/T as a function of T, and Tn if it exists.
	#NOTE Print just shows you all of the individual broken points in detail so it can be manually fixed.
def grid(V, tc=None, prnt=True, plot=False):
	#Range of T's to consider.
	if tc==None:
		tc = V.criticalT(prnt=plot)
		if prnt: print(f"Tc = {tc}")
		if tc == None:
			return None, None, 1

	maxT = tc-0.2
	
	
	#To ensure targeting of the right area, check where a transition must have already occured by seeing if \phi=0 is a local minima or maxima.
	minTy = optimize.minimize(lambda T: abs(V.d2VdT2(0,T)),tc*(2/3), bounds=[(tc*(1/2),maxT-1)], method='Nelder-Mead')
	if minTy.fun/V.fSigma()**4<1:
		minT = max(minTy.x[0],maxT*.85) #This is the point at which \phi=0 becomes a local maxima.
	else:
		return None, None, 2
	print(f'maximum T = {maxT}, minimum T = {minT}')
	
	
	numberOfEvaluations = 150
	#COARSE SAMPLE to find a sensible-ish minT and reduce number of calculations.
	Test_Ts = np.linspace(minT, maxT, num=numberOfEvaluations)
	for _T in Test_Ts:
		rollingAction = action(V, _T)
		if rollingAction is not None and rollingAction>50 and rollingAction/_T>50:
			if _T< maxT:
				minT = _T
			break
	print(f'minT={minT}')
	#FINE SAMPLE. The interesting stuff is usually happening near minT, hence the 'square'.
	#Test_Ts = moreTs = minT+(maxT-minT)*np.linspace(0, 1,num=numberOfEvaluations)**2; As = []; Ts = []
	#Trying out without the square.
	Test_Ts = moreTs = minT+(maxT-minT)*np.linspace(0, 1,num=numberOfEvaluations); As = []; Ts = []
	for _T in Test_Ts:
		rollingAction = action(V, _T)
		if rollingAction is not None and rollingAction>0:
			As.append(rollingAction)
			Ts.append(_T)
			if prnt: print(f'Temperature {_T}, Action {rollingAction}, S/T = {rollingAction/_T}')
	

	if len(As)==0:
		return None, None, 3
	if len(As)<3:
		print (RED + "Definite precision loss. Use a smaller stepsize." + RESET)
		return None, None, 4
	if len(As)<10:
		print (RED + "Possible precision loss. Use a smaller stepsize." + RESET)

	minT = min(Ts) #Update minimum T in bounds ect.

	print(len(Ts))
	print(len(As))

	Ts = [T for T,A in sorted(zip(Ts,As))]
	As = [A for T,A in sorted(zip(Ts,As))]
	
	_Ts = [T for T,A in zip(Ts,As) if A>0]
	_As = [A for T,A in zip(Ts,As) if A>0]
	As=_As
	Ts=_Ts



	#Previous interpolator just interpolates S_3/T but new interpolator interpolates the integrand of eq 3.10 of 2309.16755.
	Ts = np.array(Ts); As = np.array(As)

	b = 12*np.pi* (30/(_g_star*np.pi**2))**2 * 1/(2*np.pi)**(3/2) #Note transfer of MPl to following line to preserve precision
	Integrand = lambda T: np.array([(1/Ts[i])**2 * (As[i]/Ts[i])**(3/2) * np.exp(-As[i]/Ts[i] + 4*np.log(MPl)) * (1/T - 1/Ts[i])**3 if T<Ts[i] else 0 for i in range(len(Ts))])


	Is = [b*integrate.trapezoid(Integrand(T), Ts) for T in Ts]
	
	_Is = np.array([I for I, T, A in zip(Is,Ts,As) if I<100])
	_Ts = np.array([T for I, T, A in zip(Is,Ts,As) if I<100])
	_As = np.array([A for I, T, A in zip(Is,Ts,As) if I<100])
	

	
	if max(_Is)<0.34:
		print(_Is)
		if plot:
			plt.plot(_Ts, _As)
			plt.xlabel('Temperature'); plt.ylabel('A(T)')
			plt.show()
			plt.plot(_Ts, _Is)
			plt.xlabel('Temperature'); plt.ylabel('I(T)')
			plt.show()
		return None, None, 5

	interpolator = interpolate.Akima1DInterpolator(_Ts,_Is)
	#NOTE ERROR HERE FROM NOT BEING ABLE TO INTEGRATE ALL THE WAY TO TC!! Should be small from exponential suppression.

	moreTs = min(_Ts)+(max(_Ts)-min(_Ts))*np.linspace(0, 1,num=500)**2

	if plot:
		plt.plot(moreTs, [interpolator(_T) for _T in moreTs])
		plt.plot(_Ts, _Is)
		plt.xlabel('Temperature'); plt.ylabel('I(T)')
		plt.show()
		
	res = 0
	try:
		narrowRegion = [T for T in moreTs if 0.1<interpolator(T)<1]
		res = optimize.minimize(lambda T: abs(interpolator(T) - 0.34), (narrowRegion[0]+narrowRegion[-1])/2, bounds=[(min(narrowRegion), max(narrowRegion))],method='L-BFGS-B',tol=1e-3)
	except ValueError:
		return None, None, 6
	
	if res.success and res.fun <=0.1:
		if plot: 
			print(f"Tn = {res.x[0]}, Minimisation method L-BFGS-B")
			print(res)

		return res.x[0], interpolate.UnivariateSpline(_Ts,_As/_Ts,k=3, s=len(_Ts)+np.sqrt(2*len(_Ts))/2), 0
	else:
		res = optimize.minimize(lambda T: abs(interpolator(T) - 0.34), (narrowRegion[0]+narrowRegion[-1])/2, bounds=[(min(narrowRegion), max(narrowRegion))],method='Nelder-Mead',tol=1e-3)
		if res.success and res.fun <=0.1:
			if plot: 
				print(f"Tn = {res.x[0]}, Minimisation method Nelder-Mead")
				print(res)
				plt.plot(_Ts, [interpolator(_T) for _T in _Ts])
				plt.xlabel('Temperature'); plt.ylabel('I(T)')
				plt.show()

			#Need to return Tn, S_3/T and success error message.
			return res.x[0], interpolate.UnivariateSpline(_Ts,_As/_Ts,k=3, s=len(_Ts)+np.sqrt(2*len(_Ts))/2), 0
		
		print(res)
		if plot:
			plt.plot(_Ts, np.array(_As)/np.array(_Ts))
			plt.plot(np.linspace(_Ts[0],_Ts[-1]), (interpolate.UnivariateSpline(_Ts,_As/_Ts,k=3, s=len(_Ts)+np.sqrt(2*len(_Ts))/2)(np.linspace(_Ts[0],_Ts[-1]))))
			plt.show()
		return None, None, 7
	
	#If you reach this point something's gone totally wrong...
	return None,None,8

			
	
	
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
	V=Potential2.Potential(0,*[0.2, 2, 1, 100**2],1,4)
	
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
	

