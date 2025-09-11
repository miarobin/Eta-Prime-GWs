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
import traceback

#Terminal Colour Escape Sequences
RESET = "\033[0m"  # Reset all formatting
RED = "\033[31m"   # Red color
GREEN = "\033[32m" # Green color
CYAN = "\033[36m" # Cyan color

#Set g_star
_g_star = 47.50; MPl = 2.435E18;  


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
	
	print(f'True Min = {true_min}, False Min = {false_min}, T = {T}')
	

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
	except ValueError as e:
		if prnt: print(RED + "CosmoTransitions has returned ValueError" + RESET)
		print(e)
		traceback.print_exc()
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
		plt.plot(np.linspace(-5,V.fSigma()*1.25,num=100),V.Vtot(np.linspace(-5,V.fSigma()*1.25,num=100),T)-V.Vtot(0,T),label=f"T={T}")
	plt.legend()
	plt.show()
	
#Plots actions as a function of T
def plotAs(As, Ts):
	_Ts = [T for T,A in sorted(zip(Ts,As))]
	_As = [A for T,A in sorted(zip(Ts,As))]
	plt.plot(_Ts, np.array(_As)/np.array(_Ts))




#Finds an interpolated function of 3D Action/T as a function of T, and Tn if it exists.
	#NOTE Print just shows you all of the individual broken points in detail so it can be manually fixed.
def grid(V, tc=None, prnt=True, plot=True):
	#Range of T's to consider.
	if tc==None:
		tc = V.criticalT(prnt=plot)
		if prnt: print(f"Tc = {tc}")
		if tc == None:
			#Message 1: tc fails.
			return None, None, None, 1

	#Maximum temperature of the action scan. CT fails if T is too close to tc.
	maxT = tc*0.95 #Also want some OK-ish supercooling to get a GW signal.
	
	
	#To ensure targeting of the right area, check where a transition must have already occured by seeing if \phi=0 is a local minima or maxima.
	minTy = optimize.minimize(lambda T: abs(V.d2VdT2(0,T)),tc*(2/3), bounds=[(tc*(1/2),maxT-1)], method='Nelder-Mead')
	if minTy.fun/V.fSigma()**4<1:
		#Sometimes minTy is a terrible estimate so manually setting a minimum. 
		minT = max(minTy.x[0],maxT*.8) 
	else:
		return None, None, tc, 2
	print(f'maximum T = {maxT}, minimum T = {minT}')
	
	if plot:
		xs=np.linspace(-5,V.fSigma()*1.25,num=100)
		plt.plot(xs,V.V(xs)-V.V(0),label=f"Vtree")

		plt.plot(xs,V.Vtot(xs,minT)-V.Vtot(0,minT),linestyle='-.',label=f"T={round(minT,4)} Vtot")
		plt.plot(xs,V.Vtot(xs,(minT+maxT)/2)-V.Vtot(0,(minT+maxT)/2),linestyle='-.',label=f"T={round((maxT+minT)/2,4)} Vtot")
		
		plt.plot(xs,V.V1T(xs,minT)-V.V1T(0,minT),linestyle='--',label=f"T={round(minT,4)} V1T")
		plt.plot(xs,V.V1T(xs,(minT+maxT)/2)-V.V1T(0,(minT+maxT)/2),linestyle='--',label=f"T={round((maxT+minT)/2,4)} V1T")
		
		if V.Polyakov:
			plt.plot(xs,V.VGluonic(xs,minT)-V.VGluonic(0,minT),linestyle=':',label=f"T={round(minT,4)} VGluonic")
			plt.plot(xs,V.VGluonic(xs,(minT+maxT)/2)-V.VGluonic(0,(minT+maxT)/2),linestyle=':',label=f"T={round((maxT+minT)/2,4)} VGluonic")
	
		plt.legend()
		plt.xlabel('sigma')
		plt.ylabel('V')
		plt.show()
	
	numberOfEvaluations = 90
	#COARSE SAMPLE to find a sensible-ish minT and reduce number of calculations.
	Test_Ts = np.linspace(minT, maxT, num=numberOfEvaluations)
	for _T in Test_Ts:
		rollingAction = action(V, _T)
		#Checking it's a sensible result (NB without these conditions it's an absolute numerical disaster.
		if rollingAction is not None and rollingAction>50 and rollingAction/_T>50:
			if _T< maxT:
				minT = _T
			break
	
	#FINE SAMPLE.
	Test_Ts = moreTs = minT+(maxT-minT)*np.linspace(0, 1,num=numberOfEvaluations+50); As = []; Ts = []
	for i,_T in enumerate(Test_Ts):
		rollingAction = action(V, _T)
		if rollingAction is not None and rollingAction>0:
			As.append(rollingAction)
			Ts.append(_T)
			if prnt: print(f'Temperature {_T}, Action {rollingAction}, S/T = {rollingAction/_T}')
			
		#if plot and i%20==0:
		#	plotV(V,[0,_T-1,_T,_T+1])
	

	if len(As)==0:
		return None, None, tc, 3
	if len(As)<3:
		print (RED + "Definite precision loss. Use a smaller stepsize." + RESET)
		return None, None, tc, 4
	if len(As)<10:
		print (RED + "Possible precision loss. Use a smaller stepsize." + RESET)

	minT = min(Ts) #Update minimum T in bounds ect.

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
	

	#Either no nucleation or Ts have started to high.
	if max(_Is)<0.34:
		print(_Is)
		if plot:
			plt.plot(_Ts, _As)
			plt.xlabel('Temperature'); plt.ylabel('A(T)')
			plt.show()
			plt.plot(_Ts, _Is)
			plt.xlabel('Temperature'); plt.ylabel('I(T)')
			plt.show()
		return None, None, tc, 5

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
		return None, None, tc, 6
	
	if res.success and res.fun <=0.1:
		if plot: 
			print(f"Tn = {res.x[0]}, Minimisation method L-BFGS-B")
			plt.plot(_Ts, [interpolator(_T) for _T in _Ts])
			plt.xlabel('Temperature'); plt.ylabel('I(T)')
			plt.show()
			
			interp = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(moreTs)
			plotAs(_As,_Ts)#Comparing with original data
			plt.plot(moreTs,interp)#Check if the interpolator is doing well.
			plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
			plt.show()
			print(res)
			
			tn=res.x[0]
			xs=np.linspace(-5,V.fSigma()*1.25,num=100)
			plt.plot(xs,V.V(xs)-V.V(0),label=f"Vtree")

			plt.plot(xs,V.Vtot(xs,tc)-V.Vtot(0,tc),linestyle='-.',label=f"T={round(tc,4)} Tc Vtot")
			plt.plot(xs,V.Vtot(xs,tn)-V.Vtot(0,tn),linestyle='-.',label=f"T={round(tn,4)} Tn Vtot")
			
			plt.plot(xs,V.V1T(xs,tc)-V.V1T(0,tc),linestyle='--',label=f"T={round(tc,4)} Tc V1T")
			plt.plot(xs,V.V1T(xs,tn)-V.V1T(0,tn),linestyle='--',label=f"T={round(tn,4)} Tn V1T")
			
			if V.Polyakov:
				plt.plot(xs,V.VGluonic(xs,tc)-V.VGluonic(0,tc),linestyle=':',label=f"T={round(tc,4)} Tc VGluonic")
				plt.plot(xs,V.VGluonic(xs,tn)-V.VGluonic(0,tn),linestyle=':',label=f"T={round(tn,4)} Tn VGluonic")
		
			plt.legend()
			plt.xlabel('sigma')
			plt.ylabel('V')
			plt.show()
			
					
		spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))
		
		#Checking to see if the spline doesn't oscillate.
		if all(spl.derivatives(_Ts)[1]>0):
			return res.x[0], spl, tc, 0
		else: 
			spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=1, s=len(_Ts)+np.sqrt(2*len(_Ts)))
			if plot:
				plotAs(_As,_Ts)#Comparing with original data
				plt.plot(moreTs,interp)#Check if the interpolator is doing well.
				plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
				plt.show()
				print(res)
			if all(spl.derivatives(_Ts)[1]>0):
				return res.x[0], spl, tc, 0
			else:
				return 0, 0, 0, 17
		
	#If previous Tn failed, trying again with a new method.
	else:
		res = optimize.minimize(lambda T: abs(interpolator(T) - 0.34), (narrowRegion[0]+narrowRegion[-1])/2, bounds=[(min(narrowRegion), max(narrowRegion))],method='Nelder-Mead',tol=1e-3)
		if res.success and res.fun <=0.1:
			if plot: 
				print(f"Tn = {res.x[0]}, Minimisation method Nelder-Mead")
				print(res)
				plt.plot(_Ts, [interpolator(_T) for _T in _Ts])
				plt.xlabel('Temperature'); plt.ylabel('I(T)')
				plt.show()
					
				interp = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(moreTs)
				plotAs(_As,_Ts)#Comparing with original data
				plt.plot(moreTs,interp)#Check if the interpolator is doing well.
				plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
				plt.show()
				
				#For manually checking each point.
				validInput=False
				while not validInput:
					toRemove = input ("Enter 'r' to remove and flag or 'k' to keep:")
					print(f'You have selected {toRemove}')
					if str(toRemove.strip()) == 'r':
						print('Point added to removal list.')
						return 0, 0, 0, 17
					elif str(toRemove.strip()) == 'k':
						print('Point kept.')
						validInput=True
					else:
						print('Input character not valid. Try again please')

			#Need to return Tn, S_3/T and success error message.
			spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))
		
			#Checking to see if the spline doesn't oscillate.
			if all(spl.derivatives(_Ts)[1]>0):
				return res.x[0], spl, tc, 0
			else: 
				spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=1, s=len(_Ts)+np.sqrt(2*len(_Ts)))
				if plot:
					plotAs(_As,_Ts)#Comparing with original data
					plt.plot(moreTs,interp)#Check if the interpolator is doing well.
					plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
					plt.show()
					print(res)
				if all(spl.derivatives(_Ts)[1]>0):
					return res.x[0], spl, tc, 0
				else:
					return 0, 0, 0, 17
		
		print(res)
		if plot:
			print('abject failure')
			plt.plot(_Ts, np.array(_As)/np.array(_Ts))
			plt.plot(np.linspace(_Ts[0],_Ts[-1]), (interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(np.linspace(_Ts[0],_Ts[-1]))))
			plt.show()
		return None, None, tc, 7
	
	#If you reach this point something's gone totally wrong...
	return None,None, tc, 8

			
	
	
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
	Tn, grd, message = grid(V,prnt=True,plot=True)
	
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
	
	#xi, muSig, lmb, kappa, m2Sig, N, F, detPow
	V=Potential2.Potential(0,*[0.2, 2, 1, 100**2],1,4,1)
	
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
	

