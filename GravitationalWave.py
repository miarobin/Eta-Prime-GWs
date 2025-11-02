from cosmoTransitions.tunneling1D import SingleFieldInstanton
from cosmoTransitions.tunneling1D import PotentialError
from cosmoTransitions import helper_functions
from functools import reduce
import Potential2
from scipy import interpolate, optimize, integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import csv
import traceback
from debug_plot import debug_plot

class BadlyIRDivergent(Exception):
    pass

#Terminal Colour Escape Sequences
RESET = "\033[0m"  # Reset all formatting
RED = "\033[31m"   # Red color
GREEN = "\033[32m" # Green color
CYAN = "\033[36m" # Cyan color

#Set constants
MPl = 2.435E18; 


#Calculate the action S3 for a given temperature T using CosmoTransitions SingleFieldInstanton class.
def action(V,T,prnt=True, plot=False):
    #First find & classify the true & false minima
    vT = V.findminima(T)
    IRDivSmoothing = Potential2.IRDIVSMOOTHING
    
    #If rhs is none at this point it means there's only one minima. Return action as None.
    if vT == None:
        print('No vT found')
        return None, 0
    
    true_min = min([0,vT], key=lambda phi: V.Vtot(phi,T))
    false_min = max([0,vT], key=lambda phi: V.Vtot(phi,T))
    
    print(f'True Min = {true_min}, False Min = {false_min}, T = {T}')
    
 
    if false_min > true_min:
        if prnt:
            print('Attempting to calculate tunnelling in the wrong direction.')
        return None, 0
    
 
    #4-point effective vertices.
    gSig_eff = np.abs(3*V.lambdas - V.c*V.fSIGMA**(V.F*V.detPow-4)*(V.F*V.detPow)*(V.F*V.detPow-1)*(V.F*V.detPow-2)*(V.F*V.detPow-3)/V.F**2)
    gPi_eff = np.abs(V.lambdas*(V.F**2+1) + V.lambdaa*(V.F**2-4)
                   - V.c*V.fSIGMA**(V.F*V.detPow-4)*(V.detPow/V.F)*(V.detPow*V.F**3-4*V.F**2+V.detPow*V.F+6))/((V.F**2-1))
        
    pSig = lambda sig: gSig_eff * (T/(np.abs(V.MSq['Sig'][0](sig,T))**(1/2)+1e-24))
    pPi = lambda sig: gPi_eff * (V.F**2-1) * (T/(np.abs(V.MSq['Pi'][0](sig,T))**(1/2)+1e-24))
    
    numTests = 500
    sigmas = np.linspace(false_min,true_min,num=numTests)
    cut = [i for i,sig in enumerate(sigmas) if pSig(sig)<16*np.pi and pPi(sig)<16*np.pi] #If far too big just throw away point completely. Maybe something like 16 pi?
 
    #Straight up how much of the tunneling potential fails to be perturbative.
    PFrac = len(cut)/numTests
 
    if IRDivSmoothing:
    #A single temperature fit to try & smooth out IR divergences.
 
        if len(cut)<numTests*.9:
            raise BadlyIRDivergent("At least 10 % of points failed to be perturbative.")
 
        _sigmas = sigmas[cut]
            
        weights=[]
        for sig in _sigmas:
            if pSig(sig)<4*np.pi and pPi(sig)<4*np.pi:
                weights.append(1.)
            else:
                weights.append((4*4*np.pi/max(pSig(sig),pPi(sig))-1)/3)#Retains some data.
 
        _V = interpolate.UnivariateSpline(_sigmas, V.Vtot(_sigmas,T),w=weights,s=len(_sigmas)*1.1)
 
    
    else:
        _V = lambda _sigma: V.Vtot(_sigma,T)
 
 
    #DO SOME PLOTS!
    if plot:
        plt.plot(sigmas,V.Vtot(sigmas,T)/V.fSIGMA**4-V.Vtot(0,T)/V.fSIGMA**4, label='Vtot')
        if IRDivSmoothing:
            plt.plot(_sigmas,weights,label='weights')
            plt.plot(sigmas,_V(sigmas)/V.fSIGMA**4-_V(0)/V.fSIGMA**4,label='Weighted Vtot')
        plt.plot(sigmas,-pSig(sigmas)/V.fSIGMA,label='Sigma effective')
        plt.plot(sigmas,-pPi(sigmas)/V.fSIGMA,label='Pi effective')
        plt.plot(sigmas,V.MSq['Sig'][0](sigmas,T)/V.fSIGMA**2,label='sig-mass')
        plt.plot(sigmas,V.MSq['Pi'][0](sigmas,T)/V.fSIGMA**2,label='pi-mass')
 
        plt.legend()
        debug_plot(name="debug", overwrite=False)
        #plt.show()
 
    try:
        #Initialise instanton in CosmoTransitions.
        Instanton = SingleFieldInstanton(true_min, false_min, _V, alpha=2)
        Profile = Instanton.findProfile()
            
        #Find the action & return it.
        action = Instanton.findAction(Profile)
        if action < 0 and prnt:
            print('negative action')
            return None, 0
        elif action < 0:
            return None, 0
            
        return action, PFrac
            
    
    #Sometimes CosmoTransitions throws these errors (not entirely sure why). Might be worth investigating these later. For now, just returning None.
    except helper_functions.IntegrationError:
        if prnt: print(RED + "CosmoTransitions has returned IntegrationError" + RESET)
        return None, 0
    except ValueError as e:
        if prnt: print(RED + "CosmoTransitions has returned ValueError" + RESET)
        print(e)
        traceback.print_exc()
        return None, 0
    except PotentialError:
        if prnt: print(RED + "CosmoTransitions has returned PotentialError" + RESET)
        return None, 0
    except AssertionError:
        if prnt: print(RED + "CosmoTransitions has returned PotentialError" + RESET)
        return None, 0
	
	
#Plots the potential as a function of temperature
def plotV(V, Ts):
	for T in Ts:
		plt.plot(np.linspace(-5,V.fSIGMA*Potential2.SIGMULT,num=100),V.Vtot(np.linspace(-5,V.fSIGMA*Potential2.SIGMULT,num=100),T)-V.Vtot(0,T),label=f"T={T}")
	plt.legend()
	debug_plot(name="debug", overwrite=False)
	#plt.show()
	
#Plots actions as a function of T
def plotAs(As, Ts):
	_Ts = [T for T,A in sorted(zip(Ts,As))]
	_As = [A for T,A in sorted(zip(Ts,As))]
	plt.plot(_Ts, np.array(_As)/np.array(_Ts))




#Finds an interpolated function of 3D Action/T as a function of T, and Tn if it exists.
	#NOTE Print just shows you all of the individual broken points in detail so it can be manually fixed.
def grid(V, tc=None, prnt=True, plot=True, ext_minT=None):
	IRDivSmoothing = Potential2.IRDIVSMOOTHING
	
	#Range of T's to consider.
	if tc==None:
		tc = V.criticalT(prnt=plot)
		if prnt: print(f"Tc = {tc}")
		if tc == None:
			#Message 1: tc fails.
			return None, None, None, 1
	if ext_minT:
		if tc<ext_minT:
			return None, None, tc, 13 #No possibility for PT.

	#Maximum temperature of the action scan. CT fails if T is too close to tc.
	maxT = tc*0.99 #Also want some OK-ish supercooling to get a GW signal.
	
	
	#To ensure targeting of the right area, check where a transition must have already occured by seeing if \phi=0 is a local minima or maxima.
	minTy = optimize.minimize(lambda T: abs(V.d2VdT2(0,T)),tc*(2/3), bounds=[(tc*(1/2),maxT-1)], method='Nelder-Mead')
	if minTy.fun/V.fSIGMA**4<1:
		#Sometimes minTy is a terrible estimate so manually setting a minimum based on where we cutoff the noise monitoring. 
		if ext_minT is not None:
			minT = max(minTy.x[0],tc*.75,ext_minT) 
		else:
			minT = max(minTy.x[0],tc*.75)
	else:
		return None, None, tc, 2
	print(f'maximum T = {maxT}, minimum T = {minT}')
	
	if plot:
		xs=np.linspace(-5,V.fSIGMA*Potential2.SIGMULT,num=100)
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
		debug_plot(name="debug", overwrite=False)
		#plt.show()
	
	numberOfEvaluations = 90
	#COARSE SAMPLE to find a sensible-ish minT and reduce number of calculations.
	Test_Ts = np.linspace(minT, maxT, num=numberOfEvaluations)
	for i,_T in enumerate(Test_Ts):
		try:
			if plot and i%15==0:
				#Plots perturbativity every 5th evaluation.
				rollingAction,_ = action(V, _T,plot=True)
			else:
				rollingAction,_ = action(V, _T)
		except BadlyIRDivergent as e:
			print(e) #Don't really mind about IR divergences at this point since we only want an initial minimum T.
			rollingAction=None
		#Checking it's a sensible result (NB without these conditions it's an absolute numerical disaster.
		if rollingAction is not None and rollingAction>50 and rollingAction/_T>50:
			if prnt: print(f'Temperature {_T}, Action {rollingAction}, S/T = {rollingAction/_T}')
			if _T< maxT:
				minT = _T
			break
	
	#FINE SAMPLE.
	Test_Ts = moreTs = minT+(maxT-minT)*np.linspace(0, 1,num=numberOfEvaluations+50); As = []; Ts = []; IRDiv = False
	for i,_T in enumerate(Test_Ts):
		try:
			if plot and i%15==0:
				#Plots perturbativity every 5th evaluation.
				rollingAction,_ = action(V, _T,plot=True)
			else:
				rollingAction,_ = action(V, _T)
		except BadlyIRDivergent as e:
			print(e)
			IRDiv=True #If the PT fails to nucleate because of IR divergences, we return a different error code.
			rollingAction=None
			
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


	#ADD WALL VELOCITY!
	#NOTE TO FUTURE SELVES: defining SM part of g_star at the CRITICAL TEMPERATURE.
	b = 12*np.pi* (30/(V._g_star(tc)*np.pi**2))**2 * 1/(2*np.pi)**(3/2) #Note transfer of MPl to following line to preserve precision
	Integrand = lambda T: np.array([(1/Ts[i])**2 * (As[i]/Ts[i])**(3/2) * np.exp(-As[i]/Ts[i] + 4*np.log(MPl)) * (1/T - 1/Ts[i])**3 if T<Ts[i] else 0 for i in range(len(Ts))])


	Is = [b*integrate.trapezoid(Integrand(T), Ts) for T in Ts]
	
	_Is = np.array([I for I, T, A in zip(Is,Ts,As) if I<150])
	_Ts = np.array([T for I, T, A in zip(Is,Ts,As) if I<150])
	_As = np.array([A for I, T, A in zip(Is,Ts,As) if I<150])
	
	if max(Is) > 150 and max(_Is)<0.34:
		return None, None, tc, 12 #StepSize is too small!

	if len(_As)==0:
		return None, None, tc, 9 #An action has been found, but not less than I=100. Not sure exactly what this means but it's bad news.
	if len(_As)<=3:
		moreTs = np.linspace(min(_Ts),max(_Ts),num=10); _Ts=[];_As=[];_Is=[]
		for _T in moreTs:
			try:
				rollingAction,_=action(V,_T)
			except BadlyIRDivergent as e:
				print(e)
				IRDiv=True
				rollingAction=None #Same steps as above.
				
			if rollingAction is not None and rollingAction>0:
				_As.append(rollingAction)
				_Ts.append(_T)
				_Is.append(b*integrate.trapezoid(Integrand(_T), Ts)) #COULD MAKE MORE ACCURATE!
		_Ts = np.array(_Ts); _As = np.array(_As); _Is = np.array(_Is)
		
		if len(_As)<=3:
			return None, None, tc, 10
	

	#Either no nucleation or Ts have started to high.
	if max(_Is)<0.34:
		print(_Is)
		if plot:
			plt.plot(_Ts, _As)
			plt.xlabel('Temperature'); plt.ylabel('A(T)')
			debug_plot(name="debug", overwrite=False)
			#plt.show()
			plt.plot(_Ts, _Is)
			plt.xlabel('Temperature'); plt.ylabel('I(T)')
			debug_plot(name="debug", overwrite=False)
			#plt.show()
		if IRDiv:
			return None, None, tc, 11 #PT failed probably due to IR divergences.
		else:
			return None, None, tc, 5 #PT failed as it just fails to nucleate in time.

	interpolator = interpolate.Akima1DInterpolator(_Ts,_Is)
	#NOTE ERROR HERE FROM NOT BEING ABLE TO INTEGRATE ALL THE WAY TO TC!! Should be small from exponential suppression.

	moreTs = min(_Ts)+(max(_Ts)-min(_Ts))*np.linspace(0, 1,num=500)**2

	if plot:
		plt.plot(moreTs, [interpolator(_T) for _T in moreTs])
		plt.plot(_Ts, _Is)
		plt.xlabel('Temperature'); plt.ylabel('I(T)')
		debug_plot(name="debug", overwrite=False)
		#plt.show()
		
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
			debug_plot(name="debug", overwrite=False)
			#plt.show()
			
			interp = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(moreTs)
			plotAs(_As,_Ts)#Comparing with original data
			plt.plot(moreTs,interp)#Check if the interpolator is doing well.
			plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
			debug_plot(name="debug", overwrite=False)
			#plt.show()
			print(res)
			
			tn=res.x[0]
			xs=np.linspace(-5,V.fSIGMA*Potential2.SIGMULT,num=100)
			plt.plot(xs,V.V(xs)-V.V(0),label=f"Vtree")

			plt.plot(xs,V.Vtot(xs,tc)-V.Vtot(0,tc),linestyle='-.',label=f"T={round(tc,4)} Tc Vtot")
			plt.plot(xs,V.Vtot(xs,tn)-V.Vtot(0,tn),linestyle='-.',label=f"T={round(tn,4)} Tn Vtot")
			plt.plot(xs,V.Vtot(xs,tn-2)-V.Vtot(0,tn-2),linestyle='-.',label=f"T={round(tn,4)-2} Vtot")
			
			plt.plot(xs,V.V1T(xs,tc)-V.V1T(0,tc),linestyle='--',label=f"T={round(tc,4)} Tc V1T")
			plt.plot(xs,V.V1T(xs,tn)-V.V1T(0,tn),linestyle='--',label=f"T={round(tn,4)} Tn V1T")
			plt.plot(xs,V.V1T(xs,tn-2)-V.V1T(0,tn-2),linestyle='--',label=f"T={round(tn,4)-2} V1T")
			
			if V.Polyakov:
				plt.plot(xs,V.VGluonic(xs,tc)-V.VGluonic(0,tc),linestyle=':',label=f"T={round(tc,4)} Tc VGluonic")
				plt.plot(xs,V.VGluonic(xs,tn)-V.VGluonic(0,tn),linestyle=':',label=f"T={round(tn,4)} Tn VGluonic")
				plt.plot(xs,V.VGluonic(xs,tn-2)-V.VGluonic(0,tn-2),linestyle=':',label=f"T={round(tn,4)-2} VGluonic")
		
			plt.legend()
			plt.xlabel('sigma')
			plt.ylabel('V')
			debug_plot(name="debug", overwrite=False)
			#plt.show()
			
					
		spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))
		
		#Checking to see if the spline doesn't oscillate.
		if all(spl.derivatives(_Ts)[1]>0):
			if not IRDivSmoothing:
				_,DivFrac = action(V, res.x[0])
				return res.x[0], spl, tc, DivFrac
			else:
				return res.x[0], spl, tc, 0
		else: 
			spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=1, s=len(_Ts)+np.sqrt(2*len(_Ts)))
			if plot:
				plotAs(_As,_Ts)#Comparing with original data
				plt.plot(moreTs,interp)#Check if the interpolator is doing well.
				plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
				debug_plot(name="debug", overwrite=False)
				#plt.show()
				print(res)
			if all(spl.derivatives(_Ts)[1]>0):
				if not IRDivSmoothing:
					_,DivFrac = action(V, res.x[0])
					return res.x[0], spl, tc, DivFrac
				else:
					return res.x[0], spl, tc, 0
			else:
				return None, None, tc, 17
		
	#If previous Tn failed, trying again with a new method.
	else:
		res = optimize.minimize(lambda T: abs(interpolator(T) - 0.34), (narrowRegion[0]+narrowRegion[-1])/2, bounds=[(min(narrowRegion), max(narrowRegion))],method='Nelder-Mead',tol=1e-3)
		if res.success and res.fun <=0.1:
			if plot: 
				print(f"Tn = {res.x[0]}, Minimisation method Nelder-Mead")
				print(res)
				plt.plot(_Ts, [interpolator(_T) for _T in _Ts])
				plt.xlabel('Temperature'); plt.ylabel('I(T)')
				debug_plot(name="debug", overwrite=False)
				#plt.show()
					
				interp = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(moreTs)
				plotAs(_As,_Ts)#Comparing with original data
				plt.plot(moreTs,interp)#Check if the interpolator is doing well.
				plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
				debug_plot(name="debug", overwrite=False)
				#plt.show()
				
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
				if not IRDivSmoothing:
					_,DivFrac = action(V, res.x[0])
					return res.x[0], spl, tc, DivFrac
				else:
					return res.x[0], spl, tc, 0
			else: 
				spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=1, s=len(_Ts)+np.sqrt(2*len(_Ts)))
				if plot:
					plotAs(_As,_Ts)#Comparing with original data
					plt.plot(moreTs,interp)#Check if the interpolator is doing well.
					plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
					debug_plot(name="debug", overwrite=False)
					#plt.show()
					print(res)
				if all(spl.derivatives(_Ts)[1]>0):
					if not IRDivSmoothing:
						_,DivFrac = action(V, res.x[0])
						return res.x[0], spl, tc, DivFrac
					else:
						return res.x[0], spl, tc, 0
				else:
					return None,None,tc,17
		
		print(res)
		if plot:
			print('abject failure')
			plt.plot(_Ts, np.array(_As)/np.array(_Ts))
			plt.plot(np.linspace(_Ts[0],_Ts[-1]), (interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(np.linspace(_Ts[0],_Ts[-1]))))
			debug_plot(name="debug", overwrite=False)
   			#plt.show()
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
		debug_plot(name="debug", overwrite=False)
		#plt.show()
	
	except ValueError as e:
		print(e)


	
#Compute beta/H for gravitational wave signals.
def beta_over_H(V, Tn, act):
	return act.derivatives(Tn)[1]*Tn
	
def alpha(V, Tn):
	minima = V.findminima(Tn)
	delV = abs(V.Vtot(0,Tn) - V.Vtot(minima,Tn))
	ddelVdT = abs(V.dVdT(0,Tn) - V.dVdT(minima,Tn))
	
	return (30/(V._g_star(Tn)*np.pi**2*Tn**4)) * (-delV + Tn*ddelVdT/4)
	
	
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
	debug_plot(name="debug", overwrite=False)
	#plt.show()
	

