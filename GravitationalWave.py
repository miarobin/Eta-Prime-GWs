import config
from cosmoTransitions.tunneling1D import SingleFieldInstanton
from cosmoTransitions.tunneling1D import PotentialError
from cosmoTransitions import helper_functions
from functools import reduce
import Potential2
from scipy import interpolate, optimize, integrate
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import traceback
if config.PLOT_RUN:
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


def IRProblem():
	return None

#Calculate the action S3 for a given temperature T using CosmoTransitions SingleFieldInstanton class.
def action(V,T, plot=False):
    prnt = config.PRNT_RUN
    #First find & classify the true & false minima
    vT = V.findminima(T)
    IRDivSmoothing = config.IRDIVSMOOTHING
    
    #If rhs is none at this point it means there's only one minima. Return action as None.
    if vT == None:
        print('No vT found')
        return None, 0
    
    true_min = min([0,vT], key=lambda phi: V.Vtot(phi,T))
    false_min = max([0,vT], key=lambda phi: V.Vtot(phi,T))
    
    if prnt: print(f'True Min = {true_min}, False Min = {false_min}, T = {T}')
    
 
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
        _V = lambda sigma: V.Vtot(sigma,T)
 
 
    #DO SOME PLOTS!
    if plot:
        plt.plot(sigmas,V.Vtot(sigmas,T)/V.fSIGMA**3-V.Vtot(0,T)/V.fSIGMA**3, label='Vtot')
        if IRDivSmoothing:
            plt.plot(_sigmas,weights,label='weights')
        plt.plot(sigmas,_V(sigmas)/V.fSIGMA**3-_V(0)/V.fSIGMA**3,label='Weighted Vtot')
        #plt.plot(sigmas,-pSig(sigmas)/V.fSIGMA,label='Sigma effective')
        #plt.plot(sigmas,-pPi(sigmas)/V.fSIGMA,label='Pi effective')
        #plt.plot(sigmas,V.MSq['Sig'][0](sigmas,T)/V.fSIGMA**2-V.MSq['Sig'][0](0,T)/V.fSIGMA**2,label='sig-mass')
        #plt.plot(sigmas,V.MSq['Pi'][0](sigmas,T)/V.fSIGMA**2-V.MSq['Sig'][0](0,T)/V.fSIGMA**2,label='pi-mass')
        #plt.ylim = (_V(V.fSIGMA)/V.fSIGMA**4-_V(0)/V.fSIGMA**4-0.01, +0.01)
        plt.title(f'T={T}')
        plt.legend()
        debug_plot(name="debug", overwrite=False)
  
 
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
	
	
#Plots actions as a function of T
def plotAs(As, Ts):
	_Ts = [T for T,A in sorted(zip(Ts,As))]
	_As = [A for T,A in sorted(zip(Ts,As))]
	plt.plot(_Ts, np.array(_As)/np.array(_Ts))




#Finds an interpolated function of 3D Action/T as a function of T, and Tn if it exists.
	#NOTE Print just shows you all of the individual broken points in detail so it can be manually fixed.
def grid(V, tc=None, ext_minT=None):
	IRDivSmoothing = config.IRDIVSMOOTHING
	plot = config.PLOT_RUN
	prnt = config.PRNT_RUN
	
	### SETTING THE RANGE OF TEMPERATURES TO SCAN OVER ###
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
	if ext_minT is None:
		minTy = optimize.minimize(lambda T: abs(V.d2VdT2(0,T)-config.TOL*V.fSIGMA**4)/V.fSIGMA**2,tc*.8, bounds=[(tc*.70,maxT)])
	else:
		if ext_minT<maxT:
			minTy = optimize.minimize(lambda T: abs(V.d2VdT2(0,T)-config.TOL*V.fSIGMA**4)/V.fSIGMA**2,(ext_minT+maxT)/2, bounds=[(ext_minT,maxT)])
		else:
			return None, None, tc, 23	

	if prnt: print(f'minTy = {minTy}')

	if minTy.fun/V.fSIGMA**2<1:
		#Sometimes minTy is a terrible estimate so manually setting a minimum based on where we cutoff the noise monitoring. 
		if ext_minT is not None:
			if prnt: print(f'extmin={ext_minT}')
			minT = max(minTy.x[0],tc*.75,ext_minT) 
		else:
			minT = max(minTy.x[0],tc*.75)
	else:
		return None, None, tc, 2
	if prnt: print(f'maximum T = {maxT}, minimum T = {minT}')
	

	#Some potential plots around the important area.
	if plot:
		xs=np.linspace(-5,V.fSIGMA*config.SIGMULT,num=100)
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


	#Setting sample size based on temperature jumps.
	if (maxT - minT)/50>0.5:
		numberOfEvaluations = 50
	else:
		numberOfEvaluations = 30
	#COARSE SAMPLE to find a sensible-ish minT and reduce number of calculations.
	Sample_Ts = np.linspace(minT, maxT, num=numberOfEvaluations)
	for i,_T in enumerate(Sample_Ts):
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
	if (maxT-minT)/(numberOfEvaluations+35) > 0.075: 
		Sample_Ts = moreTs = minT+(maxT-minT)*np.linspace(0, 1,num=numberOfEvaluations+35); As = []; Ts = []; IRDiv = False
	else:
		Sample_Ts = moreTs = minT+(maxT-minT)*np.linspace(0, 1,num=numberOfEvaluations+5); As = []; Ts = []; IRDiv = False

	for i,_T in enumerate(Sample_Ts):
		try:
			if plot and i%15==0:
				#Plots perturbativity every 5th evaluation.
				rollingAction,_ = action(V, _T,plot=True)
			else:
				rollingAction,_ = action(V, _T)
		except BadlyIRDivergent as e:
			if prnt: print(e)
			IRDiv=True #If the PT fails to nucleate because of IR divergences, we return a different error code.
			rollingAction=None
			
		if rollingAction is not None and rollingAction>0:
			As.append(rollingAction)
			Ts.append(_T)
			if prnt: print(f'Temperature {_T}, Action {rollingAction}, S/T = {rollingAction/_T}')

			b = 12*np.pi* (30/(V._g_star(tc)*np.pi**2))**2 * 1/(2*np.pi)**(3/2)
			weightContrib = b * (1/_T)**2 * (rollingAction/_T)**(3/2) * np.exp(-rollingAction/_T + 4*np.log(MPl))
			if prnt: print(f'Contribution to Weight: {weightContrib}')

			if weightContrib < 1e-20: #Stop if the (way overconservative) contribution to the weight is too small.
				break

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



	### WEIGHT FUNCTION ###
	#NOTE TO FUTURE SELVES: defining SM part of g_star at the CRITICAL TEMPERATURE.
	b = 12*np.pi* (30/(V._g_star(tc)*np.pi**2))**2 * 1/(2*np.pi)**(3/2) #Note transfer of MPl to following line to preserve precision
	def Integrand(T, intAs, intTs):
		return np.array([(1/intTs[i])**2 * (intAs[i]/intTs[i])**(3/2) * np.exp(-intAs[i]/intTs[i] + 4*np.log(MPl)) * (1/T - 1/intTs[i])**3 if T<intTs[i] else 0 for i in range(len(intTs))])


	#Previous interpolator just interpolates S_3/T but new interpolator interpolates the integrand of eq 3.10 of 2309.16755.
	Ts = np.array(Ts); As = np.array(As)
	Is = [b*integrate.trapezoid(Integrand(T, As, Ts), Ts) for T in Ts]

	if prnt: print(Is)
	if prnt: print(Ts)

	ICutoff = 150
	_Is = np.array([I for I, T, A in zip(Is,Ts,As) if I<ICutoff])
	_Ts = np.array([T for I, T, A in zip(Is,Ts,As) if I<ICutoff])
	_As = np.array([A for I, T, A in zip(Is,Ts,As) if I<ICutoff])
	
	if max(Is) > ICutoff and max(_Is)<0.34:
		print('HELLO WE MADE IT')
		if prnt: print(Is)
		if prnt: print(Ts)
		
		if prnt: print(_Is)
		if prnt: print(_Ts)
		if abs(max(_Is))<config.TOL:
			return None, None, tc, 9 #An action has been found, but not less than I=ICutoff.

		#Stepsize is too small so now iterating between the two values.
		#1: find the lowest two values with Is above 150.
		jminus = np.where(np.array(Is)>ICutoff)[0][-1]
		jplus = np.where(np.array(Is)<0.34)[0][0]
		checkBetween = np.linspace(Ts[jminus], Ts[jplus], num=20, endpoint=False)
	
		for i,_T in enumerate(checkBetween):
			try:
				rollingAction,_ = action(V,_T)
				if prnt and rollingAction is not None: print(f'Temperature {_T}, Action {rollingAction}, S/T = {rollingAction/_T}')
			except BadlyIRDivergent as e:
				if prnt: print(e)
				rollingAction=None 
			if rollingAction is None or rollingAction<0:
				return None, None, tc, 12 #If this happens, kill the code and look at the point individually.

			
			_Ts = np.insert(_Ts, i, _T)
			_As = np.insert(_As, i, rollingAction)
		

		#Recalculating weights...
		_Is = np.array([b*integrate.trapezoid(Integrand(_T, _As, _Ts), _Ts) for _T in _Ts])
		#Try again...
		_Is = np.array([I for I, T, A in zip(_Is,_Ts,_As) if I<ICutoff])
		_Ts = np.array([T for I, T, A in zip(_Is,_Ts,_As) if I<ICutoff])
		_As = np.array([A for I, T, A in zip(_Is,_Ts,_As) if I<ICutoff])
			

	if len(_As)==0:
		return None, None, tc, 9 #An action has been found, but not less than I=100. Not sure exactly what this means but it's bad news.
	if len(_As)<=3:
		moreTs = np.linspace(min(_Ts),max(_Ts),num=10); _Ts=[];_As=[];_Is=[]
		for _T in moreTs:
			try:
				rollingAction,_=action(V,_T)
			except BadlyIRDivergent as e:
				IRDiv=True
				rollingAction=None #Same steps as above.
				
			if rollingAction is not None and rollingAction>0:
				_As.append(rollingAction)
				_Ts.append(_T)
				_Is.append(b*integrate.trapezoid(Integrand(_T,_As,_Ts), Ts))
		_Ts = np.array(_Ts); _As = np.array(_As); _Is = np.array(_Is)
		
		if len(_As)<=3:
			return None, None, tc, 10
	

	#Either no nucleation or Ts have started to high.
	if max(_Is)<0.34:
		if prnt: print(_Is)
		if plot:
			plt.plot(_Ts, _As)
			plt.xlabel('Temperature'); plt.ylabel('A(T)')
			debug_plot(name="debug", overwrite=False)
			
			plt.plot(_Ts, _Is)
			plt.xlabel('Temperature'); plt.ylabel('I(T)')
			debug_plot(name="debug", overwrite=False)
			
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
			
			
			interp = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(moreTs)
			plotAs(_As,_Ts)#Comparing with original data
			plt.plot(moreTs,interp)#Check if the interpolator is doing well.
			plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
			debug_plot(name="debug", overwrite=False)
			
			print(res)
			
			tn=res.x[0]
			xs=np.linspace(-5,V.fSIGMA*config.SIGMULT,num=100)
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
			
			
					
		spl = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))
		
		#Checking to see if the spline doesn't oscillate.
		if all(spl.derivatives(_Ts)[1]>0):#maybe moreTs?
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
				
					
				interp = interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(moreTs)
				plotAs(_As,_Ts)#Comparing with original data
				plt.plot(moreTs,interp)#Check if the interpolator is doing well.
				plt.xlabel('Temperature'); plt.ylabel('S(T)/T')
				debug_plot(name="debug", overwrite=False)
				

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
					
					print(res)
				if all(spl.derivatives(_Ts)[1]>0):
					if not IRDivSmoothing:
						_,DivFrac = action(V, res.x[0])
						return res.x[0], spl, tc, DivFrac
					else:
						return res.x[0], spl, tc, 0
				else:
					return None,None,tc,17
		
		if prnt: print(res)
		if plot:
			print('abject failure')
			plt.plot(_Ts, np.array(_As)/np.array(_Ts))
			plt.plot(np.linspace(_Ts[0],_Ts[-1]), (interpolate.UnivariateSpline(_Ts,_As/_Ts,k=2, s=len(_Ts)+np.sqrt(2*len(_Ts)))(np.linspace(_Ts[0],_Ts[-1]))))
			debug_plot(name="debug", overwrite=False)
			
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
		if not config.PLOT_RUN: debug_plot(name="debug", overwrite=False)
		else: plt.show()
	
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
	V=config.Potential(0,*[0.2, 2, 1, 100**2],1,4,1)
	
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
	

