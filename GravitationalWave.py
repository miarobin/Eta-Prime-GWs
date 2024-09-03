from cosmoTransitions.tunneling1D import SingleFieldInstanton
from cosmoTransitions.tunneling1D import PotentialError
from cosmoTransitions import helper_functions
from functools import reduce
import Potential
from scipy import interpolate, optimize
import numpy as np
import matplotlib.pyplot as plt
import csv

#Terminal Colour Escape Sequences
RESET = "\033[0m"  # Reset all formatting
RED = "\033[31m"   # Red color
GREEN = "\033[32m" # Green color
CYAN = "\033[36m" # Cyan color

#Calculate g_star
data_array = np.loadtxt('gstar_data.dat')
Ts = data_array[:,0]
g_stars = data_array[:,1]
_g_star = interpolate.interp1d(Ts, g_stars)
##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2


#Calculate the action S3 for a given temperature T using CosmoTransitions SingleFieldInstanton class.
def action(V,T,prnt=False):
	#First find & classify the true & false minima
	lhs, rhs = V.findminima(T)
	
	#If rhs is none at this point it means there's only one minima. Return action as None.
	if rhs == None: return None
	
	true_min = min([lhs,rhs], key=lambda h: V.Vtot(h,T))
	false_min = max([lhs,rhs], key=lambda h: V.Vtot(h,T))

	try:
		#Initialise instanton in CosmoTransitions.
		Instanton = SingleFieldInstanton(true_min, false_min, lambda h: V.Vtot(h,T), alpha=2)
		Profile = Instanton.findProfile()
		
		#Find the action & return it.
		action = Instanton.findAction(Profile)
		if action<0: 
			if prnt: print("Negative Action")
			return None
		else: 
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
	
	

def plotV(V, Ts):
	for T in Ts:
		plt.plot(np.linspace(-3.5*v,2,num=100),V.Vtot(np.linspace(-3.5*v,2,num=100),T),label=f"T={T}")
	plt.legend()
	plt.show()
	
def plotAs(As, Ts):
	_Ts = [T for T,A in sorted(zip(Ts,As))]
	_As = [A for T,A in sorted(zip(Ts,As))]
	plt.plot(_Ts, np.array(_As)/np.array(_Ts))
	plt.plot(_Ts, np.ones((len(_Ts)))*140)
	plt.show()



#Finds an interpolated function of 3D Action/T as a function of T, and Tn if it exists.
def grid(V, tc=None, prnt=False):

	#Range of T's to consider.
	if tc==None:
		tc = V.criticalT(prnt=prnt)
		if prnt: print(f"Tc = {tc}")
		if tc == None:
			return None, None, 1
	print(tc)
	maxT = tc*0.9998
	#maxT = tc*0.96

	#Set up for the scan of S3 against T:
	stepSize = .15; bounds = []; flare = []; guess = None
	#First point to find if we go up or down
	T1 = maxT; T2 = maxT-stepSize
	A1=action(V,T1); A2=action(V,T2)
	if prnt: print(f"A1 = {A1}, A2 = {A2}, T1 = {T1}, T2 = {T2}")
	if A1 is None or A2 is None:
		if A1 is None:
			T1+=0.5; A1=action(V,T1,prnt=prnt)
		if A2 is None:
			T2+=0.5; A2=action(V,T2,prnt=prnt)
		if A1 is None or A2 is None:
			if prnt: plotV(V, [T1,T2])
			return None, None, 2
	#Bank of Ts and corresponding As
	Ts = [T1, T2]
	As = [A1, A2]
	
	#ASCEND
	if A1/T1<=140:
		#A variable stepsize...
		miniStepSize = (tc-T1)*0.2
		
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
			miniStepSize=(tc-Tb)*0.2
			#If it gets too small we're probably in trouble.?
			if miniStepSize < 0.00001: 
				return None, None, 3
						
			Ab = action(V,Tb+miniStepSize); Tb += miniStepSize
			if prnt: print(f"Aa = {Aa}, Ab = {Ab}, Ta = {Ta}, Tb = {Tb}")
			if Ab is None:
				#Get Rodrigo to calculate this point.
				if prnt: 
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
				if prnt: 
					plotV(V, Ts+[Tb,0,400])
					plotAs(As+[Ab],Ts+[Tb])
				return None,None, 5 
			else:
				As.append(Ab); Ts.append(Tb)
					
		#Ascending stopping criteria achieved if code exits while loop. Flare around temperature to more accurately calculate gradient.
		bounds = [(Ta*0.99,Tb*1.01)]; guess = Ta + miniStepSize/2
		flare = [Tb + (tc - Tb)*adj for adj in [-1.5,-1.1,-0.65,-0.5,-0.2,-0.1,-0.05,0.1,0.05] if Tb + (tc - Tb)*adj < tc] 
		
	
	#DESCEND
	elif A1/T1 >= 140 and A2/T2 >= 140: 
	#Fixed Stepsize since we don't need to sneak this time
		while A2/T2>=140 and A1/T1>=140:
			#Use newton's method to descend the curve
			m = (A1/T1 - A2/T2)/(T1-T2); c = (A2*(T1/T2) - A1*(T2/T1))/(T1-T2)
			T1 = (140-c)/m; T2=T1-stepSize
			
			if T1<10 or T2<10: #Basically no way there's a PT now.
				if prnt: plotAs(As,Ts)
				return None, None,6
			
			A1=action(V,T1,prnt=prnt)
			A2=action(V,T2,prnt=prnt)
			
			if prnt: print(f"A1 = {A1}, A2 = {A2}, T1 = {T1}, T2 = {T2}")

		
			#First check you can find an action...
			if A1 is not None and A2 is not None:
				#Add findings to the 'bank'
				Ts = Ts + [T1,T2]
				As = As + [A1,A2]
			
			else:
				if A1 is None:
					A1=action(V,T1+0.5,prnt=prnt); T1+=0.5
				if A2 is None:
					A2=action(V,T2+0.5,prnt=prnt); T2+=0.5
				if A1 is None or A2 is None:
					if prnt: 
						plotV(V, Ts+[T1,T2,tc])
						plotAs(As,Ts)
					return None, None,7
				else:
					Ts = Ts + [T1,T2]
					As = As + [A1,A2]
					if prnt: print(f"A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
		
			if (A1/T1) < (A2/T2) and A1/T1>=140:	
				#We have passed the minima of the S3/T at this point and not reached S3/T=140 so return none found

				if prnt: 
					print(f"A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
					plotAs(As + [A1,A2],Ts + [T1,T2])
				return None, None, 0
			


		#If we found Tc by descending, we'll want to do flare around descent values
		flare = [T1 + adj*stepSize for adj in [-0.5,-1.2,-2,-2.5,-3,-4,-6,-10]]
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
		if prnt:
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
	#plt.plot(np.array(_Ts), np.array(_As)/np.array(_Ts))
	#plt.plot(np.array(_Ts), np.ones((len(_Ts)))*140)
	#plt.show()

	interpolator = interpolate.UnivariateSpline(_Ts,np.array(_As)/np.array(_Ts),k=5, s=len(_Ts)+np.sqrt(2*len(_Ts))/2)

	if prnt: print(f"Bounds = {bounds}, guess = {guess}")
	res = 0
	try:
		res = optimize.minimize(lambda T: abs(interpolator(T) - 140.), guess, bounds=bounds,method='L-BFGS-B',tol=1e-2)
	except ValueError:
		return None, None, 12
	
	if res.success and res.fun <=5:
		if prnt: 
			print(f"Tn = {res.x[0]}, Nruns = {len(Ts)}")
			plotAs(As,Ts)
		return res.x[0], interpolator, 0
	else:
		res = optimize.minimize(lambda T: abs(interpolator(T) - 140.), guess, bounds=bounds,method='Nelder-Mead')
		if res.success and res.fun <=5:
			if prnt: 
				print(f"Tn = {res.x[0]}, Nruns = {len(Ts)}")
				plotAs(As,Ts)
				print(res)
			return res.x[0], interpolator, 0
		
		print(res)
		if prnt:
			plt.plot(_Ts, np.array(_As)/np.array(_Ts))
			plt.plot(np.linspace(_Ts[0],_Ts[-1]), (interpolator(np.linspace(_Ts[0],_Ts[-1]))))
			plt.plot(_Ts, np.ones((len(_Ts)))*140)
			plt.show()
		return None, None, 9
	
	#If you reach this point something's gone totally wrong...

	#T = np.linspace(T2,maxT, num=25)
	#plt.plot(v/T, np.log(140*np.ones((25))))
	#plt.plot(v/T, np.log(interpolator(T)))
	#plt.xlabel(f"$v/T$")
	#plt.ylabel(f"Log($S_3/T$)")
	#plt.show()
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


#Calculate the action S4 at zero temperature using CosmoTransitions SingleFieldInstanton class.
def euclidAction(V,prnt=False):
	#First find & classify the true & false minima
	lhs, rhs = V.findminima(1)
	
	if prnt:
		plt.plot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),V.Vtot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),100),label="Vtot at T=100GeV")
		plt.plot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),V.Vtot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),0),label="Vtot at T=0GeV")
		plt.plot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),V.V1_cutoff(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100)), label="V1")
		plt.plot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),V.V(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100)), label = "Vtree")
		plt.scatter([lhs,rhs],V.Vtot([lhs,rhs],0))
		plt.legend()
		plt.show()
	
	#If rhs is none at this point it means there's only one minima. Return action as None.
	if rhs == None: return None
	
	true_min = min([lhs,rhs], key=lambda h: V.Vtot(h,1))
	false_min = max([lhs,rhs], key=lambda h: V.Vtot(h,1))

	try:
		#Initialise instanton in CosmoTransitions.
		Instanton = SingleFieldInstanton(true_min, false_min, lambda h: V.Vtot(h,1), alpha=3)
		Profile = Instanton.findProfile()
		
		#Find the action & return it.
		action = Instanton.findAction(Profile)
		if action<0: 
			if prnt: print("Negative Action")
			return None
		else: 
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
	except RuntimeWarning:
		if prnt: print(RED + "CosmoTransitions has returned RuntumeWarning" + RESET)
		return None
	
#Compute beta/H for gravitational wave signals.
def beta_over_H(V, Tn, act):
	return act.derivatives(Tn)[1]*Tn
	
def alpha(V, Tn):
	minima = V.findminima(Tn)
	delV = abs(V.Vtot(minima[0],Tn) - V.Vtot(minima[1],Tn))
	ddelVdT = abs(V.dVdT(minima[0],Tn) - V.dVdT(minima[1],Tn))
	#Note g_star is defined at the top of the document.
	return (30/(_g_star(Tn)*np.pi**2*Tn**4)) * (-delV + Tn*ddelVdT)
	
	
def gravitationalWave(V):
	#This method is slow so we only want to run it once, then just pass through the interpolated function to everything else.
	Tn, grd, message = grid(V,prnt=True)
	
	#Calculate nucleation temperature
	if Tn == None:
		#No transition occurs
		return None
	#Calculate beta/H
	beta_H = beta_over_H(V, Tn, grd)
	#Calculate alpha
	alp = alpha(V, Tn)
	
	return(Tn, beta_H, alp)


def wallVelocity(V, alpha, T):
	#Calculating the wall velocity at temperature T. Assumes a constant temperature.
	
	#Jouget velocity.
	vJ = (1/np.sqrt(3)) * (1 + np.sqrt(3*alpha**2 + 2*alpha))/(1+alpha)
	#Radiative power
	rho_r = np.pi**2 * _g_star(T) * T**4 / 30
	
	lhs, rhs = V.findminima(T)
	if rhs is not None:
		v = np.sqrt(abs(V.Vtot(lhs,T) - V.Vtot(rhs,T))/(alpha*rho_r))  
		if v < vJ: return v
		else: return 1
	else:
		return None


def BubbleorNot(V, tc=None, prnt=False):
	#Range of T's to consider.
	if tc==None:
		tc = V.criticalT()
		if tc == None:
			return None, 1
	
	maxT = tc*0.985
	#maxT = tc*0.95

	print(f"Epsilon = {V.ep}, Delta = {V.gammaedelta()[1]}")
	#Set up for the scan of S3 against T:
	stepSize = 2.5; bounds = []; flare = []; guess = None
	#First point to find if we go up or down
	T1 = maxT; T2 = maxT-stepSize
	A1=action(V,T1); A2=action(V,T2)
	if prnt: print(f"A1 = {A1}, A2 = {A2}, T1 = {T1}, T2 = {T2}")
	if A1 is None or A2 is None:
		if A1 is None:
			A1=action(V,T1+0.5,prnt=prnt); T1+=0.5
		if A2 is None:
			A2=action(V,T2+0.5,prnt=prnt); T2+=0.5
		if A1 is None or A2 is None:
			if prnt: plotV(V, [T1,T2])
			return None, 2
	if prnt: print(f"A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
	#Bank of Ts and corresponding As
	Ts = [T1, T2]
	As = [A1, A2]
	

	if A1/T1<=140 or A2/T2 <=140:
		#Found a bubbling solution.
		return 1, 0
		
	#DESCEND
	elif A1/T1 >= 140 and A2/T2 >= 140: 
	#Fixed Stepsize since we don't need to sneak this time
		while A2/T2>=140 and A1/T1>=140:
			#Use newton's method to descend the curve
			m = (A1/T1 - A2/T2)/(T1-T2); c = (A2*(T1/T2) - A1*(T2/T1))/(T1-T2)
			T1 = (140-c)/m; T2=T1-stepSize
			
			if T1<10 or T2<10: #Basically no way there's a PT now.
				if prnt: plotAs(As,np.array(Ts))
				_Ts = [T for T,A in sorted(zip(Ts,As))]
				_As = [A for T,A in sorted(zip(Ts,As))]
				interpolator = interpolate.UnivariateSpline(_Ts,np.array(_As)/np.array(_Ts))
				res = optimize.minimize(lambda T: abs(interpolator(T) - 140.), (max(Ts)-min(Ts))/2, method='L-BFGS-B',tol=1e-2)
				return 0, res.x[0]
			
			A1=action(V,T1,prnt=prnt)
			A2=action(V,T2,prnt=prnt)
			
			if prnt: print(f"A1 = {A1}, A2 = {A2}, T1 = {T1}, T2 = {T2}")
		
			#First check you can find an action...
			if A1 is not None and A2 is not None:
				#Add findings to the 'bank'
				Ts = Ts + [T1,T2]
				As = As + [A1,A2]
			
			else:
				if A1 is None:
					A1=action(V,T1+0.5,prnt=prnt); T1+=0.5
				if A2 is None:
					A2=action(V,T2+0.5,prnt=prnt); T2+=0.5
				if A1 is None or A2 is None:
					if prnt: 
						plotV(V, Ts+[T1,T2,tc])
						plotAs(As,Ts)
					return None, 7
				else:
					Ts = Ts + [T1,T2]
					As = As + [A1,A2]
					if prnt: print(f"A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
		
			if (A1/T1) < (A2/T2) and A1/T1>=140:	
				#We have passed the minima of the S3/T at this point and not reached S3/T=140 so return none found
				if prnt: 
					print(f"Passed minima with A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
					plotAs(As + [A1,A2],v/np.array(Ts + [T1,T2]))
				_Ts = [T for T,A in sorted(zip(Ts,As))]
				_As = [A for T,A in sorted(zip(Ts,As))]
				interpolator = interpolate.UnivariateSpline(_Ts,np.array(_As)/np.array(_Ts))
				res = optimize.minimize(lambda T: abs(interpolator(T) - 140.), (max(Ts)-min(Ts))/2, method='L-BFGS-B',tol=1e-2)
				return 0, res.x[0]
			
		#If we exit the while loop, we've passed 140.
		if prnt: print(f"Crossed 140 with A1/T1 = {(A1/T1)}, A2/T2 = {(A2/T2)}")
		return 1, 0
		
	elif A1/T1 >= 140 and A2/T2 <= 140: 
		return 1, 0

	else:
		if prnt: print(f"A1/T1 = {A1/T1}, A2/T2 = {A2/T2}")
		return None, 8
		
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

def ActionPlot(V, save=False):
	tc = V.criticalT()
	print(f"Critical temperature = {tc}")
	lhs, rhs = V.findminima(0)
	print(f"lhs = {lhs}, rhs = {rhs}")
	v_star = V.vStar()
	Ts = [0,50,100,200]
	for T in Ts:
		plt.plot(np.linspace(-3,1, num=100), V.Vtot(v*np.linspace(-3,1, num=100),T)/v**4-V.Vtot(v,T)/(v**4),label=f"T = {T} GeV")
	plt.legend()
	plt.show()
	
	print(f"delta = {V.gammaedelta()[1]}")
	print(f"gamma_e = {V.gammaedelta()[0]}")

	v_Ts = [1.9500000000000002,2.,2.05,2.1,2.1500000000000004,2.2,2.25,
			2.3,2.35,2.4000000000000004,2.45,2.5,2.55,2.6,2.6500000000000004,
			2.7,2.75,2.8,2.85,2.9000000000000004,2.95,3.,3.05,3.1,3.1500000000000004,
			3.2,3.25,3.3,3.35,3.4,3.45,3.5,3.5500000000000003,3.6,3.65,3.7,3.75,3.8000000000000003]
	#v_Ts = np.linspace(1.95, 3.8, num=100)
	As = [action(V, v/v_T) for v_T in v_Ts]

	if save:
		file_path = f'actionValues.csv'
		save_arrays_to_csv(file_path, ['As','v_Ts'], As, v_Ts)

	
	plt.plot(v_Ts, np.array(As)*np.array(v_Ts)/v)
	plt.xlabel(r"$v/T$")
	plt.ylabel("Action/T")
	
	'''with open('ActionRo2.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		x=[];y=[]
		for i,row in enumerate(spamreader):
			if i !=0:
				x.append(float(row[0]))
				y.append(float(row[1]))
				
		plt.plot(x,y, label='Rodrigos Plot')

	plt.title(f"eps = {V.ep}, beta = {round(V.beta,2)},delta = {round(V.gammaedelta()[1],2)}, gamma4 = {V.g4}")
	plt.legend()'''
	
	plt.show()

if __name__ == "__main__":
	## Choose epsilon and gamma4 to slot into the potential, and the temperature to evaluate at.
	eps = 0.04; g4 = 1.4; delta = -0.08; beta = np.sqrt(0.1)
	print(f"$\gamma_a$ = {Potential.gammaa(eps,g4,delta)}")
	#The Potential Object.
	V = Potential.Potential(eps,g4,Potential.gammaa(eps,g4,delta),beta,higgs_corr=True,loop=True)
	print(f"Thermal Minimum:{optimize.minimize(lambda h:V.V1T(h,150),-246).x}")
	print(f"Tree level maximum: {optimize.minimize(lambda h:-V.V(h), -246).x}")
	#Tn, beta_H, alp = gravitationalWave(V)
	#print(f"Tn = {Tn}, beta/H = {beta_H}, alpha = {alp}")

	ActionPlot(V,save=True)

	#print(f"Eps = {eps}, g4 = {g4}, delta = {delta}, beta = {round(beta,2)}")
	#print(f"S_4 = {euclidAction(V,prnt=True)}")



	
	#Calculate & print v_star.
	v_star = V.vStar()
	print(f'$v_*$ = {v_star}')
	
	'''
	#PLOTTING THE POTENTIAL
	#lhs, rhs = V.findminima(0)
	#maxima = optimize.minimize(lambda h: -V.V(h), lhs + 1, method='Nelder-Mead',bounds=[(lhs,rhs)]).x[0]
	#plt.plot(np.linspace(-2,0.5, num=100)-maxima/v, V.V(v*np.linspace(-2,0.5, num=100))/v**4 - V.V(maxima)/v**4,label='Tree')
	#plt.plot(np.linspace(-2,0.5, num=100), V.V1T(v*np.linspace(-2,0.5, num=100),10)/v**4-V.V(rhs)/v**4,label='Thermal')

	'''
	Ts = [4*np.pi*v]
	for T in Ts:
		plt.plot(np.linspace(-2.5,0.5, num=100)-v_star/v, V.Vtot(v*np.linspace(-2.5,0.5, num=100),T)/v**4-V.Vtot(v_star,T)/(v**4),label=f"T = {T} GeV")
	'''
	with open('RaBSM0.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		x=[];y=[]
		for i,row in enumerate(spamreader):
			if i !=0:
				x.append(float(row[0]))
				y.append(float(row[1]))
				
		plt.plot(x,y, label='Rodrigo T = 100 GeV')
	'''
	
	plt.title(f"$\epsilon = ${eps}, $\gamma_a = ${0.75},\n $m_h =${mh}, $m_W =${mW}, $m_Z =${mZ}, $m_t =${mt},\n $\lambda =${round(l,5)}, D = {round((2*mW**2 + mZ**2 + 2*mt**2)/(8*v**2),6)}")
	plt.xlabel(f"$h/v_h$")
	plt.ylabel(f"$V(h,T)/v_h^4$")
	plt.legend()
	plt.show()
	'''
	#LOOK AT WHERE THE MINIMA SIT
	print(f"Thermal Minimum:{optimize.minimize(lambda h:V.V1T(h,150),-246).x}")
	print(f"Tree level maximum: {optimize.minimize(lambda h:-V.V(h), -246).x}")
	
	
	#Looking at nucleation temperature to compare with Rodrigo
	#With beta = 0.1
	V0 = Potential.Potential(eps,g4,0.75,beta,msbar=True)
	Tn_0 = nucleationTemperature(V0, plot=True, label="With MSbar")
	#With beta = 0.05
	
	V1 = Potential.Potential(eps,g4,0.75,beta,msbar=False)
	Tn_1 = nucleationTemperature(V1, plot=True, label = "With Cut-off")'''
	'''
	
	with open('ActionPlotRo.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		x=[];y=[]
		for i,row in enumerate(spamreader):
			if i !=0:
				x.append(float(row[0]))
				y.append(float(row[1]))
				
		plt.plot(x,y, label='Rodrigos Plot')
	
	'''



