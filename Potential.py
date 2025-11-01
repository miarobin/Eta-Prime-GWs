import cosmoTransitions.finiteT as fT
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate, integrate
from itertools import takewhile
import os
from debug_plot import debug_plot

#FOR A LINEAR POTENTIAL (i.e. a symmetric and a non-symmetric phase)

##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2


class NotImplemented(Exception):
    pass


class Potential:

	def __init__(self, xi, muSig, lmb, kappa, m2Sig, N, F, muSSI=0, loop=False):
		#All the parameters needed to construct the potential.
		self.lmb = lmb
		self.kappa = kappa
		self.m2Sig = m2Sig
		self.muSig = muSig
		self.xi = xi
		self.loop = loop
		self.N = N
		self.F = F
		self.muSSI = muSSI
		#Looks like loops not needed for this calculation.

		#The coefficient of the determinant term as it's quite verbose:
		det_coef = 2*self.N * (self.F/self.N)*(self.F/self.N-1)*(1/np.sqrt(6))**(self.F/self.N)
		#Field dependent masses for fermions & bosons, their derivative wrt h (twice) and their respective DoF
		if self.F==3:
			self.mSq = {
				#Phi Mass	
				'Phi': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig
							- det_coef * self.muSig * phi**max([abs(self.F/self.N - 2),0]) 
							+ (1/6)*(3*self.kappa + 9*self.lmb) * phi**2,
						1.],
				#Eta Prime Mass
				'Eta': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig 
							+ det_coef * self.muSig * phi**max([abs(self.F/self.N - 2),0]) 
							+ (1/6)*(3*self.lmb + self.kappa) * phi**2,
						1.],
				#X Mass
				'X': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig - 3 * self.xi 
		  					+ 0.5 * det_coef * self.muSig * phi**max([abs(self.F/self.N - 2),0]) 
							+ (1/6)*(3*self.lmb + 3*self.kappa) * phi**2,
						8.],
				#Pi Mass
				'Pi': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig - 3 * self.xi 
		   					- 0.5 * det_coef * self.muSig * phi**max([abs(self.F/self.N - 2),0]) 
							+ (1/6)*(3*self.lmb + self.kappa) * phi**2,
						8.]
						}

		elif self.F==4:
			#Generic N, muSSI=!0:
			self.mSq = {
				#Phi Mass
				'Phi': [lambda phi, T: (3/8)*((self.kappa + 4*self.lmb)*phi**2) - self.m2Sig 
													+ (self.muSig/self.N**2) * 8**((-2+self.N)/self.N) * (-4+self.N) * phi**(4/self.N-2),
#													+ (1/12)*T**2 * (8*self.kappa + 17*self.lmb),
						1.],
				#Eta Prime Mass
				'Eta': [lambda phi, T: (1/8)*((self.kappa + 4*self.lmb)*phi**2) - self.m2Sig
													- (self.muSig/self.N**2) * 8**((-2+self.N)/self.N) * (-4+self.N) * phi**(4/self.N-2),
#													+ (1/12)*T**2 * (8*self.kappa + 17*self.lmb),
						1.],
				#X Mass
				'X': [lambda phi, T: (1/8)*((3*self.kappa + 4*self.lmb)*phi**2) - self.m2Sig
													+ (self.muSig/self.N) * (8**((-2 + self.N)/self.N) * phi**(4/self.N-2)),
#													+ (1/12)*T**2 * (8*self.kappa + 17*self.lmb),
						12.],
				#Pi Mass
				'Pi': [lambda phi, T: (1/8)*((self.kappa + 4*self.lmb)*phi**2) -  self.m2Sig
													- (self.muSig/self.N) * (8**((-2 + self.N)/self.N) * phi**(4/self.N-2)),
#													+ (1/12)*T**2 * (8*self.kappa + 17*self.lmb),
						12.],

				}


		else:
			raise NotImplemented(f"F={self.F} not implemented yet in mSq")
			
		
	def V(self,phi): #Comment which equation numbers!!!
		##The tree level, zero temperature potential.
		phi = np.array(phi)
		#if self.F == 3:
		#	return -(0.5*self.m2Sig) * phi**2 - self.muSig/(3*np.sqrt(6)) * phi**3 + 0.25 * (self.lmb/2 + self.kappa/6) * phi**4
		if self.F == 4:
			return -(0.5*self.m2Sig) * phi**2 + (self.lmb/8 + self.kappa/32) * phi**4 - 4**(3-3/self.N)/32 * self.muSig * phi**(4/self.N)
		
		else:
			raise NotImplemented(f"F={self.F} not implemented yet in V tree")


	def V1T(self,phi,T):
		phi = np.array(phi)
		if T==0:
			return np.zeros(phi.shape)
		
		if self.F == 3:
			return np.reshape((np.sum([n*Jb_spline((m2(phi,T)/T**2)) for m2, n in [self.mSq['Phi'],self.mSq['Eta'],
																		  			self.mSq['X'],self.mSq['Pi']]],axis=0))*T**4/(2*np.pi**2), phi.shape)
		elif self.F == 4:
			return np.reshape((np.sum([n*Jb_spline((m2(phi,T)/T**2)) for m2, n in [self.mSq['Phi'],self.mSq['Eta'],
																					self.mSq['Pi'],self.mSq['X']]],axis=0))*T**4/(2*np.pi**2), phi.shape)
		else:
			raise NotImplemented(f"F={self.F} not implemented yet in V1T")
		
	def VMedium(self, phi, l, T):
		phi = np.array(phi); l = np.array(l)
		integrals = []
		
		if T==0: return 0
		
		if self.N == 2:
			originalShape = np.shape(phi)
			if phi.shape==():
				phi=np.array([phi])

			eps_P = lambda x: np.array(np.sqrt(x**2 + (3.2*phi)**2))
	
			if l.shape!=():
				integrals=[]
				for _l in l:
					integrand = lambda x,i: x**2 * np.log(1 + 2*_l*np.exp(-eps_P(x)[i]/T) + np.exp(-2*eps_P(x)[i]/T))
					integrals.append(np.array([-4*self.F*T*integrate.quad(lambda x: integrand(x,i), 0, np.inf)[0]/(np.pi**2) for i in range(len(phi))]))
				return np.array(integrals)
			
			else:
				integrand = lambda x,i: np.reshape(np.array(x**2 * np.log(1 + 2*l*np.exp(-eps_P(x)[i]/T) + np.exp(-2*eps_P(x)[i]/T))),phi.shape)
				return np.reshape(np.array([-4*self.F*T*integrate.quad(lambda x: integrand(x,i), 0, np.inf)[0]/(np.pi**2) for i in range(len(phi))]),originalShape)

		if self.N==3:
			if self.F==4:
				return 0

	def VPoly(self, l, T):
		if T == 0:
			return 0
		if self.N==2:
			return -(210.5)**3 * T * (1-24*l**2 * np.exp(-858/T))


	def V1_cutoff(self, h):
		# One loop corrections to effective potential in cut-off regularisation scheme.
		#MAY NOT NEED THESE YET!
		h = np.array(h)
			
		return np.reshape(np.sum([n * (m2(h)**2 * (np.log(np.abs(m2(h)/m2(0)) + 1e-100) - 1.5) + 2*m2(h)*m2(0)) for m2, n in [self.mSq['W'],self.mSq['Z'],self.mSq['t'],self.mSq['h'],self.mSq['Wl'],self.mSq['Zl']]],axis=0)/(64.*np.pi**2), h.shape)
		

	def Vtot(self,phi,T, polyakov=True):
		#This finds the total (one-loop/tree level) thermal effective potential.
		phi = np.array(phi)
		
		if polyakov:
			gluonic = lambda l,_phi: self.VPoly(l, T) + self.VMedium(_phi, l, T)
	
			if phi.shape==():
				res = optimize.minimize(lambda l: gluonic(l, phi), 0.5, bounds=[(0,1)]).x[0]
				return self.V(phi) + self.V1T(phi,T).real + self.VPoly(res, T) + self.VMedium(phi, res, T)
			else:
				res = np.array([optimize.minimize(lambda l: gluonic(l, _ph), 0.5, bounds=[(0,1)]).x[0] for _ph in phi])
				
				return np.array([self.V(phi[i]) + self.V1T(phi[i],T).real + self.VPoly(res[i], T) + self.VMedium(phi[i], res[i], T) for i in range(len(phi))])
		
		return self.V(phi) + self.V1T(phi,T).real

	def dVdT(self,phi,T,eps=0.001):
		#Uses finite difference method to fourth order. Takes scalar h and T.
		return (self.Vtot(phi,T-2*eps) - 8*self.Vtot(phi,T-eps) + 8*self.Vtot(phi,T+eps) - self.Vtot(phi,T+2*eps)) / (12.*eps)
			
	def d2VdT2(self,phi,T,eps=0.001):
		#Uses finite difference method to fourth order. Takes scalar h and T.
		return (self.dVdT(phi,T-2*eps) - 8*self.dVdT(phi,T-eps) + 8*self.dVdT(phi,T+eps) - self.dVdT(phi,T+2*eps)) / (12.*eps)


	def fSigmaApprox(self):
		if self.F==4 and self.N==1 and (self.kappa + 4*self.lmb - self.muSig)>0:
			return 2*(2*self.m2Sig)**0.5 / (self.kappa + 4*self.lmb - self.muSig)**0.5
		elif self.F==4 and self.N>1:
			return 2*(2*self.m2Sig)**0.5 / (self.kappa + 4*self.lmb)**0.5
		else:
			return self.findminima(0,rstart=5000)

	def findminima(self,T,rstart=None,rcounter=1):
		#For a linear sigma model. Returns the minimum away from the origin if it exists, else None.
		if rstart == None:
			rstart = self.fSigma()*.8
		
		#Debugging statements
		#print(f'rstart={rstart}, T={T}, fPI={self.fSIGMA}')
		#print(f'min bound = {(0.5-.05*rcounter)*self.fSIGMA}, max bound = {self.fSIGMA*1.05}')
		
		#Roll down to minimum from the RHS:
		res = optimize.minimize(lambda X: self.Vtot(X, T), rstart,method='Nelder-Mead',bounds=[((0.5-.05*rcounter)*self.fSIGMA,self.fSIGMA*1.05)])
		#print(res)
		#print(f'closeness criteria = {abs(res.x[0]-(0.5-.05*rcounter)*self.fSIGMA)/self.fSIGMA}, rcounter={rcounter}')

		#Now check to see if the algorithm succeeded
		if not res.success or abs(res.x[0]-(0.5-.05*rcounter)*self.fSIGMA)/self.fSIGMA < TOL or res.x[0]<(0.5-.05*rcounter)*self.fSIGMA:
			#If so, try a new start closer to the axis to avoid overshooting.
			if rcounter<=4:
				if rstart is not None:
					#Closer to axis by 20%.
					return self.findminima(T,rstart=rstart*0.8,rcounter=rcounter+1)
				else: return None
			else:
				print(f'T={T}')
				return None
		#Check the roll didn't find the zero sigma minimum.
		elif res.x[0]<10:
			return None
		else: return res.x[0]

	def deltaV(self,T, rstart=None, num_res=False):
		#Finds the difference between the symmetric and broken minima.
		if rstart is not None: vT = self.findminima(T, rstart=rstart)
		else: vT = self.findminima(T)
		if vT is not None:
			return + self.Vtot(0, T) - self.Vtot(vT, T)
		else:
			if num_res:
				return 1e30
			else:
				return None
			



	def criticalT(self, guessIn=None,prnt=True,minT=None):
		prnt=True
		if self.tc is not None:
			return self.tc
		#Critical temperature is when delta V is zero (i.e. both minima at the same height) THIS HAS TO BE QUITE ACCURATE!
		
		#Scale with which we can compare potential magnitudes (think large values of sigma, V~sigma^4)
		scale = self.fSigma()
		
		if minT is not None:
			minTemp=minT
		else:
			minTemp = np.sqrt(scale)
					
		
		#First a coarse scan. Find the minimum deltaV from this initial scan, then do a finer scan later.
		Ts_init = np.linspace(minTemp,scale*2.,num=450); deltaVs_init=[]

		for T in Ts_init:
			#Computing the difference between symmetric and broken minima.
			deltaV =  self.deltaV(T, rstart=scale)
			
			#Basically starting from low T and increasing until deltaV flips signs (i.e. minima have crossed eachother)
			if deltaV is not None and deltaV > 0: deltaVs_init.append([T,deltaV])
			if deltaV is not None and deltaV < 0: break

		deltaVs_init=np.array(deltaVs_init)
		

		if prnt:
			ax=plt.subplot()
			for T,_ in deltaVs_init:
				ax.scatter(self.findminima(T),T)
				ax.scatter(0,T)
			ax.set_xlabel('RHS Minima'); plt.ylabel('T')
			plt.show()  

		if len(deltaVs_init)<3:
			print('Coarse scan finds nothing')
			return None
		#JUST taking deltaV's which are greater than zero BUT decreasing. Note the reason for this is often going further can confuse python later.
		
		#NOTE TO SELF! THIS MIGHT END UP BEING A PROBLEM!
		#j = list(takewhile(lambda x: np.concatenate(([0],np.diff(deltaVs_init[:,1])))[x]<=0, range(len(deltaVs_init[:,0])))); deltaVs_init=deltaVs_init[j]
		#k = list(takewhile(lambda x: deltaVs_init[x,1]>0, range(len(deltaVs_init[:,0]))))
		#print(k)

		#deltaVs_init=deltaVs_init[k]
		#This is the temperature with deltaV closest to zero:
		T_init = deltaVs_init[-1,0]
		if prnt:
			ax=plt.subplot()
			ax.plot(deltaVs_init[:,1], deltaVs_init[:,0])
			ax.set_xlabel('Delta V'); ax.set_ylabel('Temperature')
			plt.show()

			print(f"Coarse grain scan finds {T_init} being closest Delta V to 0")

		def plotV(V, Ts):
			ax=plt.subplot()
			for T in Ts:
				ax.plot(np.linspace(-10,self.fSIGMA*1.25,num=100),V.Vtot(np.linspace(-10,self.fSIGMA*1.25,num=100),T)-V.Vtot(0,T),label=f"T={T}")
				if self.findminima(T) is not None:
					ax.scatter(self.findminima(T), V.Vtot(self.findminima(T),T)-V.Vtot(0,T))
			plt.legend()
			plt.show()  
		def splitV(V, T):
			ax=plt.subplot()
			ax.plot(np.linspace(-10,self.fSIGMA*1.25,num=1000),V.Vtot(np.linspace(-10,self.fSIGMA*1.25,num=1000),T)-V.Vtot(0,T),label=f"VTot at Tc={T}")
			ax.plot(np.linspace(-10,self.fSIGMA*1.25,num=1000),V.V1T(np.linspace(-10,self.fSIGMA*1.25,num=1000),T)-V.V1T(0,T),label=f"V1T at Tc={T}")
			ax.plot(np.linspace(-10,self.fSIGMA*1.25,num=1000),V.V(np.linspace(-10,self.fSIGMA*1.25,num=1000))-V.V(0),label=f"Vtree")
			plt.legend()
			plt.show()  
			
		#Find delta V for a finer scan of temperatures & interpolate between them.
		Ts = np.linspace(T_init-10,T_init+10,num=150); deltaVs = np.array([[T, self.deltaV(T, rstart=scale*.8)] for T in Ts if self.deltaV(T,rstart=scale) is not None])
		
		if len(deltaVs)<5: return None #Catches if there are just not enough points to make a verdict.
		
		
		#Ensure each deltaV is decreasing with increasing T.
		deltaVs = deltaVs[deltaVs[:, 1]!=None]
		j = list(takewhile(lambda x: np.concatenate(([0],np.diff(deltaVs[:,1])))[x]<=0, range(len(deltaVs[:,0])))); deltaVs=deltaVs[j]
		
		if len(deltaVs)<5: return None #Again, catching where there are too few points.

		#Interpolates across the deltaVs. This will allow us to minimise later.
		func = interpolate.UnivariateSpline(deltaVs[:,0], deltaVs[:,1], k=3, s=0)
		if prnt:
			ax=plt.subplot()
			ax.plot(deltaVs[:,0], abs(func(deltaVs[:,0])))
			ax.plot(deltaVs[:,0], deltaVs[:,1],color = 'red')
			ax.set_xlabel('Temperature'); ax.set_ylabel('DeltaV')
			plt.show()
		
		#Choose a 'guess' to be slightly closer to the higher temperature range.
		if guessIn==None:
			guess = (max(deltaVs[:,0])-min(deltaVs[:,0]))*0.85 + min(deltaVs[:,0])
		else: guess = guessIn

		#Minimise interpolated function (two methods in case one fails)
		res = optimize.minimize(lambda x: abs(func(x)), guess,bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0])*1.2)])
		if prnt: print(res)
		if res.success and res.fun<scale**3 and self.findminima(res.x[0]) is not None:
			
			if prnt: plotV(self, [res.x[0]*0.99,res.x[0],res.x[0]*1.01])    
			if prnt: splitV(self, res.x[0])     

			self.tc = res.x[0]
			return res.x[0]
		elif res.fun<scale**2 and self.findminima(res.x[0]) is not None:
			if prnt: plotV(self, [res.x[0]*0.99,res.x[0],res.x[0]*1.01])    
			if prnt: splitV(self, res.x[0])         

			self.tc = res.x[0]
			return res.x[0]
		else:
			#Sometimes this will just hit the boundary & fail.
			res = optimize.minimize(lambda x: abs(func(x)), guess,method='Nelder-Mead',bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0])*1.1)])
			if prnt: print(res)
			if res.success and res.fun<scale**3 and self.findminima(res.x[0]) is not None:
				#print(f'minimum = {self.findminima(res.x[0])}, deltaV={self.deltaV(res.x[0])}')    
				if prnt: plotV(self, [res.x[0]*0.99,res.x[0],res.x[0]*1.01])
				if prnt: splitV(self, res.x[0])
				
				self.tc = res.x[0]
				return res.x[0]
			else:
				return None
	#### These are supplemental ####
		
	def T0(self, plot=False):
	#NOTE TO SELF this might need updating to find a numerical minima.
		D = (2*mW**2 + mZ**2 + 2*mt**2)/(8*v**2)
		print(f"D={D}")
		#Expression for T0 from the draft.
		T0 = np.sqrt(l*v**2/(4*D*self.ga**2) * (6*(1-self.ep)*self.g4*self.ga**(-1) - 3*self.ga**(-2)*self.g4**2 - 2))
		#Plot the potential at T0:
		if plot:
			plt.plot(np.linspace(-2*v,0.5*v, num=100), self.Vtot(np.linspace(-3*v,0.5*v, num=100),T0))
			debug_plot(name="debug", overwrite=False)
   			#plt.show()
		return T0
			
	
	def findPeaks(self, T):
		lhs,rhs = self.findminima(T)
		if rhs == None: return [lhs]
        
		range=np.linspace(lhs-1,rhs+1,num=1000)
    
		minima = signal.find_peaks([self.Vtot(x, T) for x in range])[0]
		maxima = signal.find_peaks([-self.Vtot(x, T) for x in range])[0]

		return np.concatenate((range[minima], range[maxima]))

	def minmaxPlot(self, T_min, T_max):
		extrema = {}

		for T in np.linspace(T_min, T_max):
			extrema.update({T:self.findPeaks(T)})

		scat2 = np.array([(T, ext) for T in extrema.keys() for ext in extrema[T]])
		plt.scatter(scat2[:,0], scat2[:,1])
	
	def imaginary(self,h,T):
		return (self.V1T(h,T)).imag + (self.V1_cutoff(h)).imag
	def real(self,h,T):
		return self.V(h) + (self.V1T(h,T)).real + (self.V1_cutoff(h)).real

		

##SPLINE FIT
# Spline fitting, Jb
_xbmin = -3.72402637
# We're setting the lower acceptable bound as the point where it's a minimum
# This guarantees that it's a monatonically increasing function, and the first
# deriv is continuous.
_xbmax = 1.41e3
_Jb_dat_path = fT.spline_data_path+"/finiteT_b.dat.txt"
if os.path.exists(_Jb_dat_path):
    _xb, _yb = np.loadtxt(_Jb_dat_path).T
else:
    # x = |xmin|*sinh(y), where y in linear
    # (so that we're not overpopulating the uniteresting region)
    _xb = np.linspace(np.arcsinh(-1.3*20),
                         np.arcsinh(-20*_xbmax/_xbmin), 1000)
    _xb = abs(_xbmin)*np.sinh(_xb)/20
    _yb = fT.Jb_exact2(_xb)
    np.savetxt(_Jb_dat_path, np.array([_xb, _yb]).T)


_tckb = interpolate.splrep(_xb, _yb)
def Jb_spline(X,n=0):
    """Jb interpolated from a saved spline. Input is (m/T)^2."""
    X = np.array(X)
    x = X.ravel()
    y = interpolate.splev(x,_tckb, der=n).ravel()
    y[x < _xbmin] = interpolate.splev(_xbmin,_tckb, der=n)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)


		
if __name__ == "__main__":
	print('hello')
	print(len(_xb))
	result = []
	for mT in np.linspace(0.1,0.5):
		result.append(Jb_spline(mT))
	print(result)
	plt.plot(np.linspace(0.1,0.5),result)
	debug_plot(name="debug", overwrite=False)
	#plt.show()
