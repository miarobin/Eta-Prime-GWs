import cosmoTransitions.finiteT as fT
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate, integrate
from itertools import takewhile
import os

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

	def findminima(self,T,rstart=None,rcounter=1, tolerance=None):
		#For a linear sigma model. Returns the minimum away from the origin.
		if rstart == None:
			rstart = self.fSigmaApprox()
		#Roll down to rhs minimum:
		res = optimize.minimize(lambda X: self.Vtot(X, T), rstart,method='Nelder-Mead',tol=tolerance)
		#Check the two rolls didn't find the same minimum:
		if not res.success or res.x[0]<0.5:
			#If so, try a new start
			if rcounter<=4:
				if rstart is not None:
					return self.findminima(T,rstart=rstart*0.9,rcounter=rcounter+1)
				else: return None
			else:
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
			



	def	criticalT(self, guessIn=None,prnt=True):
		#Critical temperature is when delta V is zero (i.e. both minima at the same height) THIS HAS TO BE QUITE ACCURATE!
		
		#Scale with which we can compare potential magnitudes (think large values of phi, V~phi^4)
		scale = self.findminima(0)
		print(3)
		
		#First a coarse scan. Find the minimum deltaV from this initial scan, then do a finer scan later.
		Ts_init = np.linspace(50,10000,num=400); deltaVs_init=[]

		for T in Ts_init:
			deltaV =  self.deltaV(T, rstart=scale)
			if deltaV is not None and deltaV > 0: deltaVs_init.append([T,deltaV])
			if deltaV < 0: break
		print(deltaVs_init)
		deltaVs_init=np.array(deltaVs_init)
		
		print(4)
		if prnt:
			for T,_ in deltaVs_init:
				plt.scatter(self.findminima(T),T)
				plt.scatter(0,T)
				plt.xlabel('Delta V'); plt.ylabel('T')
			plt.show()	
		print(5)
		#JUST taking deltaV's which are greater than zero BUT decreasing. Note the reason for this is often going further can confuse python later.
		j = list(takewhile(lambda x: np.concatenate(([0],np.diff(deltaVs_init[:,1])))[x]<=0, range(len(deltaVs_init[:,0])))); deltaVs_init=deltaVs_init[j]
		k = list(takewhile(lambda x: deltaVs_init[x,1]>0, range(len(deltaVs_init[:,0]))))


		deltaVs_init=deltaVs_init[k]
		#This is the temperature with deltaV closest to zero:
		T_init = deltaVs_init[-1,0]
		if prnt:
			plt.plot(deltaVs_init[:,1], deltaVs_init[:,0])
			plt.xlabel('Delta V'); plt.ylabel('Temperature')
			plt.show()

			print(f"Coarse grain scan finds {T_init} being closest Delta V to 0")

		def plotV(V, Ts):
			for T in Ts:
				plt.plot(np.linspace(-10,self.fSigmaApprox()*.5,num=100),V.Vtot(np.linspace(-10,self.fSigmaApprox()*.5,num=100),T)-V.Vtot(0,T),label=f"T={T}")
				if self.findminima(T) is not None:
					plt.scatter(self.findminima(T), V.Vtot(self.findminima(T),T)-V.Vtot(0,T))
			plt.legend()
			plt.show()	
		print(6)
	
		#Find delta V for a finer scan of temperatures & interpolate between them. 
		Ts = np.linspace(T_init*0.95,T_init*1.35,num=100); deltaVs = np.array([[T, self.deltaV(T, rstart=scale)] for T in Ts if self.deltaV(T,rstart=scale) is not None])
		
		if len(deltaVs)<5: return None #Catches if there are just not enough points to make a verdict.
		
		
		#Ensure each deltaV is decreasing with increasing T.
		j = list(takewhile(lambda x: np.concatenate(([0],np.diff(deltaVs[:,1])))[x]<=0, range(len(deltaVs[:,0])))); deltaVs=deltaVs[j]
		
		if len(deltaVs)<5: return None #Again, catching where there are too few points.

		#Interpolates across the deltaVs. This will allow us to minimise later.
		func = interpolate.UnivariateSpline(deltaVs[:,0], deltaVs[:,1], k=3, s=0)
		if prnt:
			plt.plot(deltaVs[:,0], abs(func(deltaVs[:,0])))
			plt.plot(deltaVs[:,0], deltaVs[:,1],color = 'red')
			plt.xlabel('Temperature'); plt.ylabel('DeltaV')
			plt.show()
		
		#Choose a 'guess' to be slightly closer to the higher temperature range.
		if guessIn==None:
			guess = (max(deltaVs[:,0])-min(deltaVs[:,0]))*0.85 + min(deltaVs[:,0])
		else: guess = guessIn
		
		#Minimise interpolated function (two methods in case one fails)
		res = optimize.minimize(lambda x: abs(func(x)), guess,bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0])*1.2)])
		if prnt: print(res)
		if res.success and res.fun<scale**3:
			
			if prnt: plotV(self, [res.x[0]*0.99,res.x[0],res.x[0]*1.01])			

			return res.x[0]
		else:
			res = optimize.minimize(lambda x: abs(func(x)), guess,method='Nelder-Mead',bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0]))])
			if prnt: print(res)
			if res.success and res.fun<5*scale**3:
				if prnt: plotV(self, [res.x[0]*0.99,res.x[0],res.x[0]*1.01])		
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
			plt.show()
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
	plt.show()
