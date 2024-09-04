from cosmoTransitions.finiteT import Jb_exact2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate
from itertools import takewhile

#FOR A LINEAR POTENTIAL (i.e. a symmetric and a non-symmetric phase)

##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2


class Potential:

	def __init__(self, lmb, kappa, m2Sig, muSig, xi, loop=False):
		#All the parameters needed to construct the potential.
		self.lmb = lmb
		self.kappa = kappa
		self.m2Sig = m2Sig
		self.muSig = muSig
		self.xi = xi
		self.loop = loop
		#Looks like loops not needed for this calculation.


		#Higgs dependent masses for fermions & bosons, their derivative wrt h (twice) and their respective DoF
		self.mSq = {
			#Phi Mass	
			'Phi': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig - (2/np.sqrt(6))*self.muSig * phi + 3*(self.lmb/2 + self.kappa/6) * phi**2,
					1.],
			#Eta Prime Mass
			'Eta': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig + (2/np.sqrt(6))*self.muSig * phi + (self.lmb/2 + self.kappa/6) * phi**2,
					1.],
			#X Mass
			'X': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig - 3 * self.xi + (1/np.sqrt(6))*self.muSig * phi + (self.lmb/2 + 3*self.kappa/6) * phi**2,
					8.],
			#Pi Mass
			'Pi': [lambda phi, T: (5*self.lmb/6 + self.kappa/2) * T**2 - self.m2Sig - 3 * self.xi - (1/np.sqrt(6))*self.muSig * phi + (self.lmb/2 + self.kappa/6) * phi**2,
					8.]
					}
		
	def V(self,phi):
		##The tree level, zero temperature potential.
		phi = np.array(phi)
		return -(0.5*self.m2Sig) * phi**2 - self.muSig/(3*np.sqrt(6)) * phi**3 + 0.25 * (self.lmb/2 + self.kappa/6) * phi**4
    
	def V1T(self,phi,T):
		phi = np.array(phi)
		if T==0:
			return np.zeros(phi.shape)
		
		return np.reshape((np.sum([n*Jb_exact2((m2(phi,T)/T**2)) for m2, n in [self.mSq['Phi'],self.mSq['Eta'],self.mSq['X'],self.mSq['Pi']]],axis=0))*T**4/(2*np.pi**2), phi.shape)


	def V1_cutoff(self, h):
		# One loop corrections to effective potential in cut-off regularisation scheme.
		#MAY NOT NEED THESE YET!
		h = np.array(h)
			
		return np.reshape(np.sum([n * (m2(h)**2 * (np.log(np.abs(m2(h)/m2(0)) + 1e-100) - 1.5) + 2*m2(h)*m2(0)) for m2, n in [self.mSq['W'],self.mSq['Z'],self.mSq['t'],self.mSq['h'],self.mSq['Wl'],self.mSq['Zl']]],axis=0)/(64.*np.pi**2), h.shape)
		

	def Vtot(self,phi,T):
		#This finds the total (one-loop/tree level) thermal effective potential.
		phi = np.array(phi)
		if self.loop:
			return self.V(phi) + self.V1T(phi,T).real + self.V1_cutoff(phi).real
		else:
			return self.V(phi) + self.V1T(phi,T).real

	def dVdT(self,phi,T,eps=0.001):
		#Uses finite difference method to fourth order. Takes scalar h and T.
		return (self.Vtot(phi,T-2*eps) - 8*self.Vtot(phi,T-eps) + 8*self.Vtot(phi,T+eps) - self.Vtot(phi,T+2*eps)) / (12.*eps)
			
	def d2VdT2(self,phi,T,eps=0.001):
		#Uses finite difference method to fourth order. Takes scalar h and T.
		return (self.dVdT(phi,T-2*eps) - 8*self.dVdT(phi,T-eps) + 8*self.dVdT(phi,T+eps) - self.dVdT(phi,T+2*eps)) / (12.*eps)

	def findminima(self,T,rstart=6000,rcounter=1, tolerance=None):
		#For a linear sigma model. Returns the minimum away from the origin.
		#Roll down to rhs minimum:
		rhs = optimize.minimize(lambda X: self.Vtot(X, T), rstart,tol=tolerance).x[0]
		#Check the two rolls didn't find the same minimum:
		if abs(rhs)<2.5:
			#If so, try a new start
			if rcounter<=3:
				rhs = self.findminima(T,rstart=rstart*1.1,rcounter=rcounter+1)
			else:
				return None
		return rhs

	def deltaV(self,T,num_res=False):
		#Finds the difference between the symmetric and broken minima.
		vT = self.findminima(T)
		if vT is not None:
			return + self.Vtot(0, T) - self.Vtot(vT, T)
		else:
			if num_res:
				return 1e30
			else:
				return None


	def	criticalT(self, guessIn=None,prnt=True):
		#Critical temperature is when delta V is zero (i.e. both minima at the same height)
		#Find delta V for a range of temperatures & interpolate between them. 

		Ts = np.linspace(4000,6000,num=200); deltaVs = np.array([[T, self.deltaV(T)] for T in Ts if self.deltaV(T) is not None])
		
		if len(deltaVs)<5: return None #Catches if there are just not enough points to make a verdict.
		print(deltaVs)
		
		
		#Ensure each deltaV is decreasing with increasing T.
		j = list(takewhile(lambda x: np.concatenate(([0],np.diff(deltaVs[:,1])))[x]<=0, range(len(deltaVs[:,0])))); deltaVs=deltaVs[j]
		print(deltaVs)
		if len(deltaVs)<5: return None		

		func = interpolate.UnivariateSpline(deltaVs[:,0], abs(deltaVs[:,1]), k=3,s=0)
		if prnt:
			plt.plot(deltaVs[:,0], func(deltaVs[:,0]))
			plt.plot(deltaVs[:,0], deltaVs[:,1],color = 'red')
			plt.show()
		
		if guessIn==None:
			guess = (max(deltaVs[:,0])-min(deltaVs[:,0]))*0.75 + min(deltaVs[:,0])
		else: guess = guessIn
		print(f'guess = {guess}; min = {min(deltaVs[:,0])}; max = {max(deltaVs[:,0])}')
		#Minimise interpolated function (two methods in case one fails)
		res = optimize.minimize(lambda x: abs(func(x)), guess,bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0]))])
		if prnt: print(res)
		if res.success and res.fun<1e13:
			return res.x[0]
		else:
			res = optimize.minimize(lambda x: abs(func(x)), guess,method='Nelder-Mead',bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0]))])
			if prnt: print(res)
			if res.success and res.fun<5e13:
				return res.x[0]
			else:
				if guessIn == None: return self.criticalT(guessIn = (max(deltaVs[:,0])-min(deltaVs[:,0]))*0.9 + min(deltaVs[:,0]))
				else: return None

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

		
		
if __name__ == "__main__":
	print('hello')
