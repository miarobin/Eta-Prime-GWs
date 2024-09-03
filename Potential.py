from cosmoTransitions.tunneling1D import SingleFieldInstanton
from cosmoTransitions.finiteT import Jb_spline, Jf_spline
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate
from itertools import takewhile

##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2



def gammaa(ep, g4, delta):
	if ep<1-np.sqrt(8/9):
		ge = (3*(1-ep)-np.sqrt(1+9*(ep**2-2*ep)))/2
	else:
		ge=np.sqrt(2)
	return 1/(ge/g4 - delta)

class Potential:
	def F(self, h):
		return np.sqrt(np.cos(self.beta)**2*(1+self.ga*h/v)**2 + np.sin(self.beta)**2)
	def dFdh(self, h):
		return np.cos(self.beta)**2*(self.ga/v)*(1+h*self.ga/v)/self.F(h)
		

	def __init__(self, ep,g4,ga,beta,higgs_corr=True,loop=True,msbar=False):
		self.ep = ep
		self.g4 = g4
		self.ga = ga
		self.beta = beta
		self.hcorr = higgs_corr
		self.loop = loop
		self.msbar = msbar


		#Higgs dependent masses for fermions & bosons, their derivative wrt h (twice) and their respective DoF
		self.mSq = {
			#Higgs Mass	
			'h': [lambda h: mh**2 + 3.*l*v*self.g4*(1-self.ep)*h + 1.5*l*self.g4**2*h**2, 
					1.],
			#W Mass
			'W': [lambda h: mW**2*self.F(h)**2,
					4.],
			#Z Mass
			'Z': [lambda h: mZ**2*self.F(h)**2,
					2.],
			#Wl Mass
			'Wl': [lambda h: self.dFdh(h)/self.F(h) * (mh**2*h + 1.5*np.sqrt(l)*mh*self.g4*(1-self.ep)*h**2 + 0.5*l*self.g4**2*h**3) + mW**2*self.F(h)**2,
					2.],
			#Zl Mass
			'Zl': [lambda h: self.dFdh(h)/self.F(h) * (mh**2*h + 1.5*np.sqrt(l)*mh*self.g4*(1-self.ep)*h**2 + 0.5*l*self.g4**2*h**3) + mZ**2*self.F(h)**2,
					1.],
			#top mass
			't': [lambda h: mt**2*self.F(h)**2,
					-12.],
			#bottom mass
			'b': [lambda h: mb**2*self.F(h)**2,
					-12.],
			#Goldstone Mass
			'G': [lambda h: self.dFdh(h)*(mh**2*h + 1.5*np.sqrt(l)*mh*self.g4*(1-self.ep)*h**2 + 0.5*l*self.g4**2*h**3)/self.F(h), 
					3.],
					}
		
	def V(self,h):
		##The tree level, zero temperature potential.
		h = np.array(h)
		return (0.5*mh**2)*h**2 + 0.5*np.sqrt(l)*mh*self.g4*(1-self.ep)*h**3 + 0.125*l*self.g4**2*h**4
    
	def V1T(self,h,T):
		h = np.array(h)
		if T==0:
			return np.zeros(h.shape)
		if self.hcorr:
			return np.reshape((np.sum([n*Jb_spline((m2(h)/T**2)) for m2, n in [self.mSq['W'],self.mSq['Z'],self.mSq['h'],self.mSq['Wl'],self.mSq['Zl']]],axis=0) 
					+ np.sum([-n*Jf_spline((m2(h)/T**2)) for m2, n in [self.mSq['t']]],axis=0))*T**4/(2*np.pi**2), h.shape)
		else:
			return np.reshape((np.sum([n*Jb_spline((m2(h)/T**2)) for m2, n in [self.mSq['W'],self.mSq['Z']]],axis=0) 
					+ np.sum([-n*Jf_spline((m2(h)/T**2)) for m2, n in [self.mSq['t']]],axis=0))*T**4/(2*np.pi**2), h.shape)

	def V1_msbar(self, h, Q=2*v):
		h = np.array(h)
		
		return np.reshape(np.sum([n * m2(h)**2 * (np.log(np.abs(m2(h)/Q**2) + 1e-100) - 1.5) for m2, n in [self.mSq['t'],self.mSq['h']]],axis=0) + np.sum([n * m2(h)**2 * (np.log(np.abs(m2(h)/Q**2) + 1e-100) - 5/6) for m2, n in [self.mSq['W'],self.mSq['Z'],self.mSq['Wl'],self.mSq['Zl']]],axis=0),h .shape)/(64.*np.pi**2)
		
		
	def V1_cutoff(self, h):
		# One loop corrections to effective potential in cut-off regularisation scheme.
		h = np.array(h)
			
		return np.reshape(np.sum([n * (m2(h)**2 * (np.log(np.abs(m2(h)/m2(0)) + 1e-100) - 1.5) + 2*m2(h)*m2(0)) for m2, n in [self.mSq['W'],self.mSq['Z'],self.mSq['t'],self.mSq['h'],self.mSq['Wl'],self.mSq['Zl']]],axis=0)/(64.*np.pi**2), h.shape)
		

	def Vtot(self,h,T):
		h = np.array(h)
		if self.loop:
			if self.msbar:
				return self.V(h) + self.V1T(h,T).real + self.V1_msbar(h).real
			else:
				return self.V(h) + self.V1T(h,T).real + self.V1_cutoff(h).real
		else:
			return self.V(h) + self.V1T(h,T).real

			
	def dVdT(self,h,T,eps=0.001):
		#Uses finite difference method to fourth order. Takes scalar h and T.
		return (self.Vtot(h,T-2*eps) - 8*self.Vtot(h,T-eps) + 8*self.Vtot(h,T+eps) - self.Vtot(h,T+2*eps)) / (12.*eps)
			
	def d2VdT2(self,h,T,eps=0.001):
		#Uses finite difference method to fourth order. Takes scalar h and T.
		return (self.dVdT(h,T-2*eps) - 8*self.dVdT(h,T-eps) + 8*self.dVdT(h,T+eps) - self.dVdT(h,T+2*eps)) / (12.*eps)

	def findminima(self,T,lstart=-2*v,rcounter=1, tolerance=None):
		#Roll down to lhs minimum:
		lhs = optimize.minimize(lambda X: self.Vtot(X, T), lstart*rcounter,tol=tolerance).x[0]
		#Roll down to rhs minimum:
		rhs = optimize.minimize(lambda X: self.Vtot(X, T), 0,tol=tolerance).x[0]
		#Check the two rolls didn't find the same minimum:
		if abs(lhs-rhs)<2.5:
			#If so, try a new start
			if rcounter<=3:
				lhs,rhs = self.findminima(T,lstart=lstart,rcounter=rcounter+1)
			else:
				return ((lhs+rhs)/2, None)
		return (lhs, rhs)

	def deltaV(self,T,num_res=False):
		lhs, rhs = self.findminima(T)
		if rhs is not None:
			return self.Vtot(lhs, T) - self.Vtot(rhs, T)
		else:
			if num_res:
				return 1e30
			else:
				return None


	def	criticalT(self, guessIn=None,prnt=False):
		#Find delta V for a range of temperatures & interpolate between them. 
		'''if prnt:
			lhs, rhs = self.findminima(0)
			if rhs is not None:
				if lhs < -3000: lhs = -3500
				plt.plot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),self.Vtot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),0),label="Vtot at T=0GeV")
				plt.plot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),self.Vtot(np.linspace(lhs-0.5*v,rhs +0.5*v,num=100),200),label="Vtot at T=200GeV")
				plt.legend()
				plt.show()'''
		
		Ts = np.linspace(120,130,num=500); deltaVs = np.array([[T, self.deltaV(T)] for T in Ts if self.deltaV(T) is not None])
		#Ensure each deltaV is decreasing with increasing T
		if len(deltaVs)<5: return None
		print(deltaVs)
		
		j = list(takewhile(lambda x: np.concatenate(([0],np.diff(deltaVs[:,1])))[x]<=0, range(len(deltaVs[:,0])))); deltaVs=deltaVs[j]
		print(deltaVs)
		if len(deltaVs)<5: return None		

		func = interpolate.UnivariateSpline(deltaVs[:,0], abs(deltaVs[:,1])/v**2, k=3,s=0)
		if prnt:
			plt.plot(deltaVs[:,0], func(deltaVs[:,0]))
			plt.plot(deltaVs[:,0], deltaVs[:,1]/v**2,color = 'red')
			plt.show()
		
		if guessIn==None:
			guess = (max(deltaVs[:,0])-min(deltaVs[:,0]))*0.75 + min(deltaVs[:,0])
		else: guess = guessIn
		print(f'guess = {guess}; min = {min(deltaVs[:,0])}; max = {max(deltaVs[:,0])}')
		#Minimise interpolated function (two methods in case one fails)
		res = optimize.minimize(lambda x: abs(func(x)), guess,bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0]))])
		if prnt: print(res)
		if res.success and res.fun<1:
			return res.x[0]
		else:
			res = optimize.minimize(lambda x: abs(func(x)), guess,method='Nelder-Mead',bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0]))])
			if prnt: print(res)
			if res.success and res.fun<5:
				return res.x[0]
			else:
				if guessIn == None: return self.criticalT(guessIn = (max(deltaVs[:,0])-min(deltaVs[:,0]))*0.9 + min(deltaVs[:,0]))
				else: return None

		
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
			
	def gammaedelta(self):
		print(f"eps in V = {self.ep}")
		print(f"3(1-eps)={3*(1-self.ep)}")
		print(f"sqrt bit = {np.sqrt(1+9*(self.ep**2-2*self.ep))}")
		ge = (3*(1-self.ep)-np.sqrt(1+9*(self.ep**2-2*self.ep)))/2
		print(f"ge in V = {ge}")
		de = ge/self.g4 - 1/(self.ga)
		return(ge,de)
		
	def vStar(self):
		ge,delta=self.gammaedelta()
		return -v*(ge/self.g4 - delta)
	
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
	## Choose epsilon and gamma4 to slot into the potential, and the temperature to evaluate at.
	eps = 0.02; g4 = 1; delta = -0.1; beta = 0.05
	print(f"$\gamma_a$ = {gammaa(eps,g4,delta)}")
	
	#The Potential Object.
	V = Potential(eps,g4,gammaa(eps,g4,delta),beta,higgs_corr=True,loop=True)
	
	print(f"Thermal Minimum:{optimize.minimize(lambda h:V.V1T(h,150),-246).x}")
	print(f"Tree level maximum: {optimize.minimize(lambda h:-V.V(h), -246).x}")
	
	#Tn, beta_H, alp = gravitationalWave(V)
	#print(f"Tn = {Tn}, beta/H = {beta_H}, alpha = {alp}")

	Ts = [246,190]
	for T in Ts:
		plt.plot(np.linspace(-2*v,.5*v, num=100), np.abs(V.imaginary(np.linspace(-2*v,.5*v, num=100),T)),label=f"T = {T} GeV, Imaginary")
		plt.plot(np.linspace(-2*v,.5*v, num=100),np.abs(V.real(np.linspace(-2*v,.5*v, num=100),T)),label=f"T = {T} GeV, Real")

	plt.title(f"$\epsilon = ${eps}, $\delta$={delta}, $\gamma_4$={g4}, $beta$={beta}")
	plt.xlabel(f"$h$")
	plt.ylabel(f"$|V(h,T)|$")
	plt.legend()
	plt.show()
