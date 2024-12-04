import cosmoTransitions.finiteT as fT
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate, integrate
from itertools import takewhile
import os

#FOR A LINEAR POTENTIAL (i.e. a symmetric and a non-symmetric phase)


class NotImplemented(Exception):
    pass
class NonLinear(Exception):
	pass
class InvalidPotential(Exception):
    pass


class Potential:

    def __init__(self, mSq, c, lambdas, lambdaa, N, F, CsakiTerm=False):
		#All the parameters needed to construct the potential.
        self.mSq = mSq
        self.c = c
        self.lambdas = lambdas
        self.lambdaa = lambdaa
        self.N = N
        self.F = F
        self.CsakiTerm = CsakiTerm
        
		#Checking to make sure the Lagrangian is linear.
        if self.CsakiTerm:
            frac = self.F/self.N
            if abs(frac - np.round(frac)) > 0.0001:
                raise NonLinear(f"Choice of N gives non-linear Lagrangian.")
				


        #The coefficient of the determinant term as it's quite verbose:
        det_coef = 2*self.N * (self.F/self.N)*(self.F/self.N-1)*(1/np.sqrt(6))**(self.F/self.N)
        #Field dependent masses for fermions & bosons, their derivative wrt h (twice) and their respective DoF
        if self.F/self.N>2:
            self.mSq = {
                #Phi Mass	
                'Phi': [lambda sig, T: - self.mSq 
						- self.c / (self.F * self.N) * ( self.F/self.N - 1 ) * sig**(self.F/self.N - 2)
						+ (3/2) * self.lambdas * sig ** 2,
                        1.],
				#Eta Prime Mass
                'Eta': [lambda sig, T: - self.mSq 
						+ self.c / (self.F * self.N) * ( self.F/self.N - 1 ) * sig**(self.F/self.N - 2)
						+ (1/2) * self.lambdas * sig ** 2,
                        1.],
				#X Mass
				'X': [lambda sig, T: - self.mSq 
						+ self.c / self.F * sig**(self.F/self.N - 2)
						+ (1/2) * self.lambdas * sig ** 2,
						self.F**2 - 1],
				#Pi Mass
				'Pi': [lambda sig, T: - self.mSq 
						- self.c / self.F * sig**(self.F/self.N - 2)
						+ (1/2) * (self.lambdas + self.lambdaa) * sig ** 2,
						self.F**2 - 1]
						}

        elif self.F/self.N<=2:
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
			
		
    def V(self,sig): 
        ##The tree level, zero temperature potential.
        sig = np.array(sig)	
        return - self.mSq - (self.c/self.F**2) * sig**(self.F/self.N) + (self.lambdas/8) * sig**4


    def V1T(self,sig,T):
        sig = np.array(sig)
        if T==0:
            return np.zeros(sig.shape)

        return np.reshape((np.sum([n*Jb_spline((m2(sig,T)/T**2)) for m2, n in [self.mSq['Phi'],self.mSq['Eta'],
                                                                                self.mSq['X'],self.mSq['Pi']]],axis=0))*T**4/(2*np.pi**2), sig.shape)


    def VGluonic(self, sig, T):
        return 0
	

    def Vtot(self,sig,T, polyakov=True):
    #This finds the total (one-loop/tree level) thermal effective potential.
        sig = np.array(sig)
              
        if polyakov:
            return self.VGluonic(sig, T) + self.V(sig) + self.V1T(sig,T).real
	
        else:
            return self.V(sig) + self.V1T(sig,T).real

    def dVdT(self,sig,T,eps=0.001):
    #Uses finite difference method to fourth order. Takes scalar h and T.
        return (self.Vtot(sig,T-2*eps) - 8*self.Vtot(sig,T-eps) + 8*self.Vtot(sig,T+eps) - self.Vtot(sig,T+2*eps)) / (12.*eps)
			
    def d2VdT2(self,sig,T,eps=0.001):
        #Uses finite difference method to fourth order. Takes scalar h and T.
        return (self.dVdT(sig,T-2*eps) - 8*self.dVdT(sig,T-eps) + 8*self.dVdT(sig,T+eps) - self.dVdT(sig,T+2*eps)) / (12.*eps)

    def fSigma(self):
        if self.F/self.N == 1:
            x = 9 * self.c * self.F**4 * self.lambdas**2 + np.sqrt(3)*np.sqrt( self.F**8 * self.lambdas**3 * (-16 * self.F**4 *self.mSq**3 + 27*self.c**2*self.lambdas) )
            return 2**(1/3) * (2 * 6**(1/3) * self.F**4 * self.mSq * self.lambdas + x**(2/3)) / (3**(2/3)*self.F**2*self.lambdas * x**(1/3))
		
        elif self.F/self.N == 2:
            return 2 * np.sqrt(self.F*self.mSq + self.c*self.N)/np.sqrt(self.F*self.lambdas)
		
        elif self.F/self.N == 3:
            return 2*(self.c*self.N + np.sqrt(self.c**2*self.N**2 + self.F**2*self.mSq*self.lambdas)/(self.F*self.lambdas))
		
        elif self.F/self.N == 4:
            if self.lambdas - 4*self.c*self.N/self.F > 0:
                return 2*np.sqrt(self.m) / np.sqrt(self.lambdas - 4*self.c*self.N/self.F)
            else:
                raise InvalidPotential(f"Unbounded potential for F={self.F}")
        else:
            raise NotImplemented(f"F/N={self.F/self.N} not implemented yet in fSigma")

    def findminima(self,T,rstart=None,rcounter=1, tolerance=None):
        #For a linear sigma model. Returns the minimum away from the origin.
        if rstart == None:
            rstart = self.fSigma()
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
		
        #Scale with which we can compare potential magnitudes (think large values of sigma, V~sigma^4)
        scale = self.fSigma()
		
        #First a coarse scan. Find the minimum deltaV from this initial scan, then do a finer scan later.
        Ts_init = np.linspace(50,10000,num=400); deltaVs_init=[]

        for T in Ts_init:
            deltaV =  self.deltaV(T, rstart=scale)
            if deltaV is not None and deltaV > 0: deltaVs_init.append([T,deltaV])
            if deltaV < 0: break
        print(deltaVs_init)
        deltaVs_init=np.array(deltaVs_init)
		

        if prnt:
            for T,_ in deltaVs_init:
                plt.scatter(self.findminima(T),T)
                plt.scatter(0,T)
                plt.xlabel('Delta V'); plt.ylabel('T')
            plt.show()	

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
                plt.plot(np.linspace(-10,self.fSigma()*.5,num=100),V.Vtot(np.linspace(-10,self.fSigma()*.5,num=100),T)-V.Vtot(0,T),label=f"T={T}")
                if self.findminima(T) is not None:
                    plt.scatter(self.findminima(T), V.Vtot(self.findminima(T),T)-V.Vtot(0,T))
            plt.legend()
            plt.show()	

	
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

    def imaginary(self,sig,T):
        return (self.V1T(sig,T)).imag + (self.V1_cutoff(sig)).imag
    def real(self,sig,T):
        return self.V(sig) + (self.V1T(sig,T)).real + (self.V1_cutoff(sig)).real

		

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
