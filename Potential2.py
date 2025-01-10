import cosmoTransitions.finiteT as fT
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate, integrate
from itertools import takewhile
import os
from mpl_toolkits.mplot3d import Axes3D

#FOR A LINEAR POTENTIAL (i.e. a symmetric and a non-symmetric phase)
FPI = 600

class NotImplemented(Exception):
    pass
class NonLinear(Exception):
	pass
class InvalidPotential(Exception):
    pass

#CONVERT between physical inputs and Lagrangian parameters

def masses_to_lagrangian_Csaki(_m2Sig, _m2Eta, _m2X, _m2Pi, N, F):
    if int(round(N/F))==1:
        #Not Valid
        '''
        m2 = (3*_m2Eta - _m2Sig)/2
        c = (_m2Eta - _m2Pi) * F * FPI
        ls = (_m2Sig - _m2Eta)/FPI**2
        la = (_m2X + _m2Pi - 2*_m2Eta)/FPI**2

        #CHECKS:
        if m2<0:
            print('Point is Invalid as m2<0')
            return (None,None,None,None)
        if ls<0 or la<0:
            print(r'Point is Invalid as $\lambda_\sigma$<0')
            return (None,None,None,None)
        if (-8 * F**4 * m2**3 + 27 * c**2 * ls)<0:
            print("Invalid point as only one real solution to potential")
            return (None,None,None,None)
            '''

    elif int(round(F/N))==2:
        #From Mathematica SigmaVev.nb. 
        c = F**2 * (_m2Eta + _m2Pi)/(F**2 - F + 2)
        m2 = (_m2Sig + _m2Eta)/2 - c
        frac_la_ls = ( _m2X - c*(1/2 + 1/F) ) / (2*m2 + c)
        ls = 2*(F*m2 + c*N)/(F*FPI**2)
        
        la = frac_la_ls * ls
        
        #CHECKS
        if (m2 + c)<0:
            print('Point is Invalid as squared term is less than 0')
            return (None,None,None,None)
        if ls<0:
            print(f'Point is Invalid as $\lambda_\sigma$<0')
            return (None,None,None,None)
        
    else:
        raise NotImplemented(f"F/N={F/N} not implemented yet in masses_to_lagrangian function")

    
    
    return (m2, c, ls, la)

def masses_to_lagrangian_Normal(_m2Sig, _m2Eta, _m2X, _m2Pi, N, F):
    if F == 6:
        c = (_m2Eta-_m2Pi)/FPI**4 
        ls = (_m2Sig-_m2Eta)/FPI**2 + 5*c*FPI**2/3
        m2 = -((_m2Sig + _m2Eta)/2 - ls*FPI**2)
        la = (_m2X - _m2Pi)/FPI**2 - c*FPI**2/3

    #CHECKS
        if m2<0:
            print('Point is Invalid as mass term is less than 0')
            return (None,None,None,None)
        if c>0:
            print(f'Point is Invalid as sixth power term is negative')
            return (None,None,None,None)
        if (-3*ls + np.sqrt(-24*c*m2+9*ls**2)) <0:
            print(f'Point is Invalid as vev is imaginary (CHECK - OTHER VEVs EXIST)')
            return (None,None,None,None)

    else:
        raise NotImplemented(f"F/N={F/N} not implemented yet in masses_to_lagrangian function")

    return (m2, c, ls, la)

class Potential:

    def __init__(self, m2, c, lambdas, lambdaa, N, F, CsakiTerm):
		#All the parameters needed to construct the potential.
        self.m2 = m2
        self.c = c
        self.lambdas = lambdas
        self.lambdaa = lambdaa
        self.N = N
        self.F = F
        self.CsakiTerm = CsakiTerm
        
        
        if not CsakiTerm:
            self.detPow = 1 
        if CsakiTerm:
            self.detPow = N

        
		#Checking to make sure the Lagrangian is linear.
        if self.CsakiTerm:
            frac = self.F/self.N
            if abs(frac - np.round(frac)) > 0.0001:
                raise NonLinear(f"Choice of N gives non-linear Lagrangian.")
            
        ##GLUONIC FITS
        data = np.genfromtxt(f'GridDataF{self.F}N{self.N}Corrected.csv', delimiter=',', dtype=float, skip_header=1)
        #self.linear = interpolate.LinearNDInterpolator(data[:,0:2],data[:,2])
        self.linear = interpolate.SmoothBivariateSpline(data[:,0],data[:,1],data[:,2]/1e10, kx=1,ky=1)
        self.nearest = interpolate.NearestNDInterpolator(data[:,0:2],data[:,2])
        

        #Field dependent masses for fermions & bosons, their derivative wrt h (twice) and their respective DoF
        if self.F/self.detPow>=2:
            self.mSq = {
                #Sig Mass	
                'Sig': [lambda sig, T: - self.m2 
						- self.c / (self.F * self.detPow) * ( self.F/self.detPow - 1 ) * sig**(self.F/self.detPow - 2)
						+ (3/2) * self.lambdas * sig ** 2,
                        1.],
				#Eta Prime Mass
                'Eta': [lambda sig, T: - self.m2 
						+ self.c / (self.F * self.detPow) * ( self.F/self.detPow - 1 ) * sig**(self.F/self.detPow - 2)
						+ (1/2) * self.lambdas * sig ** 2,
                        1.],
				#X Mass
				'X': [lambda sig, T: - self.m2 
						+ self.c / self.F * sig**(self.F/self.detPow - 2)
						+ (1/2) * self.lambdas * sig ** 2,
						self.F**2 - 1],
				#Pi Mass
				'Pi': [lambda sig, T: - self.m2 
						- self.c / self.F * sig**(self.F/self.detPow - 2)
						+ (1/2) * (self.lambdas + self.lambdaa) * sig ** 2,
						self.F**2 - 1]
						}

        elif self.F/self.detPow<2:
            raise InvalidPotential("F/N is too small. Diverging effective masses.")


        else:
            raise NotImplemented(f"F/N={self.F/self.detPow} not implemented yet in mSq")
	
    def _Vg(self, T,sig):
        sig = np.array(sig); sig = np.abs(sig)

        if sig.shape is not ():
            return np.reshape(np.array([self.linear.ev(T,1000)*1e10 if s>1000 else self.linear.ev(T,s)*1e10 for s in sig]),sig.shape)


        else:
            if sig>1000:
                return self.linear.ev(T,1000)*1e10 
            else:
                return self.linear.ev(T,sig)*1e10 
            
            
		
    def V(self,sig): 
        ##The tree level, zero temperature potential.
        sig = np.array(sig)	
        return - self.m2 * sig**2/2 - (self.c/self.F**2) * sig**(self.F/self.detPow) + (self.lambdas/8) * sig**4


    def V1T(self,sig,T):
        sig = np.array(sig)
        if T==0:
            return np.zeros(sig.shape)

        return np.reshape((np.sum([n*Jb_spline((m2(sig,T)/T**2)) for m2, n in [self.mSq['Sig'],self.mSq['Eta'],
                                                                                self.mSq['X'],self.mSq['Pi']]],axis=0))*T**4/(2*np.pi**2), sig.shape)


    def VGluonic(self, sig, T):
        sig = np.array(sig)
        if T==0:
            return np.zeros(sig.shape)
        return np.reshape(self._Vg(T,sig),sig.shape)
	

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
        if self.F/self.detPow == 1:
            x = 9 * self.c * self.F**4 * self.lambdas**2 + np.sqrt(3)*np.sqrt( abs(self.F**8 * self.lambdas**3 * (-8 * self.F**4 *self.m2**3 + 27*self.c**2*self.lambdas)) )
            return np.real((2 * 3**(1/3) * self.F**4 * self.m2 * self.lambdas + x**(2/3)) / (3**(2/3)*self.F**2*self.lambdas * x**(1/3)))
		
        elif self.F/self.detPow == 2:
            return 2 * np.sqrt(self.F*self.m2 + self.c*self.detPow)/np.sqrt(self.F*self.lambdas)
		
        elif self.F/self.detPow == 3:
            return (self.c*self.detPow + np.sqrt(self.c**2*self.detPow**2 + 2*self.F**2*self.m2*self.lambdas)/(self.F*self.lambdas))
		
        elif self.F/self.detPow == 4:
            if self.lambdas - 4*self.c*self.detPow/self.F > 0:
                return 2*np.sqrt(self.m) / np.sqrt(self.lambdas - 4*self.c*self.detPow/self.F)
            else:
                raise InvalidPotential(f"Unbounded potential for F={self.F}")
        
        elif self.F/self.detPow == 6:
            return np.sqrt( - (-3*self.lambdas + np.sqrt(-24*self.c*self.m2 + 9*self.lambdas**2))/(2*self.c))
        else:
            raise NotImplemented(f"F/N={self.F/self.detPow} not implemented yet in fSigma")

    def findminima(self,T,rstart=None,rcounter=1, tolerance=None):
        #For a linear sigma model. Returns the minimum away from the origin.
        if rstart == None:
            rstart = self.fSigma()
        #Roll down to rhs minimum:
        res = optimize.minimize(lambda X: self.Vtot(X, T), rstart,method='Nelder-Mead',tol=tolerance,bounds=[(0,self.fSigma()*1.1)])
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

    def deltaV(self,T, rstart=None, num_res=None):
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
        Ts_init = np.linspace(100,1000,num=400); deltaVs_init=[]

        for T in Ts_init:
            deltaV =  self.deltaV(T, rstart=scale)
            if deltaV is not None and deltaV > 0: deltaVs_init.append([T,deltaV])
            if deltaV is not None and deltaV < 0: break
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
                plt.plot(np.linspace(-10,self.fSigma()*.75,num=100),V.Vtot(np.linspace(-10,self.fSigma()*.75,num=100),T)-V.Vtot(0,T),label=f"T={T}")
                if self.findminima(T) is not None:
                    plt.scatter(self.findminima(T), V.Vtot(self.findminima(T),T)-V.Vtot(0,T))
            plt.legend()
            plt.show()	

	
		#Find delta V for a finer scan of temperatures & interpolate between them. 
        Ts = np.linspace(T_init*0.95,T_init*1.35,num=100); deltaVs = np.array([[T, self.deltaV(T, rstart=scale*.8)] for T in Ts if self.deltaV(T,rstart=scale) is not None])
		
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
            res = optimize.minimize(lambda x: abs(func(x)), guess,method='Nelder-Mead',bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0])*1.1)])
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
    fig = plt.figure()
    
    #ax = fig.add_subplot(projection='3d')
    data = np.genfromtxt(f'gridDataF3N3Corrected.csv', delimiter=',', dtype=float, skip_header=1)
    _Vg = interpolate.LinearNDInterpolator(data[:,0:2],data[:,2])
    Ts = np.linspace(5,1000,num=10)
    sigs = np.linspace(0,1000,num=10)
    x2,y2 = np.meshgrid(Ts,sigs)
    _Vg2 = interpolate.SmoothBivariateSpline(x2.flatten(), y2.flatten(),_Vg(x2.flatten(),y2.flatten()),kx=1,ky=1,s=5000)
 

    sigs = np.linspace(0,1000,num=600)
    
    plt.plot(sigs, _Vg(471,sigs).flatten(),color='red')
    plt.plot(sigs, _Vg2.ev(471,sigs).flatten(),color='blue')

    print(data[4747:4847,0])
    plt.plot(np.arange(0,1000,10), data[4747:4847,2],color='green')
    #x2,y2 = np.meshgrid(Ts,sigs)
    #ax.plot_trisurf(x2.flatten(), y2.flatten(),_Vg(x2.flatten(),y2.flatten()), linewidth=0, antialiased=False)
   
    #ax.plot_trisurf(data[:,0],data[:,1],data[:,2], linewidth=0, antialiased=False)
    plt.show()
    '''
    data2 = np.genfromtxt(f'Gridsigmastep5.csv', delimiter=',', dtype=float, skip_header=1)
    
    plt.plot(data2[:,1],data2[:,2])
    plt.xlabel(r'$\sigma$')
    plt.ylabel('V')
    plt.show()'''
    

