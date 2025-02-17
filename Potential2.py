import cosmoTransitions.finiteT as fT
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, misc, signal
from sympy import diff
from scipy import interpolate, integrate
from itertools import takewhile
import os
from mpl_toolkits.mplot3d import Axes3D

'''
    This file does the following:
    1. Converts physical particle masses to Lagrangian parameters in "masses_to_lagrangian_Csaki" and "masses_to_lagrangian_Normal". See appendix D of the draft.
    
        masses_to_lagrangian_Csaki/masses_to_lagrangian_Normal: 
        INPUTS:  (_m2Sig, _m2Eta, _m2X, _m2Pi, N, F)
                (float, float, float, float, int, int)
        OUTPUTS: (m2, c, ls, la)
        RAISES: NotImplemented (as each N, F case must be handcoded)
        
    2. Creates a 'Potential' class, which has the following properties:
    
        Potential:
        INPUTS: m2, c, lambdas, lambdaa, N, F, CsakiTerm
                (float, float, float, float, int, int, bool)
        ADDITIONAL PROPERTIES:  detPow, mSq
                                (float, dictionary)
        RAISES: NonLinear (If F/N is not integer)
                                
    3. A number of functions construct the potential. These are:
        a. "V". The tree level potential as a function of sigma.
        
            V:
            INPUTS: (sigma)
                    (np.array)
            OUTPUTS: V_tree
                    (np.array)
                    
            "V1T". The one-loop thermal correction to the potential. Uses the interpolated, full, thermal function from CosmoTransitions "Jb_spline".
            
            V1T:
            INPUTS: (sigma, T)
                    (np.array, np.array)
            OUTPUTS: V_thermal,one-loop
                    (np.array)
                    
                    
        b. "_Vg" and helper function "_Vg_f". 
            "_Vg_f" samples the interpolated gluonic potentials 'linear_small' at small sigma and 'linear_large' at large sigma for 
            single values of sigma and T. Beyond sigma = GLUONIC_CUTOFF, the potential is assumed constant. 
            "_Vg" extends this to deal with array inputs.
            
            _Vg:
            INPUTS: (sig, T)
                    (np.array, np.array)
            OUTPUTS: V_gluonic
                    (np.array)
            RAISES: NotImplemented (as each N, F case must be individually set)
            _Vg_f:
            INPUTS: (sig, T)
                    (float, float)
            OUTPUTS: V_gluonic
                    (float)
            RAISES: NotImplemented (as each N, F case must be individually set)
            
            "VGluonic" is a wrapper function for "_Vg" with VGluonic(T=0)=0.
            VGluoic:
            INPUTS: (sig, T)
                    (float, float)
            OUTPUTS: V_gluonic
                    (float)
            RAISES: NotImplemented (as each N, F case must be individually set)
        
        c. "Vtot" sums the contributions from "V", "V1T" and "VGluonic" if "Polyakov" is True.
        
            Vtot:
            INPUTS (sig, T)
                    (np.array, np.array, bool)
            OUTPUTS V_total
                    (np.array)
                    
    4. A bunch of properties of the potential which we'll need later.
        a. "dVdT" derivative of "Vtot" with respect to T. Formula: https://en.wikipedia.org/wiki/Five-point_stencil
            and "d2VdT2" second derivative of "Vtot" with respect to T. Formula: https://en.wikipedia.org/wiki/Five-point_stencil
            
            dVdT:
            INPUTS (sig,T,eps=0.001)
                    (np.array, np.array, float)
            OUTPUTS dV/dT
                    (np.array)
                    
            d2VdT2:
            INPUTS (sig,T,eps=0.001)
                    (np.array, np.array, float)
            OUTPUTS d2V/dT2
                    (np.array)
            
        b. "fSigma" is the analytic solution to the zero-temperature minimum of the tree level potential.
            
            fSigma:
            INPUTS ()
            OUTPUTS fSigma
                    (float)
            RAISES: NotImplemented (as each N, F case must be handcoded)
            
        c. "findMinimum" finds the minimum (if it exists) at sigma>0. Returns None if the second minimum does not exist. Uses Nelder-Mead minimisation.
            rstart sets the starting value for the minimisation algorithm, default is the result of "fSigms" and rcounter is a recursive mechanism for bringing the start value closer to sigma=0.
        
            findMinimum:
            INPUTS (T,rstart=None,rcounter=1)
                    (float, float, int)
            OUTPUTS minima OR None
                    (float)
                    
        d. "criticalT" finds the critical temperature of the potential. Returns None if it does not exist (i.e. second order phase transition).
            guessIn allows you to input a guess for the critical temperature and prnt plots a bunch of intermediary checks. See code for detailed explanation.
        
            criticalT:
            INPUTS (guessIn=None,prnt=True)
                    (float, bool)
            OUTPUTS criticalT OR None
                    (float)
                    
    5. "Jb_spline" is a spline interpolation of the bosonic thermal function, taken from CosmoTransitions code as it didn't work properly for some reason.
        NB X=(m/T)^2 where m is the effective mass at a particular sigma. https://github.com/clwainwright/CosmoTransitions/blob/master/cosmoTransitions/finiteT.py
    
        Jb_spline:
        INPUTS (X,n=0)
                (np.array, int)
        OUTPUTS Jb
                (np.array)
                                
        
    VARIABLE DICTIONARY:
        > sigma (float) = order parameter for the phase transition,
        > T (float) = temperature,
        
        > _m2Sig (float) = squared mass of the sigma particle,
        > _m2Eta (float) = squared mass of the eta prime particle,
        > etc
        > mSq (dict) = {Particle name (string), effective mass squared (lambda function (sigma, T) -> float), number of degrees of freedom (int)}
        
        
        > N (int) = number of colours,
        > F (int) = number of flavours,
        > Polyakov (bool) = include Polyakov corrections.
        
        > CsakiTerm (bool) = True if this is for the Csaki case, False if the normal case.
        > detPow (int) = Power of the chiral symmetry breaking determinant term. Modified in the Csaki/Normal case as per Eq 8 in the draft.
        
        > m2, c, ls/lambdas, la/lambdaa (float) = Lagrangian parameters. See Eq 4 of 2309.16755.
        
        > Tn, alpha, betaH, vW = Parameters to describe the gravitational wave.
        > message = Error message code from running GravitationalWave.grid, see "Error Codes.txt" for what each code means.
        
'''

#Maximum value of sigma for Martha's interpolator.
GLUONIC_CUTOFF = 1000

class NotImplemented(Exception):
    pass
class NonLinear(Exception):
	pass
class InvalidPotential(Exception):
    pass

#CONVERT between physical inputs and Lagrangian parameters

def masses_to_lagrangian_Csaki(_m2Sig, _m2Eta, _m2X, fPI, N, F):
    #See appendix D in draft for these formulae.
    if F/N<2:
        raise NonLinear(f"Lagrangian is non-linear with F={F} and N={N}.")    

    m2 = (_m2Sig/2) - (_m2Eta/2)*(N/F)*(4-F/N)
    c = _m2Eta*N**2 * fPI**(2-F/N)
    ls = (_m2Sig - _m2Eta*(N/F)*(2-F/N))/fPI**2
    la = (_m2X - _m2Eta*(N/F)*(N+1))/fPI**2
    
    #UNITARITY BOUNDS AND EFT EXPANSION
    if int(round(F/N))<4:
        if np.abs(ls + la*F**2*(F**2-2))>8*np.pi/3:
            print(f'Point is Invalid as unitarity is violated')
            return (None,None,None,None)
        if np.abs(c*fPI**(F/N-4))>4*np.pi:
            print(f'Point is Invalid as EFT expansion does not appear to converge')
            return (None,None,None,None)
        
    #CHECKS
    V = lambda sigma: -m2*sigma**2/2 - c*sigma**(F/N)/F**2 + ls*sigma**4/8
    dV = lambda sigma: -m2*sigma - c*sigma**(F/N-1)/(F*N) + ls*sigma**3/2
    ddV = lambda sigma: -m2 - c*(F/N-1)*sigma**(F/N-2)/(F*N) + 3*ls*sigma**2/2
    
    if int(round(F/N))==2:
        #CHECKS
        if ls<0:
            print(f'Point is Invalid as $\lambda_\sigma$<0')
            return (None,None,None,None)
    
        
    else:
        raise NotImplemented(f"F/N={F/N} not implemented yet in masses_to_lagrangian function")
    


    if V(fPI)>V(0):
        print(f'Point is Invalid as symmetric point is true minimum')
        return (None,None,None,None)
    if dV(fPI)>fPI:
        print(dV(fPI))
        print(f'Point is Invalid as fPI = {fPI} does not minimize potential')
        plt.title('Csaki')
        plt.plot(np.arange(0,2*fPI),V(np.arange(0,2*fPI)),label='V')
        plt.plot(np.arange(0,2*fPI),dV(np.arange(0,2*fPI)),label='dV')
        plt.legend()
        plt.show()
        return (None,None,None,None)
    if ddV(fPI)<0:
        print(f'Point is Invalid as fPI is not minimum')
        return (None,None,None,None)
    
    return (m2, c, ls, la)

def masses_to_lagrangian_Normal(_m2Sig, _m2Eta, _m2X, fPI, N, F):
    #See appendix D in draft for these formulae.
    m2 = (_m2Sig/2) - (_m2Eta/2)*(1/F)*(4-F)
    c = _m2Eta * fPI**(2-F)
    ls = (_m2Sig - _m2Eta*(1/F)*(2-F))/fPI**2
    la = (_m2X - _m2Eta*(1/F)*(2))/fPI**2

    #UNITARITY BOUNDS AND EFT EXPANSION
    if int(round(F))<4:
        if np.abs(ls + la*F**2*(F**2-2))>8*np.pi/3:
            print(f'Point is Invalid as unitarity is violated')
            return (None,None,None,None)
        if np.abs(c*fPI**(F-4))>4*np.pi:
            print(f'Point is Invalid as EFT expansion does not appear to converge')
            return (None,None,None,None)

    #CHECKS
    V = lambda sigma: -m2*sigma**2/2 - c*sigma**(F)/F**2 + ls*sigma**4/8
    dV = lambda sigma: -m2*sigma - c*sigma**(F-1)/(F) + ls*sigma**3/2
    ddV = lambda sigma: -m2 - c*(F-1)*sigma**(F-2)/(F) + 3*ls*sigma**2/2
    if F==2:
        if ls<0:
            print(f'Point is Invalid as highest power term is negative')
            return (None,None,None,None)
        
    elif F==3:
        if ls<0:
            print(f'Point is Invalid as highest power term is negative')
            return (None,None,None,None)
        
    elif F==4:
        if ls-c<0:
            print(f'Point is Invalid as highest power term is negative')
            return (None,None,None,None)
        
    #elif F>4:
    #    if dV(2*np.pi*fPI)<0:
    #        print(f'Point is Invalid as potential is unbounded')
    #        return (None,None,None,None)

    #else:
    #    raise NotImplemented(f"F/N={F/N} not implemented yet in masses_to_lagrangian function")




    if V(fPI)>V(0):
        print(f'Point is Invalid as symmetric point is true minimum')
        return (None,None,None,None)
    if round(dV(fPI))>fPI:
        print(dV(fPI))
        print(f'Point is Invalid as fPI = {fPI} does not minimize potential')
        plt.title('Normal')
        plt.plot(np.arange(0,2*fPI),V(np.arange(0,2*fPI)),label='V')
        plt.plot(np.arange(0,2*fPI),dV(np.arange(0,2*fPI)),label='dV')
        plt.legend()
        plt.show()
        return (None,None,None,None)
    if ddV(fPI)<0:
        print(f'Point is Invalid as fPI is not minimum')
        return (None,None,None,None)

    return (m2, c, ls, la)

class Potential:
    def __init__(self, m2, c, lambdas, lambdaa, N, F, CsakiTerm, Polyakov=True):
		#All the parameters needed to construct the potential.
        self.m2 = m2
        self.c = c
        self.lambdas = lambdas
        self.lambdaa = lambdaa
        self.N = N
        self.F = F
        self.CsakiTerm = CsakiTerm
        self.Polyakov = Polyakov
        
        if m2 is None or c is None or lambdas is None or lambdaa is None or N is None or F is None:
            raise InvalidPotential('Input parameter is None')
        
        
        if not CsakiTerm:
            self.detPow = 1 
        if CsakiTerm:
            self.detPow = N

        
		#Checking to make sure the Lagrangian is linear.
        if self.CsakiTerm:
            frac = self.F/self.N
            if abs(frac - np.round(frac)) > 0.0001:
                raise NonLinear(f"Choice of N = {self.N} and F = {self.F} gives non-linear Lagrangian.")
            
        ##GLUONIC FITS
        data = np.genfromtxt(f'GridDataF{self.F}N{self.N}Corrected.csv', delimiter=',', dtype=float, skip_header=1)
        #self.linear = interpolate.LinearNDInterpolator(data[:,0:2],data[:,2])
        self.num = round(len(data)/2)
        self.T_switch = data[self.num,0]
        self.linear_small = interpolate.SmoothBivariateSpline(data[:self.num,0],data[:self.num,1],data[:self.num,2]/1e7, kx=4,ky=3)
        self.linear_large = interpolate.SmoothBivariateSpline(data[self.num:,0],data[self.num:,1],data[self.num:,2]/1e10, kx=4,ky=3)
  
        

        #A dictionary containing {'Particle Name': [lambda function for the field dependent masses squared for the mesons, 
        #                                            and their respective DoF]}
        if self.F/self.detPow>=2:
            self.mSq = {
                #Sigma Mass	
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

        #Checking validity of the potential.
        elif self.F/self.detPow<2:
            raise InvalidPotential("F/N is too small. Diverging effective masses.")


        else:
            raise NotImplemented(f"F/N={self.F/self.detPow} not implemented yet in mSq")
	
		
    def V(self,sig): 
        ##The tree level, zero temperature potential.
        sig = np.array(sig)	
        return - self.m2 * sig**2/2 - (self.c/self.F**2) * sig**(self.F/self.detPow) + (self.lambdas/8) * sig**4


    def V1T(self,sig,T):
        #Setting to zero at T=0 GeV to avoid any computational divergences.
        sig = np.array(sig)
        if T==0:
            return np.zeros(sig.shape)

        #One-loop, thermal correction to the potential. See https://arxiv.org/pdf/hep-ph/9901312 eq. 212
        return np.reshape((np.sum([n*Jb_spline((m2(sig,T)/T**2)) for m2, n in [self.mSq['Sig'],self.mSq['Eta'],
                                                                        self.mSq['X'],self.mSq['Pi']]],axis=0))*T**4/(2*np.pi**2), sig.shape)



    def _Vg(self, sig, T):
        # Check if input1 or input2 are single numbers (scalars)
        if np.ndim(T) == 0:
            T = [T]  # Treat as a vector of one element
        if np.ndim(sig) == 0:
            sig = [sig]  # Treat as a vector of one element

        # Convert inputs to numpy arrays for easier handling
        vector1 = np.array(sig)
        vector2 = np.array(T)
        
        # Create a matrix to store the results
        matrix = np.zeros((len(vector1), len(vector2)))
        
        # Loop through each element of vector1 and vector2, applying the function
        for i, a in enumerate(vector1):
            for j, b in enumerate(vector2):
                matrix[i, j] = self._Vg_f(a, b)
    
        return np.array(matrix)
            
    def _Vg_f(self, sig, T):
        #Forcing the sigma to be a radial-type coordinate.
        sig = abs(sig)
        
        #The interpolator does a terrible job at these low temperatures. For now, just set to zero as PT usually not at such low temperature.
        if T<90:
            return 0
        
        #T_switch is temperature for switching between 'small' temperatures & 'large'.
        if T<self.T_switch:
            #Set the V_gluonic contribution to be constant after a large value of sigma where Martha's code has cut off: GLUONIC_CUTOFF.
            if sig>GLUONIC_CUTOFF:
                #NOTE we have to rescale the interpolating function.
                return self.linear_small.ev(T,GLUONIC_CUTOFF)*1e7 
            else:
                return self.linear_small.ev(T,sig)*1e7
        else:
            if sig>GLUONIC_CUTOFF:
                return self.linear_large.ev(T,GLUONIC_CUTOFF)*1e10
            else:
                return self.linear_large.ev(T,sig)*1e10
            
            

    def VGluonic(self, sig, T):
        #Wrapper function for _Vg.
        sig = np.array(sig)
        #Avoiding any computational issues with the zero temperature limit.
        if T==0:
            return np.zeros(sig.shape)
        return np.reshape(self._Vg(sig, T),sig.shape)
	

    def Vtot(self,sig,T):
    #This finds the total (one-loop/tree level) thermal effective potential.
        sig = np.array(sig)
              
        if self.Polyakov:
            return self.VGluonic(sig, T) + self.V(sig) + self.V1T(sig,T).real
	
        else:
            #Ignoring Polyakov loops.
            return self.V(sig) + self.V1T(sig,T).real

    def VIm(self,sig,T):
        #This finds the inaginary part of the effective potential.
        #Setting to zero at T=0 GeV to avoid any computational divergences.
        sig = np.array(sig)
        if T==0:
            return np.zeros(sig.shape)

        #One-loop, thermal correction to the potential, using CosmoTransitions Exact Function.
        return np.reshape((np.sum([n*fT.Jb_exact2((m2(sig,T)/T**2)) for m2, n in [self.mSq['Sig'],self.mSq['Eta'],
                                                                        self.mSq['X'],self.mSq['Pi']]],axis=0))*T**4/(2*np.pi**2), sig.shape).imag




    def dVdT(self,sig,T,eps=0.001):
    #Uses finite difference method to fourth order. Takes scalar sigma and T.
        return (self.Vtot(sig,T-2*eps) - 8*self.Vtot(sig,T-eps) + 8*self.Vtot(sig,T+eps) - self.Vtot(sig,T+2*eps)) / (12.*eps)


    def d2VdT2(self,sig,T,eps=0.001):
        #Uses finite difference method to fourth order. Takes scalar sigma and T.
        return (self.dVdT(sig,T-2*eps) - 8*self.dVdT(sig,T-eps) + 8*self.dVdT(sig,T+eps) - self.dVdT(sig,T+2*eps)) / (12.*eps)


    def fSigma(self):
        #Analytic formula for zero-temp vev depends on power of the determinant term detPow. See Mathematica notebook SigmaVev.nb for formulae.
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


    def findminima(self,T,rstart=None,rcounter=1):
        #For a linear sigma model. Returns the minimum away from the origin if it exists, else None.
        if rstart == None:
            rstart = self.fSigma()*.75
        #Roll down to minimum from the RHS:
        res = optimize.minimize(lambda X: self.Vtot(X, T), rstart,method='Nelder-Mead',bounds=[(.5+rcounter*2,self.fSigma()*1.05)])
        #Now check to see if the algorithm succeeded
        if not res.success or res.x[0]<1+rcounter*2:
            #If so, try a new start closer to the axis to avoid overshooting.
            if rcounter<=5:
                if rstart is not None:
                    #Closer to axis by 20%.
                    return self.findminima(T,rstart=rstart*0.8,rcounter=rcounter+1)
                else: return None
            else:
                return None
        #Check the roll didn't find the zero sigma minimum.
        elif res.x[0]<10:
            return None
        else: return res.x[0]

    def deltaV(self,T, rstart=None, num_res=None):
        #Finds the difference between the symmetric and broken minima.
        ##POSITIVE IF SIGMA = 0 IS FALSE MINIMUM/STATIONARY POINT; NEGATIVE IF SIGMA = 0 IS TRUE MINIMUM; NONE IF ONLY ONE MINIMA AT SIGMA = 0.
        if rstart is not None: vT = self.findminima(T, rstart=rstart)
        else: vT = self.findminima(T)
        if vT is not None:
            return + self.Vtot(0, T) - self.Vtot(vT, T)
        else:
            #In case you want a numerical result.
            if num_res:
                return 1e30
            else:
                return None
			



    def	criticalT(self, guessIn=None,prnt=True):
        #Critical temperature is when delta V is zero (i.e. both minima at the same height) THIS HAS TO BE QUITE ACCURATE!
		
        #Scale with which we can compare potential magnitudes (think large values of sigma, V~sigma^4)
        scale = self.fSigma()
		
        #First a coarse scan. Find the minimum deltaV from this initial scan, then do a finer scan later.
        Ts_init = np.linspace(75,scale,num=400); deltaVs_init=[]

        for T in Ts_init:
            #Computing the difference between symmetric and broken minima.
            deltaV =  self.deltaV(T, rstart=scale)
            
            #Basically starting from low T and increasing until deltaV flips signs (i.e. minima have crossed eachother)
            if deltaV is not None and deltaV > 0: deltaVs_init.append([T,deltaV])
            if deltaV is not None and deltaV < 0: break

        deltaVs_init=np.array(deltaVs_init)
		

        if prnt:
            for T,_ in deltaVs_init:
                plt.scatter(self.findminima(T),T)
                plt.scatter(0,T)
                plt.xlabel('Delta V'); plt.ylabel('T')
            plt.show()	

        if len(deltaVs_init)<3:
            print('Coarse scan finds nothing')
            return None
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
                plt.plot(np.linspace(-10,self.fSigma()*.25,num=100),V.Vtot(np.linspace(-10,self.fSigma()*.25,num=100),T)-V.Vtot(0,T),label=f"T={T}")
                if self.findminima(T) is not None:
                    plt.scatter(self.findminima(T), V.Vtot(self.findminima(T),T)-V.Vtot(0,T))
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
        #Imaginary part of the 1 loop potential
        return (self.V1T(sig,T)).imag
    def real(self,sig,T):
        #Real part of the 1 loop potential
        return self.V(sig) + (self.V1T(sig,T)).real




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
    print('hello world')


