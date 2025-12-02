import config
import cosmoTransitions.finiteT as fT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import interpolate, integrate
from itertools import takewhile
import os
from sympy import diff


import DressedMasses
if config.PLOT_RUN:
    from config_debug_plots import debug_plot


'''
    This file does the following:
    -1. "_g_starSM" takes a temperature in GeV and returns the number of active SM Degrees of Freedom at that temperature(TGeV):
    
        _g_starSM:
        INPUTS: (TGeV)
                (float)
        RETURNS: (g_starSM)
                (float)
    
    0. "get_detPow" takes a combination of number of flavours F, number of colours N and termType to produce the power
        on the determinant term ~-2c fPI^{pF-2} (det(\Phi)^p+det(\Phi^\dagger)^p), i.e. detPow = p.
        
        termType can be one of three options:
            "normal" -> detPow=1
            "largeN" -> detPow=1/N
            "AMSB" -> detPow=1/(F-N)            
    
        get_detPow:
        INPUTS: (N, F, termType)
                (int, int, String)
        OUTPUTS: (detPow)
                (float)
    
    1. Converts physical particle masses to Lagrangian parameters in "masses_to_lagrangian".
    
        masses_to_lagrangian/masses_to_lagrangian: 
        INPUTS:  (_m2Sig, _m2Eta, _m2X, _m2Pi, N, F)
                (float, float, float, float, int, int)
        OUTPUTS: (m2, c, ls, la)
        RAISES: NotImplemented (as each N, F and p case must be handcoded), BoundedFromBelow (if potential is not bounded from below) 
                and NonTunnelling (if tree-level true minimum is not separated from \sigma=0 axis).
        
    2. Creates a 'Potential' class, which has the following properties:
        Potential:
        INPUTS: m2, c, lambdas, lambdaa, N, F, detPow, Polyakov=True, xi=1, fSIGMA=None:
                (float, float, float, float, int, int, float, bool, float, float)
        ADDITIONAL PROPERTIES:  mSq, MSq, tc, minT, _g_star
                                (dictionary, dictionary, float, float, lambdaFunction)
        RAISES: NonLinear (If F/N is not integer), InvalidPotential (if any required input variables are None), 
                BadDressedMassConvergence (if criteria are met signifying DressedMasses.py has bad convergence in PT region).
        
        SET FUNCTIONS:
        a."setMsq" is used by DressedMasses.py to set the thermal effective masses.
        
            setMSq:
            INPUTS: (dressedMasses)
                    (dict[str, RectBivariateSpline])
            OUTPUTS: None
                                
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
                
        b. "VGluonic". Finds the gluonic potential given N and F. Needs 'VGluonicData' files. Sets confining critical temperature to xi*fPi.
            
            VGluonic:
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
            RAISES: NotImplemented (as each N, F case must be handcoded), InvalidPotential (If there are not enough solutions to tree-level potential etc)
            
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
            INPUTS (guessIn=None,minT=None)
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
        
        > detPow (float) = Power of the chiral symmetry breaking determinant term. Modified in the Normal/Large N/AMSB case as per the draft.
        
        > m2, c, ls/lambdas, la/lambdaa (float) = Lagrangian parameters. See Eq 4 of 2309.16755.
        
        > Tn, alpha, betaH, vW = Parameters to describe the gravitational wave.
        > message = Error message code from running GravitationalWave.grid, see "Error Codes.txt" for what each code means.
        
'''

#Calculate g_star
data_array = np.loadtxt('gstar_data.dat')
TsMeV = data_array[:,0]
g_stars = data_array[:,1]
interp_g_starSM = interpolate.interp1d(TsMeV, g_stars)    



def _g_starSM(TGeV):
    TGeV = np.array(TGeV) 
    result = np.empty_like(TGeV)
    mask = TGeV * 1000 >= max(TsMeV)
    result[mask] = max(g_stars)
    result[~mask] = interp_g_starSM(TGeV[~mask] * 1000)
    return result
 
 

class NotImplemented(Exception):
    pass
class NonLinear(Exception):
	pass
class InvalidPotential(Exception):
    pass
class NonUnitary(Exception):
    pass
class BoundedFromBelow(Exception):
    pass
class NonTunnelling(Exception):
    pass
class BadDressedMassConvergence(Exception):
    pass

#CONVERT between physical inputs and Lagrangian parameters

def get_detPow(N, F, termType):
    if termType=="Normal":
        return 1
    elif termType=="largeN":
        return 1/N
    elif termType=="AMSB":
        return 1/(F-N)
    elif termType=="Charge2Inst":
        return 2
    
    else:
        raise InvalidPotential("Check what have you written as termType. It's wrong.")

def masses_to_lagrangian(_m2Sig, _m2Eta, _m2X, fPI, N, F, detPow):
    if F*detPow<2 or abs(F*detPow-np.round(F*detPow))>config.TOL:
        raise NonLinear(f"Lagrangian is non-linear with F={F}, N={N} and detPow={detPow}.")    

    m2 = (_m2Sig/2) - (_m2Eta/2)*(1/(F*detPow))*(4-F*detPow)
    c = _m2Eta/detPow**2 * np.power(fPI,2-F*detPow)
    ls = (_m2Sig - _m2Eta*(1/(F*detPow))*(2-F*detPow))/fPI**2
    la = (_m2X - 2*_m2Eta*(1/(F*detPow)))/fPI**2
    
    print(f'm2={m2},c={c},ls={ls},la={la}')
    

    #CHECKS
    V = lambda sigma: -m2*sigma**2/2 - c*sigma**(F*detPow)/F**2 + ls*sigma**4/8
    dV = lambda sigma: -m2*sigma - c*detPow*sigma**(F*detPow-1)/F + ls*sigma**3/2
    ddV = lambda sigma: -m2 - c*detPow*(F*detPow-1)*sigma**(F*detPow-2)/F + 3*ls*sigma**2/2
    

    #CHIRAL SYMMETRY BREAKING:
    #Checks to ensure Chiral symmetry is spontaneously broken.
    if V(fPI)>V(0)+config.TOL:
        raise NonTunnelling('Symmetric minimum is true minimum')
    
    if abs(dV(fPI))>config.TOL:
        print(f'Point is Invalid as dV(fPI) = {dV(fPI)} different of 0')
        raise NonTunnelling('Symmetric minimum is true minimum')

    if ddV(fPI)<-config.TOL:
        print(f'Point is Invalid as fPI is not minimum')
        raise NonTunnelling('Point is Invalid as fPI is not stable')
    
    #BOUNDEDNESS FROM BELOW:                    
    #Debugging Plot:
    if config.PLOT_RUN:
        if V(4*np.pi*fPI)<V(fPI):
            plt.plot(np.linspace(0,4*np.pi*fPI),V(np.linspace(0,4*np.pi*fPI)))
            plt.title('Not BfB')
        else:
            plt.plot(np.linspace(0,4*np.pi*fPI),V(np.linspace(0,4*np.pi*fPI)))
            plt.title('BfB')
        debug_plot(name="BoundedfromBelowTest", overwrite=False)

    #Boundedness from below condition (eq 3.12):  
    if V(4*np.pi*fPI)<V(fPI):
        print(f'Point is Invalid as potential unbounded from below by cutoff')
        raise BoundedFromBelow('Point is not bounded from below by cutoff')
    
    return (m2, c, ls, la)

class Potential:
    def __init__(self, m2, c, lambdas, lambdaa, N, F, detPow, Polyakov=True, xi=1, fSIGMA=None):
		#All the parameters needed to construct the potential.
        self.m2 = m2
        self.c = c
        self.lambdas = lambdas
        self.lambdaa = lambdaa
        self.N = N
        self.F = F
        self.detPow = detPow
        self.Polyakov = Polyakov
        if self.Polyakov:
            self.xi = xi #Confining Critical Temperature scaling.
        self.tc = None
        self.minT = None #Smallest temperature it makes sense to talk about the potential.
        
        #Degrees of freedom in the plasma.
        self._g_star = lambda TGeV: _g_starSM(TGeV) + 2*(self.N**2 -1) + 7/8 * (self.F * self.N * 4) 
        

        if m2 is None or c is None or lambdas is None or lambdaa is None or N is None or F is None:
            raise InvalidPotential('Input parameter is None')
    

        if fSIGMA is not None:
            self.fSIGMA = fSIGMA
        else:
            try:
                self.fSIGMA = self.fSigma()
            except(InvalidPotential):
                print("Imaginary fSigma")
        

		#Checking to make sure the Lagrangian is linear.
        if abs(self.F*self.detPow-np.round(self.F*self.detPow)) > 1e-6:
            raise NonLinear(f"Choice of N = {self.N}, F = {self.F}, detPow = {detPow} gives non-linear Lagrangian.")


        if Polyakov:
            ##GLUONIC FITS
            #Must run VGluonicData Notebook first.
            gluonicdata = np.genfromtxt(f'VGluonicData/VGluonicDataF{int(self.F)}N{int(self.N)}.csv', delimiter=',', dtype=float, skip_header=0)
            TTcs = np.genfromtxt(f'VGluonicData/VGluonicDataF{int(self.F)}N{int(self.N)}TTcs.csv', delimiter=',', dtype=float, skip_header=0)
            sigTcs = np.genfromtxt(f'VGluonicData/VGluonicDataF{int(self.F)}N{int(self.N)}mqTcs.csv', delimiter=',', dtype=float, skip_header=0)
            
            self.VGluonicTc = interpolate.RectBivariateSpline(TTcs,sigTcs,gluonicdata)

        #A dictionary containing {'Particle Name': [lambda function for the field dependent masses squared for the mesons, 
        #                                            and their respective DoF]}
        
        #The tree level masses
        if self.F*self.detPow>=2:
            self.mSq = {
                #Sigma Mass	
                'Sig': [lambda sig: - self.m2 
						- self.c * self.detPow / (self.F) * ( self.F*self.detPow - 1 ) * sig**(self.F*self.detPow - 2)
						+ (3/2) * self.lambdas * sig ** 2,
                        1.],
				#Eta Prime Mass
                'Eta': [lambda sig: - self.m2 
						+ self.c *self.detPow / (self.F) * ( self.F*self.detPow - 1 ) * sig**(self.F*self.detPow - 2)
						+ (1/2) * self.lambdas * sig ** 2,
                        1.],
				#X Mass
				'X': [lambda sig: - self.m2 
						+ self.c * self.detPow / self.F * sig**(self.F*self.detPow - 2)
						+ (1/2) * (self.lambdas+2*self.lambdaa) * sig ** 2,
						self.F**2 - 1],
				#Pi Mass
				'Pi': [lambda sig: - self.m2 
						- self.c * self.detPow / self.F * sig**(self.F*self.detPow - 2)
						+ (1/2) * self.lambdas * sig ** 2,
						self.F**2 - 1]
						}
        
        self.MSq = None   

        #Temperature dependent masses:
        try:
            DressedMasses.SolveMasses(self)
        except BadDressedMassConvergence as e:
            raise e
        
        #Ensure the new critical temperature is set.
        self.tc = self.criticalT()
        if self.tc is not None and self.minT is not None:
            if self.minT > self.tc:
                raise BadDressedMassConvergence("Minimum converging T is greater than tc.")
        
              
    def setMSq(self, dressedMasses):
        self.MSq = {
            #Sigma Mass	
            'Sig': [lambda sig, T: dressedMasses['Sig'].ev(T,sig)*self.fSIGMA,
                    1.],
            #Eta Prime Mass
            'Eta': [lambda sig, T: dressedMasses['Eta'].ev(T,sig)*self.fSIGMA,
                    1.],
            #X Mass
            'X': [lambda sig, T: dressedMasses['X'].ev(T,sig)*self.fSIGMA,
                    self.F**2 - 1],
            #Pi Mass
            'Pi': [lambda sig, T: dressedMasses['Pi'].ev(T,sig)*self.fSIGMA,
                    self.F**2 - 1]
        }
		
    def V(self,sig): 
        ##The tree level, zero temperature potential.
        sig = np.array(sig)	
        return - self.m2 * sig**2/2 - (self.c/self.F**2) * sig**(self.F*self.detPow) + (self.lambdas/8) * sig**4


    def V1T(self,sig,T):
        #Setting to zero at T=0 GeV to avoid any computational divergences.
        sig = np.array(sig)
        if T==0:
            return np.zeros(sig.shape)
        
        #One-loop, thermal correction to the potential. See https://arxiv.org/pdf/hep-ph/9901312 eq. 212 or 
        return np.reshape((np.sum([n*Jb_spline(M2(sig,T)/T**2)-(1/4)*(M2(sig,T)-M2(sig,0))*Ib_spline(M2(sig,T)/T**2)/T**2 for M2, n in [self.MSq['Sig'],self.MSq['Eta'],
                                                                        self.MSq['X'],self.MSq['Pi']]],axis=0))*T**4/(2*np.pi**2), sig.shape)

            

    def VGluonic(self, sig, T):
        #Runs VGluonic interpolator. 
        sig = np.array(sig)
        _tconfinement = self.xi*self.fSIGMA #eq 2.9
        return np.reshape(self.VGluonicTc.ev(T/_tconfinement,sig/_tconfinement)*_tconfinement**4,sig.shape)
	

    def Vtot(self,sig,T):
    #This finds the total (one-loop/tree level) thermal effective potential.
        sig = np.array(sig)
              
        if self.Polyakov:
            return self.VGluonic(sig, T).real + self.V(sig).real + self.V1T(sig,T).real
	
        else:
            #Ignoring Polyakov loops.
            return self.V(sig) + self.V1T(sig,T).real


    #do we need such level of precision in derivatives?
    def dVdT(self,sig,T,eps=0.001):
    #Uses finite difference method to fourth order. Takes scalar sigma and T.
        return (self.Vtot(sig,T-2*eps) - 8*self.Vtot(sig,T-eps) + 8*self.Vtot(sig,T+eps) - self.Vtot(sig,T+2*eps)) / (12.*eps)


    def d2VdT2(self,sig,T,eps=0.001):
        #Uses finite difference method to fourth order. Takes scalar sigma and T.
        return (self.dVdT(sig,T-2*eps) - 8*self.dVdT(sig,T-eps) + 8*self.dVdT(sig,T+eps) - self.dVdT(sig,T+2*eps)) / (12.*eps)
    

    def fSigma(self):
        #UNVERIFIED FUNCTION!
        #Analytic formula for zero-temp vev depends on power of the determinant term detPow. See Mathematica notebook SigmaVev.nb for formulae.
        if int(self.F*self.detPow) == 1:
            x = 9 * self.c * self.F**4 * self.lambdas**2 + np.sqrt(3)*np.sqrt( abs(self.F**8 * self.lambdas**3 * (-8 * self.F**4 *self.m2**3 + 27*self.c**2*self.lambdas)) )
            return np.real((2 * 3**(1/3) * self.F**4 * self.m2 * self.lambdas + x**(2/3)) / (3**(2/3)*self.F**2*self.lambdas * x**(1/3)))
		
        elif int(self.F*self.detPow) == 2:
            if self.m2 + self.c*2/self.F**2>0:
                return np.sqrt(2) * np.sqrt(self.m2 + self.c*2/self.F**2)/np.sqrt(self.lambdas)
            else:
                raise InvalidPotential("Not enough minima")
		
        elif int(self.F*self.detPow) == 3: #DOUBLE-CHECK POWERS OF p
            return (self.c*self.detPow + np.sqrt(self.c**2*self.detPow**2 + 2*self.F**2*self.m2*self.lambdas))/(self.F*self.lambdas)
		
        elif int(self.F*self.detPow) == 4:
            if self.lambdas - 4*self.c/(self.detPow*self.F) > 0:
                return 2*np.sqrt(self.m) / np.sqrt(self.lambdas - 4*self.c/(self.detPow*self.F))
            else:
                raise InvalidPotential(f"Unbounded potential for F={self.F}")
        

        elif int(self.F*self.detPow) == 6:
            return np.sqrt( - (-3*self.lambdas + np.sqrt(-24*self.c*self.m2 + 9*self.lambdas**2))/(2*self.c))
        else:
            raise NotImplemented(f"F*detPow={int(self.F*self.detPow)} not implemented yet in fSigma")



    def findminima(self,T,rstart=None,rcounter=1):
        #For a linear sigma model. Returns the minimum away from the origin if it exists, else None.
        if rstart == None:
            rstart = self.fSIGMA*.8 
    
        
        #Roll down to minimum from the RHS:
        res = optimize.minimize(lambda X: self.Vtot(X, T), rstart,method='Nelder-Mead',bounds=[((0.5-.05*rcounter)*self.fSIGMA,self.fSIGMA*1.05)])
 
    
        #Now check to see if the algorithm succeeded
        if not res.success or abs(res.x[0]-(0.5-.05*rcounter)*self.fSIGMA)/self.fSIGMA < config.TOL or res.x[0]<(0.5-.05*rcounter)*self.fSIGMA:
            #If so, try a new start closer to the axis to avoid overshooting.
            if rcounter<=4:
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
			


    def	criticalT(self, guessIn=None,minT=None):
        prnt=config.PRNT_RUN
        plot=config.PLOT_RUN
        if self.tc is not None:
            return self.tc
        #Critical temperature is when delta V is zero (i.e. both minima at the same height) THIS HAS TO BE QUITE ACCURATE!
		
        #Scale with which we can compare potential magnitudes (think large values of sigma, V~sigma^4)
        scale = self.fSIGMA
        
        if minT is not None:
            minTemp=minT
        else:
            minTemp = np.sqrt(scale)
                    
		
        #First a coarse scan. Find the minimum deltaV from this initial scan, then do a finer scan later.
        Ts_init = np.linspace(minTemp,scale*config.TMULT,num=450); deltaVs_init=[]

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
                plt.xlabel('RHS Minima'); plt.ylabel('T')
            debug_plot(name="debug", overwrite=False)
            #plt.show()	

        if len(deltaVs_init)<3:
            print('Coarse scan finds nothing')
            return None
        
        #This is the temperature with deltaV closest to zero:
        T_init = deltaVs_init[-1,0]
        if prnt:
            plt.plot(deltaVs_init[:,1], deltaVs_init[:,0])
            plt.xlabel('Delta V'); plt.ylabel('Temperature')
            debug_plot(name="debug", overwrite=False)

            print(f"Coarse grain scan finds {T_init} being closest Delta V to 0")

        #Some plotting functions for a PLOT_RUN:
        def plotV(V, Ts):
            for T in Ts:
                plt.plot(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=100),V.Vtot(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=100),T)-V.Vtot(0,T),label=f"T={T}")
                if self.findminima(T) is not None:
                    plt.scatter(self.findminima(T), V.Vtot(self.findminima(T),T)-V.Vtot(0,T))
            plt.legend()
            debug_plot(name="debug", overwrite=False)
            
        def splitV(V, T):
            plt.plot(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=1000),V.Vtot(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=1000),T)-V.Vtot(0,T),label=f"VTot at Tc={T}")
            plt.plot(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=1000),V.V1T(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=1000),T)-V.V1T(0,T),label=f"V1T at Tc={T}")
            plt.plot(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=1000),V.V(np.linspace(-10,self.fSIGMA*config.SIGMULT,num=1000))-V.V(0),label=f"Vtree")
            plt.legend()
            debug_plot(name="debug", overwrite=False)
	

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
            debug_plot(name="debug", overwrite=False)
            #plt.show()
		
		#Choose a 'guess' to be slightly closer to the higher temperature range.
        if guessIn==None:
            guess = (max(deltaVs[:,0])-min(deltaVs[:,0]))*0.85 + min(deltaVs[:,0])
        else: guess = guessIn


        #Minimise interpolated function (two methods in case one fails)
        res = optimize.minimize(lambda x: abs(func(x)), guess,bounds=[(min(deltaVs[:,0]),max(deltaVs[:,0])*1.2)])
        if prnt: print(res)
        if res.success and res.fun<scale**3/2 and self.findminima(res.x[0]) is not None:
			
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
            if res.success and res.fun<scale**3/2 and self.findminima(res.x[0]) is not None:
                if prnt: plotV(self, [res.x[0]*0.99,res.x[0],res.x[0]*1.01])
                if prnt: splitV(self, res.x[0])
                
                self.tc = res.x[0]
                return res.x[0]
            else:
                return None






##SPLINE FIT FOR Jb. Code modified slightly from https://arxiv.org/abs/1109.4189 and https://github.com/clwainwright/CosmoTransitions 
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


def _Jb_exact2_Im(theta):
    # Note that this is a function of theta so that you can get negative values. Input is (m/T)^2.
    if theta >= 0:
        return 0
    else:
        f1 = lambda y: y*y*np.arctan(np.sin(np.sqrt(+np.abs(theta+y*y)))/(1-np.cos(np.sqrt(+np.abs(theta+y*y)))))
        
        return integrate.quad(f1, 0, abs(theta)**.5)[0]
    

##SPLINE FIT FOR Ib

dat = np.genfromtxt(f'IBData.csv', delimiter=',', dtype=float, skip_header=1)

_xb, _yb = dat[:,0], dat[:,1]
# Spline fitting, Ib
_xbmin = -10
_xbmax = 1.41e3

_tckb_positive = interpolate.interp1d(_xb[_xb>=0], _yb[_xb>=0], kind='cubic', fill_value="extrapolate")
_tckb_negative = interpolate.interp1d(_xb[_xb<=0], _yb[_xb<=0], kind='cubic', fill_value="extrapolate")

def Ib_spline(X):
    """Ib interpolated from a saved spline. Input is (m/T)^2."""
    X = np.array(X)
    x = X.ravel()
    
    y = [_tckb_positive(xi) if xi>=0 else _tckb_negative(xi) for xi in x]
    y = np.array(y)

    y[x < _xbmin] = _tckb_negative(_xbmin)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)
		
if __name__ == "__main__":
    print('hello world') 
    