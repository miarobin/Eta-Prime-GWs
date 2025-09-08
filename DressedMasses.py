import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from scipy import optimize, differentiate
import time
import matplotlib.pyplot as plt
from scipy import interpolate
import csv
import Potential2
import cosmoTransitions
from IBInterpolation import IB


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"]= 12


def SolveMasses(V, plot=False):
    #ADD T=0 HANDLING!! 
    TRange = np.linspace(0,V.fSIGMA*1.5,num=150)
    sigmaRange = np.linspace(0.01, V.fSIGMA*1.25,num=150)
    
    MSqSigData = np.zeros((len(TRange),len(sigmaRange)))
    MSqEtaData = np.zeros((len(TRange),len(sigmaRange)))
    MSqPiData = np.zeros((len(TRange),len(sigmaRange)))
    MSqXData = np.zeros((len(TRange),len(sigmaRange)))
    RMS = np.zeros((len(TRange),len(sigmaRange))); counter=0; failPoints = []
    
    for i,T in enumerate(TRange):
        for j,sigma in enumerate(sigmaRange):
            if T==0:
                MSqSigData[i,j] = V.mSq['Sig'][0](sigma)
                MSqEtaData[i,j] = V.mSq['Eta'][0](sigma)
                MSqPiData[i,j] = V.mSq['Pi'][0](sigma)
                MSqXData[i,j] = V.mSq['X'][0](sigma)
                
            def bagEquations(vars):                

                M_sigma2, M_eta2, M_X2, M_Pi2 = vars

                prefactor = T**2 / (4 * np.pi**2)
                
                #Distinct Feynman rule structures
                c1 = (V.c/V.F**2)*V.fSIGMA**(V.F*V.detPow-4)*(V.F*V.detPow)*(V.F*V.detPow-1)*(V.F*V.detPow-2)*(V.F*V.detPow-3)
                c2 = (V.c/V.F)*V.fSIGMA**(V.F*V.detPow-4)*V.detPow*(V.F*V.detPow-2)*(V.F*V.detPow-3)
                c3 = (V.c/V.F)*V.fSIGMA**(V.F*V.detPow-4)*V.detPow*(V.detPow*V.F**3-4*V.F**2+V.detPow*V.F+6)
                ctilde = (V.F**2-1)


                lhs = np.array([M_sigma2, M_eta2, M_X2, M_Pi2])
                #is just exactly as in the paper
                rhs = np.array([
                    #Sigma Thermal Effective Mass
                    V.mSq['Sig'][0](sigma) + prefactor * (
                        (3 * V.lambdas - c1) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2 - 1) * (V.lambdas + 2 * V.lambdaa) + c2*ctilde) * Ib_spline(M_X2 / T**2)
                        + (V.lambdas + c1) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2 - 1) * V.lambdas - c2*ctilde) * Ib_spline(M_Pi2 / T**2)),

                    V.mSq['Eta'][0](sigma) + prefactor * (
                        (3 * V.lambdas - c1) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2 - 1) * (V.lambdas + 2 * V.lambdaa) + c2*ctilde) * Ib_spline(M_Pi2 / T**2)
                        + (V.lambdas + c1) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2 - 1) * V.lambdas - c2*ctilde) * Ib_spline(M_X2 / T**2)),
                        
                    V.mSq['X'][0](sigma) + prefactor * (
                        (V.lambdas + 2 * V.lambdaa + c2) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2+1)*V.lambdas + (V.F**2-4)*V.lambdaa - c3) * Ib_spline(M_X2 / T**2)
                        + (V.lambdas - c2) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * Ib_spline(M_Pi2 / T**2)),#RATTI SAYS F^2-1 SANNINO F^2+1!!

                    V.mSq['Pi'][0](sigma) + prefactor * (
                        (V.lambdas + 2 * V.lambdaa + c2) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2+1)*V.lambdas + (V.F**2-4)*V.lambdaa - c3) * Ib_spline(M_Pi2 / T**2)
                        + (V.lambdas - c2) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * Ib_spline(M_X2 / T**2))#RATTI SAYS F^2-1 SANNINO F^2+1!!
                ])

                bagEquations.lhs = lhs
                bagEquations.rhs = rhs
                

                return lhs - rhs  # residuals = LHS - RHS
            

            def jac(vars):
                if T==0:
                    return np.array([[1.,0.,0.,0.],
                                    [0.,1.,0.,0.],
                                    [0.,0.,1.,0.],
                                    [0.,0.,0.,1.]])
                
                M_sigma2, M_eta2, M_X2, M_Pi2 = vars
                prefactor = T**0 / (4 * np.pi**2)

                #Distinct Feynman rule structures
                c1 = (V.c/V.F**2)*V.fSIGMA**(V.F*V.detPow-4)*(V.F*V.detPow)*(V.F*V.detPow-1)*(V.F*V.detPow-2)*(V.F*V.detPow-3)
                c2 = (V.c/V.F)*V.fSIGMA**(V.F*V.detPow-4)*V.detPow*(V.F*V.detPow-2)*(V.F*V.detPow-3)
                c3 = (V.c/V.F)*V.fSIGMA**(V.F*V.detPow-4)*V.detPow*(V.detPow*V.F**3-4*V.F**2+V.detPow*V.F+6)
                ctilde = (V.F**2-1)
                
                res = - prefactor * np.array([
                    #dM2Sig
                    [(3 * V.lambdas - c1) * dIb_spline(M_sigma2 / T**2) - 1/prefactor, 
                    (V.lambdas + c1) * dIb_spline(M_eta2 / T**2),
                    ((V.F**2 - 1) * (V.lambdas + 2 * V.lambdaa) + c2*ctilde) * dIb_spline(M_X2 / T**2),
                    ((V.F**2 - 1) * V.lambdas - c2*ctilde) * dIb_spline(M_Pi2 / T**2)],
                    #dM2Eta
                    [(V.lambdas + c1) * dIb_spline(M_sigma2 / T**2),
                    (3 * V.lambdas - c1) * dIb_spline(M_eta2 / T**2) - 1/prefactor,
                    ((V.F**2 - 1) * V.lambdas - c2*ctilde) * dIb_spline(M_X2 / T**2),
                    ((V.F**2 - 1) * (V.lambdas + 2 * V.lambdaa) + c2*ctilde) * dIb_spline(M_Pi2 / T**2)],
                    #dM2X
                    [(V.lambdas + 2 * V.lambdaa + c2) * dIb_spline(M_sigma2 / T**2),
                    (V.lambdas - c2) * dIb_spline(M_eta2 / T**2),
                    ((V.F**2+1)*V.lambdas + (V.F**2-4)*V.lambdaa - c3) * dIb_spline(M_X2 / T**2) - 1/prefactor,
                    ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * dIb_spline(M_Pi2 / T**2)],#RATTI SAYS F^2-1 SANNINO F^2+1!!
                    #dM2Pi
                    [(V.lambdas - c2) * dIb_spline(M_sigma2 / T**2),
                    (V.lambdas + 2 * V.lambdaa + c2) * dIb_spline(M_eta2 / T**2),
                    ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * dIb_spline(M_X2 / T**2),
                    ((V.F**2+1)*V.lambdas + (V.F**2-4)*V.lambdaa - c3) * dIb_spline(M_Pi2 / T**2) - 1/prefactor]#RATTI SAYS F^2-1 SANNINO F^2+1!!
                    ])

                return res
                     

            # Initial guess (MW adjusted initial guess as it helps improve convergence.)
            initial_guess = [V.mSq['Sig'][0](sigma) + (T**2/24)*(3*V.lambdas - (V.c*V.detPow/V.F)*(V.detPow*V.F-1)*(V.detPow*V.F-2)*(V.detPow*V.F-3)*sigma**(V.detPow*V.F-4)), 
                             V.mSq['Eta'][0](sigma) + (T**2/24)*(V.lambdas + (V.c*V.detPow/V.F)*(V.detPow*V.F-1)*(V.detPow*V.F-2)*(V.detPow*V.F-3)*sigma**(V.detPow*V.F-4)),
                              V.mSq['X'][0](sigma)+ (T**2/24)*(V.lambdas + 2*V.lambdaa + (V.c*V.detPow/V.F)*(V.detPow*V.F-2)*(V.detPow*V.F-3)*sigma**(V.detPow*V.F-4)),
                               V.mSq['Pi'][0](sigma) + (T**2/24)*(V.lambdas - (V.c*V.detPow/V.F)*(V.detPow*V.F-2)*(V.detPow*V.F-3)*sigma**(V.detPow*V.F-4))]

            #print(f'T={T},sigma={sigma}')
            #print(f'mine={jac([15,15,15,15])}')
            #print(f'scipy.jacobian={differentiate.jacobian(bagEquations,[15,15,15,15])}')
            
            
            sol = root(bagEquations, initial_guess, jac=jac, method='hybr')
            RMS[i,j]=np.sqrt(np.mean((bagEquations.lhs-bagEquations.rhs)**2)) #RMS
            if sol.success and RMS[i,j]<5*V.fSIGMA:
                M_sigma2, M_eta2, M_X2, M_Pi2 = sol.x

                MSqSigData[i,j]=M_sigma2
                MSqEtaData[i,j]=M_eta2
                MSqXData[i,j]=M_X2
                MSqPiData[i,j]=M_Pi2
                

            else:
                #Try with numerical jacobian as well:
                sol = root(bagEquations, initial_guess, method='hybr')
                if sol.success and RMS[i,j]<5*V.fSIGMA:
                    M_sigma2, M_eta2, M_X2, M_Pi2 = sol.x

                    MSqSigData[i,j]=M_sigma2
                    MSqEtaData[i,j]=M_eta2
                    MSqXData[i,j]=M_X2
                    MSqPiData[i,j]=M_Pi2
                else:
                        
                    if plot:
                        print(f"Root finding did not converge well for T={T} and sigma={sigma}")
                        print(sol.message)
                    MSqSigData[i,j]=None
                    MSqEtaData[i,j]=None
                    MSqXData[i,j]=None
                    MSqPiData[i,j]=None
                    
                    failPoints.append([sigma, T])
                    

    X,Y=np.meshgrid(TRange,sigmaRange) 
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    valuesSigma = MSqSigData.ravel()
    valuesEta = MSqEtaData.ravel()
    valuesX = MSqXData.ravel()
    valuesPi = MSqPiData.ravel()
    
    _MSqSigData = interpolate.griddata(points[np.isfinite(valuesSigma)],valuesSigma[np.isfinite(valuesSigma)],(X,Y))
    _MSqEtaData = interpolate.griddata(points[np.isfinite(valuesEta)],valuesEta[np.isfinite(valuesEta)],(X,Y))
    _MSqXData = interpolate.griddata(points[np.isfinite(valuesX)],valuesX[np.isfinite(valuesX)],(X,Y))
    _MSqPiData = interpolate.griddata(points[np.isfinite(valuesPi)],valuesPi[np.isfinite(valuesPi)],(X,Y))
    
    #NB SOMETHING ABOUT THESE AXES IS TOTALLY WRONG
    dressedMasses = {
        #Sigma Mass	
        'Sig': interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqSigData,s=4*V.fSIGMA**3,ky=2,kx=2, maxit=40),
		#Eta Prime Mass
        'Eta': interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqEtaData,s=4*V.fSIGMA**3,ky=2,kx=2, maxit=40),
		#X Mass
		'X': interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqXData,s=4*V.fSIGMA**3,ky=2,kx=2, maxit=40),
		#Pi Mass
		'Pi':  interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqPiData,s=4*V.fSIGMA**3,ky=2,kx=2, maxit=40)
		}
    
    if plot:
        #Plot 1: Data vs interpolated        
        fig, ax = plt.subplots(2,2)
        plt.rcParams['figure.figsize'] = [12, 8]
        
        TIndexSample = [0, 15, 25, 50, 99]
        colours = ["red", "firebrick", "darkorange", "crimson", "rosybrown", "gold", "palevioletred"]
        
        for i,Tindex in enumerate(TIndexSample):
            T = TRange[Tindex]
            ax[0,0].scatter(sigmaRange, MSqSigData[Tindex,:], color=colours[i], label=f"T={T}",alpha=0.66)
            ax[0,1].scatter(sigmaRange, MSqEtaData[Tindex,:], color=colours[i],alpha=0.66)
            ax[1,0].scatter(sigmaRange, MSqXData[Tindex,:], color=colours[i],alpha=0.66)
            ax[1,1].scatter(sigmaRange, MSqPiData[Tindex,:], color=colours[i],alpha=0.66)
            
            ax[0,0].scatter(sigmaRange, _MSqSigData[Tindex,:], color=colours[i], label=f"T={T}",alpha=0.66,marker='1')
            ax[0,1].scatter(sigmaRange, _MSqEtaData[Tindex,:], color=colours[i],alpha=0.66,marker='1')
            ax[1,0].scatter(sigmaRange, _MSqXData[Tindex,:], color=colours[i],alpha=0.66,marker='1')
            ax[1,1].scatter(sigmaRange, _MSqPiData[Tindex,:], color=colours[i],alpha=0.66,marker='1')

            
            ax[0,0].plot(sigmaRange, dressedMasses['Sig'](T, sigmaRange).flatten(), color=colours[i])
            ax[0,1].plot(sigmaRange, dressedMasses['Eta'](T, sigmaRange).flatten(), color=colours[i])
            ax[1,0].plot(sigmaRange, dressedMasses['X'](T, sigmaRange).flatten(), color=colours[i])
            ax[1,1].plot(sigmaRange, dressedMasses['Pi'](T, sigmaRange).flatten(), color=colours[i])

        ax[0,0].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
        ax[0,0].set_ylabel(r'$m_\sigma^2$',fontsize=15)
        
        ax[0,1].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
        ax[0,1].set_ylabel(r'$m_\eta^2$',fontsize=15)

        fig.legend()
        
        #Plot 2: Data grids for effective masses
        plotMassData([MSqSigData,MSqEtaData,MSqXData,MSqPiData], V)
        plt.show()
        
        fig, ax = plt.subplots()
        plt.rcParams['figure.figsize'] = [12, 8]
        
        im0 = ax.contourf(X/V.fSIGMA, Y/V.fSIGMA, RMS)
        cbar = plt.colorbar(im0)
        cbar.set_label(r'RMS $[MeV^2]$',fontsize=14)
        ax.set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
        ax.set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        fig.suptitle(f"$f_\pi={V.fSIGMA}$")
        plt.show()
        
        #if counter>10*len(TRange)*len(sigmaRange)/100:
        #    raise Potential2.InvalidPotential('More than X% of points failed')
    
    #failPoints = []
    #for i,T in enumerate(TRange):
    #    for j,sigma in enumerate(sigmaRange):
    #        if MSqSigData[i,j] is None:
    #            failPoints.append([sigma, T])
    print('########################################################################################')
    print(failPoints)
    print('########################################################################################')
    return dressedMasses, np.array(failPoints)


def plotMassData(massData, V):
        #Make sure these are exactly the same ranges as above!
    TRange = np.linspace(0,V.fSIGMA*1.5,num=150)
    sigmaRange = np.linspace(0.01, V.fSIGMA*1.25,num=150)
    
    MSqSigData=massData[0]
    MSqEtaData=massData[1]
    MSqXData=massData[2]
    MSqPiData=massData[3]

    
    X,Y=np.meshgrid(TRange,sigmaRange)
        
    fig, ax = plt.subplots(2,2)
    plt.rcParams['figure.figsize'] = [12, 8]
        
    im0 = ax[0,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqSigData.T)
    cbar = plt.colorbar(im0)
    cbar.set_label(r'$m_\sigma^2$',fontsize=14)
    ax[0,0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        
    RHS_mins=np.array([V.findminima(T) for T in TRange]) 
    _RHS_mins = RHS_mins[RHS_mins!=None]
    _Ts = TRange[RHS_mins!=None]
    for T,mins in zip(_Ts,_RHS_mins):
        if V.Vtot(mins,T)>V.Vtot(0,T):
            ax[0,0].scatter(T/V.fSIGMA,mins/V.fSIGMA,color='firebrick')
        else:
            ax[0,0].scatter(T/V.fSIGMA,mins/V.fSIGMA,color='blueviolet')
        

    im1 = ax[0,1].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqEtaData.T)
    cbar = plt.colorbar(im1)
    cbar.set_label(r"$m_{\eta'}^2$",fontsize=14)
    ax[0,1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)

    ax[0,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
        
        
    im2 = ax[1,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqXData.T)
    cbar = plt.colorbar(im2)
    cbar.set_label(r'$m_X^2$',fontsize=14)
    ax[1,0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        
    ax[1,0].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
        
        
    im3 = ax[1,1].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqPiData.T)
    cbar = plt.colorbar(im3)
    cbar.set_label(r'$m_\pi^2$',fontsize=14)
    ax[1,1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)

    ax[1,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')

    #Plot a vertical line at the critical temperature.
    tc=V.criticalT(prnt=False)
    if tc is not None:
        ax[0,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[0,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        
        

    fig.suptitle(f"$f_\pi={V.fSIGMA}$")


def plotInterpMasses(massDict, V):
    #Make sure these are exactly the same ranges as above!
    TRange = np.linspace(0,V.fSIGMA*1.5,num=150)
    sigmaRange = np.linspace(0.01, V.fSIGMA*1.25,num=150)
    
    #Data grids for effective masses
    X,Y=np.meshgrid(TRange,sigmaRange)
    
    #Mass functions:
    MSqSig = lambda sig, T: grd['Sig'](T,sig)
    MSqEta = lambda sig, T: grd['Eta'](T,sig)
    MSqX = lambda sig, T: grd['X'](T,sig)
    MSqPi = lambda sig, T: grd['Pi'](T,sig)
    
    #Mass data:
    MSqSigVals = np.array([[MSqSig(T, sigma)[0][0] for T in Ts] for sigma in sigmas])
    MSqEtaVals = np.array([[MSqEta(T, sigma)[0][0] for T in Ts] for sigma in sigmas])
    MSqXVals = np.array([[MSqX(T, sigma)[0][0] for T in Ts] for sigma in sigmas])
    MSqPiVals = np.array([[MSqPi(T, sigma)[0][0] for T in Ts] for sigma in sigmas])
        
    fig, ax = plt.subplots(2,2)
    plt.rcParams['figure.figsize'] = [12, 8]
        
    im0 = ax[0,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqSigVals.T)
    cbar = plt.colorbar(im0)
    cbar.set_label(r'$m_\sigma^2$',fontsize=14)
    ax[0,0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
    
    #Plots location of second minimum with T.
    RHS_mins=np.array([V.findminima(T) for T in TRange]) 
    _RHS_mins = RHS_mins[RHS_mins!=None]
    _Ts = TRange[RHS_mins!=None]
    for T,mins in zip(_Ts,_RHS_mins):
        if V.Vtot(mins,T)>V.Vtot(0,T):
            ax[0,0].scatter(T/V.fSIGMA,mins/V.fSIGMA,color='firebrick')
        else:
            ax[0,0].scatter(T/V.fSIGMA,mins/V.fSIGMA,color='blueviolet')
        

    im1 = ax[0,1].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqEtaVals.T)
    cbar = plt.colorbar(im1)
    cbar.set_label(r"$m_{\eta'}^2$",fontsize=14)
    ax[0,1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)

    ax[0,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
        
        
    im2 = ax[1,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqXVals.T)
    cbar = plt.colorbar(im2)
    cbar.set_label(r'$m_X^2$',fontsize=14)
    ax[1,0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        
    ax[1,0].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
        
        
    im3 = ax[1,1].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqPiVals.T)
    cbar = plt.colorbar(im3)
    cbar.set_label(r'$m_\pi^2$',fontsize=14)
    ax[1,1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)

    ax[1,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')

    #Plots location of critical temperature
    tc=V.criticalT(prnt=False)
    if tc is not None:

        ax[0,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[0,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        
        

    fig.suptitle(f"$f_\pi={V.fSIGMA}$")
    plt.show()
        

    

    

##SPLINE FIT FOR Ib

dat = np.genfromtxt(f'IBData.csv', delimiter=',', dtype=float, skip_header=1)
ddat = np.genfromtxt(f'dIBData.csv', delimiter=',', dtype=float, skip_header=1)

_xb, _yb = dat[:,0], dat[:,1]
_dxb, _dyb = ddat[:,0], ddat[:,1]
# Spline fitting, Ib
#_xbmin = -3.72402637 #Set by CT - can decrease but left to be consistent.
_xbmin=-10
_xbmax = 1.41e3

_tckb_positive = interpolate.interp1d(_xb[_xb>=0], _yb[_xb>=0], kind='cubic', fill_value="extrapolate")
_tckb_negative = interpolate.interp1d(_xb[_xb<=0], _yb[_xb<=0], kind='cubic', fill_value="extrapolate")

_dtckb_positive = interpolate.interp1d(_dxb[_xb>0], _dyb[_xb>0], kind='cubic', fill_value="extrapolate")
_dtckb_negative = interpolate.interp1d(_dxb[_dxb<0], _dyb[_dxb<0], kind='cubic', fill_value="extrapolate")


def Ib_spline(X):
    """Ib interpolated from a saved spline. Input is (m/T)^2."""
    X = np.array(X)
    x = X.ravel()
    
    y = [_tckb_positive(xi) if xi>=0 else _tckb_negative(xi) for xi in x]
    y = np.array(y)

    y[x < _xbmin] = _tckb_negative(_xbmin)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)



def dIb_spline(X):
    """Ib interpolated from a saved spline. Input is (m/T)^2."""
    X = np.array(X)
    x = X.ravel()
    #y= [_dtckb(xi) for xi in x if xi!=0]
    y = [_dtckb_positive(xi) if xi>=0 else _dtckb_negative(xi) for xi in x]
    y = np.array(y)

    y[x==0] = 1e10
    y[x < _xbmin] = _dtckb_negative(_xbmin)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)


def Ib(X):
    #Integral solution for IB - cannot handle negative values.
    if not isinstance(X, (int, float)):
        raise ValueError("R2 must be a numeric value.")
    
    integrand = lambda x: (x**2 / np.sqrt(x**2 + X)) * (1 / (np.exp(np.sqrt(x**2 + X)) - 1))
    result, error = quad(integrand, 0, np.inf, limit=100, epsabs=1e-10, epsrel=1e-10)
    return result


def _Jb_exact2(theta):
    eps = 1e-6 
    # Note that this is a function of theta so that you can get negative values
    f = lambda y: y*y*np.log(1-np.exp(-np.sqrt(y*y+theta)))
    if theta >= 0:
        return quad(f, 0, np.inf)[0]
    else:
        f1 = lambda y: y*y*np.log(2*abs(np.sin(np.sqrt(-theta-y*y)/2)))
        return (
            quad(f, abs(theta)**.5+eps, np.inf)[0] +
            quad(f1, 0, abs(theta)**.5-eps)[0]
        )    



if __name__ == "__main__":
    '''Interpolator'''

    R2_test =5
    print(f"IB({R2_test}) ≈ {Ib_spline(R2_test)} (interpolated)")
    print(f"IB({R2_test}) = {Ib(R2_test)} (direct)")

    # Interpolation vs integral solution
    R2_vals = np.linspace(-5, 10, 2000)
    Ib_vals = np.array([Ib(R2) for R2 in R2_vals])

    plt.scatter(R2_vals, Ib_vals, label='Original IB(R2)')
    plt.plot(R2_vals, Ib_spline(R2_vals), '--', label='Interpolated')
    plt.plot(R2_vals, dIb_spline(R2_vals), '--', label='dIB Interpolated')
    plt.plot(R2_vals, Potential2.Jb_spline(R2_vals), '-.', label='Jb CosmoTransitions')
    plt.plot(R2_vals, [_Jb_exact2(val) for val in R2_vals], '-.', label='Jb Exact')
    plt.xlabel('R2')
    plt.ylabel('IB(R2)')
    plt.legend()
    plt.grid(True)
    plt.show()    
    
    '''Test of dressed mass code'''

   	#NORMAL (fixed c = 8.19444444444445E-09)
    N=3; F=6
    m2Sig = 90000.0; m2Eta = 239722.22222222200; m2X = 250000.0; fPI=833.3333333333330
    m2Sig = 90000.0; m2Eta = 131111.11111111100; m2X = 400000.0; fPI = 1000.0 #VERY BROKEN!
    N_Linput = [*Potential2.masses_to_lagrangian(m2Sig,m2Eta,m2X,fPI,N,F,Potential2.get_detPow(N,F,"Normal"))]

    #V = Potential2.Potential(*N_Linput, N, F, Potential2.get_detPow(N,F,"Normal"))
    
    ##VAN DER WOUDE COMPARISON
    m2 = -4209; ls = 16.8; la = 12.9; c = 2369; F=3; N=3
    fPI=88
    V = Potential2.Potential(m2,c,ls,la,3,3,1,Polyakov=False)

    grd,_ = SolveMasses(V, plot=True)
    
    
    Ts = np.linspace(0,V.fSigma()*1.25)
    sigmas = np.linspace(0,V.fSigma()*1.25,num=100)
    X,Y=np.meshgrid(Ts,sigmas)
    
    #Mass functions:
    MSqSig = lambda sig, T: grd['Sig'](T,sig)
    MSqEta = lambda sig, T: grd['Eta'](T,sig)
    MSqX = lambda sig, T: grd['X'](T,sig)
    MSqPi = lambda sig, T: grd['Pi'](T,sig)
    
    MSqSigVals = np.array([[MSqSig(T, sigma)[0][0] for T in Ts] for sigma in sigmas])
    MSqEtaVals = np.array([[MSqEta(T, sigma)[0][0] for T in Ts] for sigma in sigmas])
    MSqXVals = np.array([[MSqX(T, sigma)[0][0] for T in Ts] for sigma in sigmas])
    MSqPiVals = np.array([[MSqPi(T, sigma)[0][0] for T in Ts] for sigma in sigmas])

    #NOTE X and Y ARE SWAPPED BY CONTOURF!!

    fig, ax = plt.subplots(2,2)
    plt.rcParams['figure.figsize'] = [12, 8]
    
    im0 = ax[0,0].pcolormesh(X/fPI, Y/fPI, MSqSigVals.T)
    cbar = plt.colorbar(im0)
    cbar.set_label(r'Sigma Effective Mass Squared $[MeV^2]$',fontsize=14)
    ax[0,0].set_ylabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,0].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
    
    im1 = ax[0,1].contourf(X/fPI, Y/fPI, MSqEtaVals.T)
    cbar = plt.colorbar(im1)
    cbar.set_label(r'Eta Effective Mass Squared $[MeV^2]$',fontsize=14)
    ax[0,1].set_ylabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,1].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
    
    im2 = ax[1,0].contourf(X/fPI, Y/fPI, MSqXVals.T)
    cbar = plt.colorbar(im2)
    cbar.set_label(r'X Effective Mass Squared $[MeV^2]$',fontsize=14)
    ax[1,0].set_ylabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,0].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
    
    im3 = ax[1,1].contourf(X/fPI, Y/fPI, MSqPiVals.T)
    cbar = plt.colorbar(im3)
    cbar.set_label(r'Pi Effective Mass Squared $[MeV^2]$',fontsize=14)
    ax[1,1].set_ylabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,1].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
    

    fig.suptitle(f"$f_\pi={fPI}$")
    plt.show()
    

    

    Ts = np.linspace(0, fPis[0]*1.1, 100)
    sigmas = np.linspace(0, fPis[0]*1.1, 100)
    #MSqSigVals = np.array([MSqSig(fPis[0], T)[0] for T in Ts])
    #MSqEtaVals = np.array([MSqEta(fPis[0], T)[0] for T in Ts])
    #MSqXVals = np.array([MSqX(fPis[0], T)[0] for T in Ts])
    #MSqPiVals = np.array([MSqPi(fPis[0], T)[0] for T in Ts])
    
    MSqSigVals = np.array([MSqSig(sigma, fPis[0]/4)[0] for sigma in sigmas])
    MSqEtaVals = np.array([MSqEta(sigma, fPis[0]/4)[0] for sigma in sigmas])
    MSqXVals = np.array([MSqX(sigma, fPis[0]/4)[0] for sigma in Ts])
    MSqPiVals = np.array([MSqPi(sigma, fPis[0]/4)[0] for sigma in Ts])
    
    plt.figure(figsize=(10, 6))

    plt.plot(Ts, MSqSigVals, 'gold', linestyle= '-', linewidth=2.5, alpha=0.4 ,label='$MI_\\sigma^2$' )
    #plt.plot(Ts, np.sqrt(M_sigma2_list), 'indigo', linestyle= 'dotted', linewidth=1.5, alpha=1. ,label='$M_\\sigma^2$' )

    plt.plot(Ts, MSqXVals,  'red', linestyle= '-', linewidth=2.5, alpha=0.4, label='$MI_a^2$')
    #plt.plot(Ts, np.sqrt(M_a2_list),  'black', linestyle= 'dotted', linewidth=1.5, alpha=1., label='$M_a^2$')

    plt.plot(Ts, MSqEtaVals,  'yellow', linestyle= '-', linewidth=2.5, alpha=0.4,label='$MI_\\eta^2$')
    #plt.plot(Ts, np.sqrt(M_eta2_list),  'teal', linestyle= 'dotted', linewidth=1.5, alpha=1.,label='$M_\\eta^2$')

    plt.plot(Ts, MSqPiVals, 'orange', linestyle= '-', linewidth=2.5, alpha=0.4,label='$MI_\\pi^2$')
    #plt.plot(Ts, np.sqrt(M_pi2_list), 'deeppink', linestyle= 'dotted', linewidth=1.5, alpha=1.,label='$M_\\pi^2$')


    plt.xlabel('Temperature $T$ [MeV]')
    plt.ylabel('Thermal Masses [MeV]')
    plt.title(f'Thermal Masses vs Temperature $f_\pi={fPis[0]}$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

    