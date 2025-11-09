import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from scipy import optimize, differentiate
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy import interpolate
import csv
import Potential2
import cosmoTransitions
from IBInterpolation import IB
from scipy.ndimage import gaussian_filter
from debug_plot import debug_plot


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"]= 12

NUMBEROFPOINTS = 150
EPSILON = 0.1



def SolveMasses_adaptive(V, coarse_points=50, fine_points=150):
    """
      adaptive scan:
      1) Quick coarse scan over all T.
      2) If no phase transition → skip refinement.
      3) If phase transition exists → re-run SolveMasses only around Tc.
    """
    global NUMBEROFPOINTS
    old_POINTS = NUMBEROFPOINTS      
    old_TMULT = Potential2.TMULT      # store original T max multiplier
    
    
    # 1) Coarse scan (fast)
    print(f"[Adaptive] Coarse scan with {coarse_points} points...")
    NUMBEROFPOINTS = coarse_points
    Potential2.TMULT = old_TMULT      # keep same T max (just fewer points)
   
    dressed, RMS, _ = SolveMasses(V, plot=False)
    tc = V.criticalT(plot=False)

    # 2) If no phase transition → done
    if tc is None:
        print("[Adaptive] No Tc found. Skipping refinement.")
        NUMBEROFPOINTS = old_POINTS
        Potential2.TMULT = old_TMULT
        return dressed, RMS, None

    Tmax = tc * 1.2
    Potential2.TMULT = Tmax / V.fSIGMA
    
    

    # Restore fine resolution
    NUMBEROFPOINTS = fine_points
    Potential2.TMULT = old_TMULT
    return result


def SolveMasses(V, plot=False):
    plot=Potential2.PLOT_RUN
    
    #Distinct Feynman rule structures.
    c1 = (V.c/V.F**2)*V.fSIGMA**(V.F*V.detPow-4)*(V.F*V.detPow)*(V.F*V.detPow-1)*(V.F*V.detPow-2)*(V.F*V.detPow-3)
    c2 = (V.c/V.F)*V.fSIGMA**(V.F*V.detPow-4)*V.detPow*(V.F*V.detPow-2)*(V.F*V.detPow-3)
    c3 = (V.c/V.F)*V.fSIGMA**(V.F*V.detPow-4)*V.detPow*(V.detPow*V.F**3-4*V.F**2+V.detPow*V.F+6)
    ctilde = (V.F**2-1)


    #Setting up the scan.
    TRange = np.linspace(0. ,V.fSIGMA*Potential2.TMULT,num=NUMBEROFPOINTS)
    sigmaRange = np.linspace(EPSILON, V.fSIGMA*Potential2.SIGMULT,num=NUMBEROFPOINTS)
    
    MSqSigData = np.zeros((len(TRange),len(sigmaRange)))
    MSqEtaData = np.zeros((len(TRange),len(sigmaRange)))
    MSqPiData = np.zeros((len(TRange),len(sigmaRange)))
    MSqXData = np.zeros((len(TRange),len(sigmaRange)))
    RMS = np.zeros((len(TRange),len(sigmaRange))); failPoints = []
    
    # Store previous solutions at each grid (T, σ) to enable 2D warm starts 
    solution_grid = np.full((len(TRange), len(sigmaRange), 4), np.nan)
    
    t_total = time.time()

    for i,T in enumerate(TRange):
        t_row = time.time()
        prev_solution = None # stores last successful solution (Mσ², Mη², MX², Mπ²)
        for j,sigma in enumerate(sigmaRange):
            if T<EPSILON:
                MSqSigData[i,j] = V.mSq['Sig'][0](sigma)
                MSqEtaData[i,j] = V.mSq['Eta'][0](sigma)
                MSqPiData[i,j] = V.mSq['Pi'][0](sigma)
                MSqXData[i,j] = V.mSq['X'][0](sigma)
                RMS[i,j] = 0
                
                continue
            
            def bagEquations(vars):                

                M_sigma2, M_eta2, M_X2, M_Pi2 = vars

                prefactor = T**2 / (4 * np.pi**2)

                lhs = np.array([M_sigma2, M_eta2, M_X2, M_Pi2])

                rhs = np.array([
                    #Sigma Thermal Dressed Mass.
                    V.mSq['Sig'][0](sigma) + prefactor * (
                        (3 * V.lambdas - c1) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2 - 1) * (V.lambdas + 2 * V.lambdaa) + c2*ctilde) * Ib_spline(M_X2 / T**2)
                        + (V.lambdas + c1) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2 - 1) * V.lambdas - c2*ctilde) * Ib_spline(M_Pi2 / T**2)),

                    #Eta Prime Thermal Dressed Mass.
                    V.mSq['Eta'][0](sigma) + prefactor * (
                        (3 * V.lambdas - c1) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2 - 1) * (V.lambdas + 2 * V.lambdaa) + c2*ctilde) * Ib_spline(M_Pi2 / T**2)
                        + (V.lambdas + c1) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2 - 1) * V.lambdas - c2*ctilde) * Ib_spline(M_X2 / T**2)),
                        
                    #X Thermal Dressed Mass.
                    V.mSq['X'][0](sigma) + prefactor * (
                        (V.lambdas + 2 * V.lambdaa + c2) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2+1)*V.lambdas + (V.F**2-4)*V.lambdaa - c3) * Ib_spline(M_X2 / T**2)
                        + (V.lambdas - c2) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * Ib_spline(M_Pi2 / T**2)),

                    #Pi Thermal Dressed Mass.
                    V.mSq['Pi'][0](sigma) + prefactor * (
                        (V.lambdas + 2 * V.lambdaa + c2) * Ib_spline(M_eta2 / T**2)
                        + ((V.F**2+1)*V.lambdas + (V.F**2-4)*V.lambdaa - c3) * Ib_spline(M_Pi2 / T**2)
                        + (V.lambdas - c2) * Ib_spline(M_sigma2 / T**2)
                        + ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * Ib_spline(M_X2 / T**2))
                ])

                bagEquations.lhs = lhs
                bagEquations.rhs = rhs
                
                #residuals = LHS - RHS
                return lhs - rhs 
            

            #Also useful to define the jacobian to assist the scipy root function.
            def jac(vars):
                if T<EPSILON:
                    return np.array([[1.,0.,0.,0.],
                                    [0.,1.,0.,0.],
                                    [0.,0.,1.,0.],
                                    [0.,0.,0.,1.]])
                
                M_sigma2, M_eta2, M_X2, M_Pi2 = vars
                prefactor = 1. / (4 * np.pi**2)
                
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
                    ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * dIb_spline(M_Pi2 / T**2)],#VdW SAYS F^2-1 SANNINO F^2+1!!
                    #dM2Pi
                    [(V.lambdas - c2) * dIb_spline(M_sigma2 / T**2),
                    (V.lambdas + 2 * V.lambdaa + c2) * dIb_spline(M_eta2 / T**2),
                    ((V.F**2-1)*V.lambdas + V.F**2 * V.lambdaa + c3) * dIb_spline(M_X2 / T**2),
                    ((V.F**2+1)*V.lambdas + (V.F**2-4)*V.lambdaa - c3) * dIb_spline(M_Pi2 / T**2) - 1/prefactor]#VdW SAYS F^2-1 SANNINO F^2+1!!
                    ])

                return res
                     
            if j > 0 and np.all(np.isfinite(solution_grid[i, j-1])):
                initial_guess = solution_grid[i, j-1]            # left neighbor
            elif i > 0 and np.all(np.isfinite(solution_grid[i-1, j])):
                initial_guess = solution_grid[i-1, j]            # above neighbor
            elif i > 0 and j > 0 and np.all(np.isfinite(solution_grid[i-1, j-1])):
                initial_guess = solution_grid[i-1, j-1]          # diagonal neighbor
                
            elif prev_solution is not None:
                initial_guess = prev_solution  # same row fallback

            else:
                # keep your original Debye-based guess
                initial_guess = [
                    V.mSq['Sig'][0](sigma) + (T**2/24)*(3*V.lambdas - c1 * sigma**(V.F*V.detPow-4)),
                    V.mSq['Eta'][0](sigma) + (T**2/24)*(V.lambdas + c1 * sigma**(V.F*V.detPow-4)),
                    V.mSq['X'][0](sigma)  + (T**2/24)*(V.lambdas + 2*V.lambdaa + c2 * sigma**(V.F*V.detPow-4)),
                    V.mSq['Pi'][0](sigma) + (T**2/24)*(V.lambdas - c2 * sigma**(V.F*V.detPow-4)),
                ]

            
            #Scipy root function to solve the coupled equations.
            sol = root(bagEquations, initial_guess, jac=jac, method='hybr',tol=1.49012e-08)
            #Root mean squared error with a cutoff at 1.
            RMS[i,j]=min(np.sqrt(np.mean((bagEquations.lhs-bagEquations.rhs)**2)),1)

            if sol.success and RMS[i,j]<1/V.fSIGMA: #old(np.sqrt(V.fSIGMA))
                M_sigma2, M_eta2, M_X2, M_Pi2 = sol.x

                MSqSigData[i,j]=M_sigma2
                MSqEtaData[i,j]=M_eta2
                MSqXData[i,j]=M_X2
                MSqPiData[i,j]=M_Pi2
                
                # SAVE solution for future neighbor-based warm-start
                solution_grid[i, j] = sol.x
                prev_solution = sol.x

                

            else:
                #Try with numerical jacobian as well:
                sol = root(bagEquations, initial_guess, method='hybr')
                
                if sol.success and RMS[i,j]<1/V.fSIGMA:
                    M_sigma2, M_eta2, M_X2, M_Pi2 = sol.x

                    MSqSigData[i,j]=M_sigma2
                    MSqEtaData[i,j]=M_eta2
                    MSqXData[i,j]=M_X2
                    MSqPiData[i,j]=M_Pi2
                else:
                        
                    MSqSigData[i,j]=None
                    MSqEtaData[i,j]=None
                    MSqXData[i,j]=None
                    MSqPiData[i,j]=None
                    
                    failPoints.append([sigma, T])
                    
        #print(f"T-row {i}/{len(TRange)} took {time.time() - t_row:.2f} s")       
        print(f"T-row {i}/{len(TRange)} took {time.time() - t_row:.2f} s", flush=True)
     

    X,Y=np.meshgrid(TRange,sigmaRange) 
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    valuesSigma = MSqSigData.ravel()
    valuesEta = MSqEtaData.ravel()
    valuesX = MSqXData.ravel()
    valuesPi = MSqPiData.ravel()
    
    #First interpolator (bad) getting the data into the right shape.
    _MSqSigData = interpolate.griddata(points[np.isfinite(valuesSigma)],valuesSigma[np.isfinite(valuesSigma)],(X,Y))
    _MSqEtaData = interpolate.griddata(points[np.isfinite(valuesEta)],valuesEta[np.isfinite(valuesEta)],(X,Y))
    _MSqXData = interpolate.griddata(points[np.isfinite(valuesX)],valuesX[np.isfinite(valuesX)],(X,Y))
    _MSqPiData = interpolate.griddata(points[np.isfinite(valuesPi)],valuesPi[np.isfinite(valuesPi)],(X,Y))
    
    _MSqSigData=np.array(_MSqSigData)
    _MSqEtaData=np.array(_MSqEtaData)
    _MSqXData=np.array(_MSqXData)
    _MSqPiData=np.array(_MSqPiData)
    
    #Sometimes there are issues with the bounding box!
    
    if np.isnan(_MSqSigData).any():
        #First try: patch with real data.
        _broken = np.isnan(_MSqSigData)
        
        _MSqSigData[_broken] = MSqSigData[_broken]
        _MSqEtaData[_broken] = MSqEtaData[_broken]
        _MSqXData[_broken] = MSqXData[_broken]
        _MSqPiData[_broken] = MSqPiData[_broken]
        
        #Second try: patch with nearest data
        if np.isnan(_MSqSigData).any():
            _broken =  np.isnan(_MSqSigData)
            
            _MSqSigData[_broken] = np.array(interpolate.griddata(points[np.isfinite(valuesSigma)],valuesSigma[np.isfinite(valuesSigma)],(X[_broken],Y[_broken]),method='nearest'))
            _MSqEtaData[_broken] = np.array(interpolate.griddata(points[np.isfinite(valuesEta)],valuesEta[np.isfinite(valuesEta)],(X[_broken],Y[_broken]),method='nearest'))
            _MSqXData[_broken] = np.array(interpolate.griddata(points[np.isfinite(valuesX)],valuesX[np.isfinite(valuesX)],(X[_broken],Y[_broken]),method='nearest'))
            _MSqPiData[_broken] = np.array(interpolate.griddata(points[np.isfinite(valuesPi)],valuesPi[np.isfinite(valuesPi)],(X[_broken],Y[_broken]),method='nearest'))
    
    if plot:
        plotMassData([MSqSigData,MSqEtaData,MSqXData,MSqPiData], V,minimal=True)
        plotMassData([_MSqSigData,_MSqEtaData,_MSqXData,_MSqPiData], V,minimal=True)
        
    #Very accurate interpolator to the data (which may be noisy itself so beware).
    rectiSig = interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqSigData/V.fSIGMA, ky=2,kx=2)
    rectiEta = interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqEtaData/V.fSIGMA, ky=2,kx=2)
    rectiX = interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqXData/V.fSIGMA, ky=2,kx=2)
    rectiPi = interpolate.RectBivariateSpline(TRange, sigmaRange, _MSqPiData/V.fSIGMA, ky=2,kx=2)
    
    #FIRST TRY WITH AN EXACT SPLINE FIT.
    dressedMasses = {
        #Sigma Mass
        'Sig' : rectiSig,
        #Eta Prime Mass
        'Eta' : rectiEta,
        #X Mass
        'X' : rectiX,
        #Pi Mass
        'Pi' : rectiPi
        }
    V.setMSq(dressedMasses)

    #CHECK IF RUN IS NOISY OR NOT.
    tc = V.criticalT(prnt=plot)

    #Checking for the convergence of dressedMasses around the critical temperature.
    counter = 0; noisyPoint=False
    if tc is None: noisyPoint = True #Quick initial check.
    if not noisyPoint:
        for sig,T in failPoints:
            if T<tc and abs((tc-T)/tc)<0.25:
                counter += 1
                
            if counter > 5: #5 is arbitrary right now! Adjust as sensible.
                print("Dressed Masses not converging properly around phase transition region on first pass.")
                noisyPoint=True
                V.tc = None #Old critical temperature is invalid. Needs setting again.
                break


    if noisyPoint:
        failPoints=np.array(failPoints)
        #Histogram of failed points with temperature.
        hist = np.zeros(TRange.shape);starti = 0
        for iT,T in enumerate(TRange):
            if starti==0 and T>np.sqrt(V.fSIGMA): starti = iT #Starting index for histogram
            hist[iT] = len([1 for sig,Tf in failPoints if abs(Tf-T)<1e-4])

        #Now we have the histogram...
        exit = False; i=starti
        while not exit:
            if hist[i]==0 and np.sum(hist[:i])/len(failPoints)>0.33:  #Checks to see if we've cleared 33% of failed points.
                minT = TRange[i]
                exit = True
            
            elif hist[i]==0 and hist[i-1]>hist[i] and np.sum(hist[:i])/len(failPoints)>0.33:
                minT = TRange[i]
                exit = True
            i+=1
            if i>=NUMBEROFPOINTS:
                raise Potential2.BadDressedMassConvergence("Dressed Masses not converging properly around phase transition region.")
            

        #And now a minimum T which it makes sense to talk about anything relating to the PT.
        print(f'minimum T = {minT}')
        V.minT = minT

        rectiSig = interpolate.RectBivariateSpline(TRange[i:], sigmaRange, _MSqSigData[i:,:]/V.fSIGMA, ky=2,kx=2)
        rectiEta = interpolate.RectBivariateSpline(TRange[i:], sigmaRange, _MSqEtaData[i:,:]/V.fSIGMA, ky=2,kx=2)
        rectiX = interpolate.RectBivariateSpline(TRange[i:], sigmaRange, _MSqXData[i:,:]/V.fSIGMA, ky=2,kx=2)
        rectiPi = interpolate.RectBivariateSpline(TRange[i:], sigmaRange, _MSqPiData[i:,:]/V.fSIGMA, ky=2,kx=2)

        dressedMasses = {
        #Sigma Mass
        'Sig' : rectiSig,
        #Eta Prime Mass
        'Eta' : rectiEta,
        #X Mass
        'X' : rectiX,
        #Pi Mass
        'Pi' : rectiPi
        }
        V.setMSq(dressedMasses)
    

    if plot:
        tc = V.criticalT()
        #Plot 1: Individual plots of data vs interpolated.       
        fig, ax = plt.subplots(2,2)
        plt.rcParams['figure.figsize'] = [12, 8]
        
        TIndexSample = [0, 50, 100, 150, 165, 180, 195]
        colours = ["red", "firebrick", "darkorange", "crimson", "rosybrown", "gold", "palevioletred"]
        
        for i,Tindex in enumerate(TIndexSample):
            T = TRange[Tindex]
            ax[0,0].scatter(sigmaRange, MSqSigData[Tindex,:]/V.fSIGMA**2, color=colours[i], label=f"T={T}",alpha=0.66)
            ax[0,1].scatter(sigmaRange, MSqEtaData[Tindex,:]/V.fSIGMA**2, color=colours[i],alpha=0.66)
            ax[1,0].scatter(sigmaRange, MSqXData[Tindex,:]/V.fSIGMA**2, color=colours[i],alpha=0.66)
            ax[1,1].scatter(sigmaRange, MSqPiData[Tindex,:]/V.fSIGMA**2, color=colours[i],alpha=0.66)
            
            ax[0,0].scatter(sigmaRange, _MSqSigData[Tindex,:]/V.fSIGMA**2, color=colours[i], label=f"T={T}",alpha=0.66,marker='1')
            ax[0,1].scatter(sigmaRange, _MSqEtaData[Tindex,:]/V.fSIGMA**2, color=colours[i],alpha=0.66,marker='1')
            ax[1,0].scatter(sigmaRange, _MSqXData[Tindex,:]/V.fSIGMA**2, color=colours[i],alpha=0.66,marker='1')
            ax[1,1].scatter(sigmaRange, _MSqPiData[Tindex,:]/V.fSIGMA**2, color=colours[i],alpha=0.66,marker='1')

            
            ax[0,0].plot(sigmaRange, dressedMasses['Sig'](T, sigmaRange).flatten()/V.fSIGMA, color=colours[i])
            ax[0,1].plot(sigmaRange, dressedMasses['Eta'](T, sigmaRange).flatten()/V.fSIGMA, color=colours[i])
            ax[1,0].plot(sigmaRange, dressedMasses['X'](T, sigmaRange).flatten()/V.fSIGMA, color=colours[i])
            ax[1,1].plot(sigmaRange, dressedMasses['Pi'](T, sigmaRange).flatten()/V.fSIGMA, color=colours[i])
        
                

        ax[0,0].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
        ax[0,0].set_ylabel(r'$m_\sigma^2$',fontsize=15)
        
        ax[0,1].set_xlabel(r'$\sigma/f_\pi$',fontsize=15)
        ax[0,1].set_ylabel(r'$m_\eta^2$',fontsize=15)

        fig.legend()
        
        #Plot 2: Data grids for effective masses.
        if noisyPoint: 
            plotMassData([MSqSigData,MSqEtaData,MSqXData,MSqPiData], V, minT=minT)
        else:
            plotMassData([MSqSigData,MSqEtaData,MSqXData,MSqPiData], V)
        
        if tc is not None:
            for sig,T in failPoints:#Flag up failure points below 25% of the critical temperature.
                if T<tc and abs((tc-T)/tc)<0.25:
                    plt.scatter(T/V.fSIGMA,sig/V.fSIGMA,marker='d',color='orange')
        plt.savefig(f"Temporal-Plots/Vcritical.pdf", dpi=300)    
        debug_plot(name="debug", overwrite=False)        
        

        #Plot 3: RMS error and perturbativity.

        fig, ax = plt.subplots(nrows=1,ncols=2)
        
        im0 = ax[0].contourf(X/V.fSIGMA, Y/V.fSIGMA, RMS.T)
        cbar0 = plt.colorbar(im0)
        cbar0.set_label(r'RMS $[MeV^2]$',fontsize=14)
        ax[0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
        ax[0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        
        #IR Problem:
	#4-point effective vertices.
        gSig_eff = np.abs(3*V.lambdas - V.c*V.fSIGMA**(V.F*V.detPow-4)*(V.F*V.detPow)*(V.F*V.detPow-1)*(V.F*V.detPow-2)*(V.F*V.detPow-3)/V.F**2)/(24.)
        
        gPi_eff = np.abs(V.lambdas*(V.F**2+1) + V.lambdaa*(V.F**2-4)
                    - V.c*V.fSIGMA**(V.F*V.detPow-4)*(V.detPow/V.F)*(V.detPow*V.F**3-4*V.F**2+V.detPow*V.F+6))/((V.F**2-1) * (2**2))
        
        pSig = lambda sig,T: gSig_eff * (T/(np.abs(V.MSq['Sig'][0](sig,T))+1e-12)**(1/2))
        pPi = lambda sig,T: gPi_eff * (V.F**2-1) * (T/(np.abs(V.MSq['Pi'][0](sig,T))+1e-12)**(1/2))

        perturbativity = pSig(Y,X) + pPi(Y,X)
        perturbativity[pSig(Y,X)>16*np.pi]=16*np.pi
        perturbativity[pPi(Y,X)>16*np.pi]=16*np.pi
        
        im1 = ax[1].contourf(X/V.fSIGMA, Y/V.fSIGMA, perturbativity)
        cbar1 = plt.colorbar(im1)
        cbar1.set_label(r'Effective Coupling $g_eff$',fontsize=14)
        ax[1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
        ax[1].set_ylabel(r'$sigma/f_\pi$',fontsize=15)


        
        
        #Plots location of second minimum with T.
        RHS_mins=np.array([V.findminima(T) for T in TRange]) 
        _RHS_mins = RHS_mins[RHS_mins!=None]
        _Ts = TRange[RHS_mins!=None]
        for T,mins in zip(_Ts,_RHS_mins):
            if V.Vtot(mins,T)>V.Vtot(0,T):
                ax[1].scatter(T/V.fSIGMA,mins/V.fSIGMA,color='firebrick')
            else:
                ax[1].scatter(T/V.fSIGMA,mins/V.fSIGMA,color='blueviolet')
                
        fig.suptitle(r"$f_\pi={V.fSIGMA}$")
        plt.savefig("Temporal-Plots/secondminimum.pdf",dpi=300)
        debug_plot(name="debug", overwrite=False)
        #plt.show()
        
    if noisyPoint:
        return dressedMasses, RMS, minT
    else:
        return dressedMasses, RMS, None
    

def plotMassData(massData, V, minT=None, minimal=False):
    #Make sure these are exactly the same ranges as above!
    TRange = np.linspace(0,V.fSIGMA*Potential2.TMULT,num=NUMBEROFPOINTS)[::-1]
    sigmaRange = np.linspace(0.01, V.fSIGMA*Potential2.SIGMULT,num=NUMBEROFPOINTS)
    
    MSqSigData=massData[0]
    MSqEtaData=massData[1]
    MSqXData=massData[2]
    MSqPiData=massData[3]

    if not minimal:
        tc=V.criticalT(prnt=False,minT=minT)
    
    X,Y=np.meshgrid(TRange,sigmaRange)
        
    fig, ax = plt.subplots(2,2)
    plt.rcParams['figure.figsize'] = [12, 8]
        
    im0 = ax[0,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqSigData.T)
    cbar = plt.colorbar(im0)
    cbar.set_label(r'$m_\sigma^2$',fontsize=14)
    ax[0,0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        
    if not minimal:
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
    
    if not minimal:
        ax[0,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
    

    ## NOTE DEBUG FOR NAN VALUES!
    ax[0,1].scatter(X[np.isnan(MSqEtaData.T)]/V.fSIGMA,Y[np.isnan(MSqEtaData.T)]/V.fSIGMA,color='orange')
        
    im2 = ax[1,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqXData.T)
    cbar = plt.colorbar(im2)
    cbar.set_label(r'$m_X^2$',fontsize=14)
    ax[1,0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
    
    if not minimal:
        ax[1,0].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
        
        
    im3 = ax[1,1].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqPiData.T)
    cbar = plt.colorbar(im3)
    cbar.set_label(r'$m_\pi^2$',fontsize=14)
    ax[1,1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)

    if not minimal:
        ax[1,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')

    #Plot a vertical line at the critical temperature.
    if not minimal and tc is not None:
        ax[0,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[0,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        
    if not minimal and minT is not None:
        ax[0,0].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        ax[0,1].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        ax[1,0].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        ax[1,1].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        
    fig.suptitle(r"$f_\pi={V.fSIGMA}$")

    debug_plot(name="DressedMassData", overwrite=False)

def plotInterpMasses(V):
    #Make sure these are exactly the same ranges as above!
    TRange = np.linspace(0,V.fSIGMA*Potential2.TMULT,num=NUMBEROFPOINTS)
    sigmaRange = np.linspace(0.01, V.fSIGMA*Potential2.SIGMULT,num=NUMBEROFPOINTS)
    
    massDict = V.MSq
    RMS = V.RMS
    minT = V.minT
    
    #Finds location of critical temperature
    tc= V.criticalT(prnt=True,minT=minT)
    print(f'tc: {tc}')
    
    #Data grids for effective masses
    X,Y=np.meshgrid(TRange,sigmaRange)
    
    #Mass functions:
    MSqSig = lambda sig, T: massDict['Sig'][0](sig,T)
    MSqEta = lambda sig, T: massDict['Eta'][0](sig,T)
    MSqX = lambda sig, T: massDict['X'][0](sig,T)
    MSqPi = lambda sig, T: massDict['Pi'][0](sig,T)
    
    print(f'tc+10:{[MSqSig(sigma,tc+10) for sigma in np.linspace(0,V.fSIGMA*Potential2.SIGMULT,num=10)]}')
    print(f'tc:{[MSqSig(sigma,tc) for sigma in np.linspace(0,V.fSIGMA*Potential2.SIGMULT,num=10)]}')
    print(f'tc-10:{[MSqSig(sigma,tc-10) for sigma in np.linspace(0,V.fSIGMA*Potential2.SIGMULT,num=10)]}')
    
    #Mass data:
    MSqSigVals = np.array([[MSqSig(sigma, T) for T in TRange] for sigma in sigmaRange]) #Taking transpose.
    MSqEtaVals = np.array([[MSqEta(sigma, T) for T in TRange] for sigma in sigmaRange])
    MSqXVals = np.array([[MSqX(sigma, T) for T in TRange] for sigma in sigmaRange])
    MSqPiVals = np.array([[MSqPi(sigma, T) for T in TRange] for sigma in sigmaRange])
        
    fig, ax = plt.subplots(2,2)
    plt.rcParams['figure.figsize'] = [12, 8]
        
    im0 = ax[0,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqSigVals)
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
        
 
    im1 = ax[0,1].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqEtaVals)
    cbar = plt.colorbar(im1)
    cbar.set_label(r"$m_{\eta'}^2$",fontsize=14)
    ax[0,1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0,1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
 
    ax[0,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
        
        
    im2 = ax[1,0].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqXVals)
    cbar = plt.colorbar(im2)
    cbar.set_label(r'$m_X^2$',fontsize=14)
    ax[1,0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        
    ax[1,0].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
        
        
    im3 = ax[1,1].contourf(X/V.fSIGMA, Y/V.fSIGMA, MSqPiVals)
    cbar = plt.colorbar(im3)
    cbar.set_label(r'$m_\pi^2$',fontsize=14)
    ax[1,1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1,1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
 
    ax[1,1].scatter(TRange[RHS_mins!=None]/V.fSIGMA,RHS_mins[RHS_mins!=None]/V.fSIGMA,color='firebrick')
 
    if tc is not None:
 
        ax[0,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[0,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,0].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
        ax[1,1].vlines(tc/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dashed',color='grey',linewidth=3)
    
    if minT is not None:
        ax[0,0].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        ax[0,1].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        ax[1,0].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        ax[1,1].vlines(minT/V.fSIGMA,min(sigmaRange)/V.fSIGMA,max(sigmaRange)/V.fSIGMA,linestyle='dotted',color='grey',linewidth=3)
        
 
    fig.suptitle(f"$f_\pi={V.fSIGMA}$")
    debug_plot(name="debug", overwrite=False)
    #plt.show()
        
 
    #Plot 2: RMS error and perturbativity.
 
    fig, ax = plt.subplots(nrows=1,ncols=2)
        
    im0 = ax[0].contourf(X/V.fSIGMA, Y/V.fSIGMA, RMS.T)
    cbar0 = plt.colorbar(im0)
    cbar0.set_label(r'RMS $[MeV^2]$',fontsize=14)
    ax[0].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[0].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
        
    #IR Problem:
    #4-point effective vertices.
    gSig_eff = np.abs(3*V.lambdas - V.c*V.fSIGMA**(V.F*V.detPow-4)*(V.F*V.detPow)*(V.F*V.detPow-1)*(V.F*V.detPow-2)*(V.F*V.detPow-3)/V.F**2)/(24.)
        
    gPi_eff = np.abs(V.lambdas*(V.F**2+1) + V.lambdaa*(V.F**2-4)
                    - V.c*V.fSIGMA**(V.F*V.detPow-4)*(V.detPow/V.F)*(V.detPow*V.F**3-4*V.F**2+V.detPow*V.F+6))/((V.F**2-1) * (2**3))
 
    pSig = lambda sig,T: gSig_eff * (T/(np.abs(V.MSq['Sig'][0](sig,T))+1e-12)**(1/2))
    pPi = lambda sig,T: gPi_eff * (V.F**2-1) * (T/(np.abs(V.MSq['Pi'][0](sig,T))+1e-12)**(1/2))
 
    perturbativity = pSig(X,Y) + pPi(X,Y)
    perturbativity[pSig(X,Y)>16*np.pi]=16*np.pi
    perturbativity[pPi(X,Y)>16*np.pi]=16*np.pi
        
    im1 = ax[1].contourf(X/V.fSIGMA, Y/V.fSIGMA, perturbativity.T)
    cbar1 = plt.colorbar(im1)
    cbar1.set_label(r'Effective Coupling $g_eff$',fontsize=14)
    ax[1].set_xlabel(r'Temperature $T/f_\pi$',fontsize=15)
    ax[1].set_ylabel(r'$\sigma/f_\pi$',fontsize=15)
    
    
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

    y[x==0] = 1e15
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


#This only runs if we run DressedMasses.py 
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
    plt.savefig(f"Temporal-Plots/R2functions.pdf", dpi=300)
    debug_plot(name="debug", overwrite=False)
    #plt.show()    
    
    '''Test of dressed mass code'''

   	#NORMAL (fixed c = 8.19444444444445E-09)
    N=3; F=3
    m2Sig = 90000.0; m2Eta = 100.; m2X = 1750000.0; fPI=1000.
    #m2Sig = 90000.0; m2Eta = 239722.22222222200; m2X = 250000.0; fPI=833.3333333333330
    #m2Sig = 90000.0; m2Eta = 131111.11111111100; m2X = 400000.0; fPI = 1000.0 #VERY BROKEN!
    N_Linput = [*Potential2.masses_to_lagrangian(m2Sig,m2Eta,m2X,fPI,N,F,Potential2.get_detPow(N,F,"Normal"))]

    V = Potential2.Potential(*N_Linput, N, F, Potential2.get_detPow(N,F,"Normal"), fSIGMA=fPI)
    
    ###VAN DER WOUDE COMPARISON
    #m2 = -4209; ls = 16.8; la = 12.9; c = 2369; F=3; N=3
    #fPI=88
    #V = Potential2.Potential(m2,c,ls,la,3,3,1,Polyakov=False)

    #SolveMasses(V, plot=True)
    SolveMasses_adaptive(V)

    



