import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"]= 11



def plotVersusParameters(reslN, resN):
    #Eta prime mass squared/fPI squared.
    xN = resN[:,1]/resN[:,3]**2
    xlN = reslN[:,1]/reslN[:,3]**2
    
    #PLOT: 
    wishlist = [4,6,7]
    wishlistLabel = [r'$m^2$', r'$\lambda_\sigma$',r'$\lambda_a$']

    for _i,i in enumerate(wishlist):
        colormap = plt.cm.viridis 
        
        #1: contourf of the non-zero points.
        cmapMaxN = max(resN[resN[:,-1]==0,i]); cmapMaxlN = max(reslN[reslN[:,-1]==0,i])
        cmapMinN = min(resN[resN[:,-1]==0,i]); cmapMinlN = min(reslN[reslN[:,-1]==0,i])
        
        normalizeN = matplotlib.colors.Normalize(vmin=cmapMinN, vmax=cmapMaxN)
        normalizelN = matplotlib.colors.Normalize(vmin=cmapMinlN, vmax=cmapMaxlN)
        
        #Prep data for contourf plot.
        yN = resN[:,i]
        ylN = reslN[:,i]
        
        ZN = []; _xN = []; _yN = []; ZlN = []; _xlN = []; _ylN = []
        for icol in range(len(resN[:,0])):
            if resN[icol,8]> 0.001 and resN[icol,9]>0.001:
                ZN.append(1-resN[icol,9]/resN[icol,8]) #Degree of supercooling.
                _xN.append(xN[icol]);_yN.append(yN[icol])
  
                
        for icol in range(len(reslN[:,0])):
            if reslN[icol,8]> 0.001 and reslN[icol,9]>0.001:
                ZlN.append(1-reslN[icol,9]/reslN[icol,8]) #Degree of supercooling for large N.
                _xlN.append(xlN[icol]);_ylN.append(ylN[icol])
       
        
        XN,YN = np.meshgrid(xN,yN); pointsN = np.column_stack((_xN,_yN))
        XlN, YlN = np.meshgrid(xlN,ylN); pointslN = np.column_stack((_xlN, _ylN))
        
        _ZN = interpolate.griddata(pointsN,ZN,(XN,YN),method='linear')
        _ZlN = interpolate.griddata(pointslN,ZlN,(XlN,YlN),method='linear')
        fig,ax= plt.subplots(nrows=1,ncols=2)
        
        im0 = ax[0].pcolormesh(XN,YN, _ZN.T)
        cbar = plt.colorbar(im0)
        cbar.set_label(r'Degree of Supercooling $1-T_n/Tc$',fontsize=14)
        ax[0].set_ylabel(f'{wishlistLabel[_i]}',fontsize=15)
        ax[0].set_xlabel(r'$m_\eta^2/f_\pi^2$',fontsize=15)
        
        
        im1 = ax[1].pcolormesh(XlN,YlN,_ZlN.T)
        cbar = plt.colorbar(im1)
        cbar.set_label(r'Degree of Supercooling $1-T_n/Tc$',fontsize=14)
        ax[1].set_ylabel(f'{wishlistLabel[_i]}',fontsize=15)
        ax[1].set_xlabel(r'$m_\eta^2/f_\pi^2$',fontsize=15)
        
        #2: Superimposing where the errors are.
        ax[0].scatter(xN[resN[:,-1]==11],yN[resN[:,-1]==11],color='blue',label="IR Divergence")
        ax[0].scatter(xN[resN[:,-1]==16],yN[resN[:,-1]==16],color='orange',label="Non-Converging Dressed Masses")
        ax[0].scatter(xN[resN[:,-1]==17],yN[resN[:,-1]==17],color='green',label="Noisy Data Spoiling S3/T")
        ax[0].scatter(xN[resN[:,-1]==(3 or 5)],yN[resN[:,-1]==(3 or 5)],color='red',label="Well-Behaved but no PT")
        
        ax[1].scatter(xlN[reslN[:,-1]==11],ylN[reslN[:,-1]==11],color='blue')
        ax[1].scatter(xlN[reslN[:,-1]==16],ylN[reslN[:,-1]==16],color='orange')
        ax[1].scatter(xlN[reslN[:,-1]==17],ylN[reslN[:,-1]==17],color='green')
        ax[1].scatter(xlN[reslN[:,-1]==(3 or 5)],ylN[reslN[:,-1]==(3 or 5)],color='red')
        
        ax[0].legend()
        plt.show()
        

def LOWDATAplotVersusParameters(reslN, resN):
    #Eta prime mass squared/fPI squared.
    xN = resN[:,1]/resN[:,3]**2
    xlN = reslN[:,1]/reslN[:,3]**2
    
    #PLOT: 
    wishlist = [0,2,4,6,7]
    wishlistLabel = [r'$m^2_\sigma$',r'$m^2_X$',r'$m^2$', r'$\lambda_\sigma$',r'$\lambda_a$']

    for _i,i in enumerate(wishlist):
        colormap = plt.cm.viridis 
        
        #1: contourf of the non-zero points.
        cmapMaxN = max(resN[resN[:,-1]==0,i]); cmapMaxlN = max(reslN[reslN[:,-1]==0,i])
        cmapMinN = min(resN[resN[:,-1]==0,i]); cmapMinlN = min(reslN[reslN[:,-1]==0,i])
        
        normalizeN = matplotlib.colors.Normalize(vmin=cmapMinN, vmax=cmapMaxN)
        normalizelN = matplotlib.colors.Normalize(vmin=cmapMinlN, vmax=cmapMaxlN)
        
        #Prep data for contourf plot.
        yN = resN[:,i]
        ylN = reslN[:,i]
        
        ZN = []; _xN = []; _yN = []; ZlN = []; _xlN = []; _ylN = []
        for icol in range(len(resN[:,0])):
            if resN[icol,8]> 0.001 and resN[icol,9]>0.001:
                ZN.append(1-resN[icol,9]/resN[icol,8]) #Degree of supercooling.
                _xN.append(xN[icol]);_yN.append(yN[icol])
  
                
        for icol in range(len(reslN[:,0])):
            if reslN[icol,8]> 0.001 and reslN[icol,9]>0.001:
                ZlN.append(1-reslN[icol,9]/reslN[icol,8]) #Degree of supercooling for large N.
                _xlN.append(xlN[icol]);_ylN.append(ylN[icol])
       
        
        fig,ax= plt.subplots(nrows=1,ncols=2)
        
        im0 = ax[0].scatter(_xN,_yN,c=ZN)
        cbar = plt.colorbar(im0)
        cbar.set_label(r'Degree of Supercooling $1-T_n/Tc$',fontsize=14)
        ax[0].set_ylabel(f'{wishlistLabel[_i]}',fontsize=15)
        ax[0].set_xlabel(r'$m_\eta^2/f_\pi^2$',fontsize=15)
        
        
        im1 = ax[1].scatter(_xlN,_ylN,c=ZlN)
        cbar = plt.colorbar(im1)
        cbar.set_label(r'Degree of Supercooling $1-T_n/Tc$',fontsize=14)
        ax[1].set_ylabel(f'{wishlistLabel[_i]}',fontsize=15)
        ax[1].set_xlabel(r'$m_\eta^2/f_\pi^2$',fontsize=15)
        
        #2: Superimposing where the errors are.
        ax[0].scatter(xN[resN[:,-1]==11],yN[resN[:,-1]==11],color='black',label="IR Divergence")
        ax[0].scatter(xN[resN[:,-1]==16],yN[resN[:,-1]==16],color='grey',label="Non-Converging Dressed Masses")
        ax[0].scatter(xN[resN[:,-1]==17],yN[resN[:,-1]==17],color='darkred',label="Noisy Data Spoiling S3/T")
        ax[0].scatter(xN[resN[:,-1]==(3 or 5)],yN[resN[:,-1]==(3 or 5)],color='red',label="Well-Behaved but no PT")
        
        ax[1].scatter(xlN[reslN[:,-1]==11],ylN[reslN[:,-1]==11],color='black')
        ax[1].scatter(xlN[reslN[:,-1]==16],ylN[reslN[:,-1]==16],color='grey')
        ax[1].scatter(xlN[reslN[:,-1]==17],ylN[reslN[:,-1]==17],color='darkred')
        ax[1].scatter(xlN[reslN[:,-1]==(3 or 5)],ylN[reslN[:,-1]==(3 or 5)],color='red')
        
        #ax[0].legend()
        plt.tight_layout()
        plt.show()
        
#Load up the data.
resN = np.genfromtxt(f'Test_N3F6_Normal.csv', delimiter=',', dtype=float, skip_header=1)
reslN = np.genfromtxt(f'Test_N3F6_largeN.csv', delimiter=',', dtype=float, skip_header=2)

LOWDATAplotVersusParameters(reslN,resN)