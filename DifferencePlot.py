import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate, optimize
import csv
from multiprocessing import Pool
from debug_plot import debug_plot


'''
    Identical function operation to "differencePlot" in Plotter.py. Just in a separate file allowing for more control over plotting parameters.

        "plotDifference" allows you to see, for the Csaki (half-transparency) and normal (full-transparency) cases, 
        the difference in GW data for cases with the same zero-temp masses. These are indicated by having the same marker. 
        However, this function can only handle up to 22 data points or it does not have enough markers.
                
        TLDR CSAKI IS HALF TRANSPARENCY; NORMAL IS FULL TRANSPARENCY. CAN ONLY HANDLE 22 DATA POINTS.
                
        plotDifference:
            INPUTS: (resC, resN)
                (np.array, np.array)

'''


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"]= 11



def plotDifference(reslN, resN):
    #So you can see exactly where the points have moved to
    markers = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","D","d","|"]
    print(reslN[reslN[:,10]!=0,1])
    colormap = plt.cm.viridis #or any other colormap
    normalize = matplotlib.colors.Normalize(vmin=np.min(np.concatenate((reslN[reslN[:,10]!=0,1],resN[resN[:,10]!=0,1]))), vmax=np.max(np.concatenate((reslN[reslN[:,10]!=0,1],resN[resN[:,10]!=0,1])))) #Set manually this time.
	#For the colour map
    fig= plt.subplot()
    

    for i in range(len(reslN)):
        if (reslN[i,10]!=0 and reslN[i,11]>0) and (resN[i,10]!=0 and resN[i,11]>0):
            #large N term has half-transparency.
            plt.scatter(reslN[i,11], reslN[i,10], c = reslN[i,1], alpha=1/(2), marker=markers[-1], cmap=colormap, norm=normalize)
            #Normal term has full-transparency.
            plt.scatter(resN[i,11], resN[i,10], c = resN[i,1], alpha=1/(1), marker=markers[-1], cmap=colormap, norm=normalize)
            #Pops off the last marker to move onto the next one.
            markers.pop()

    fig.set_xscale("log")
    fig.set_yscale("log")
    
    ##Set the following plotting parameters as required:
    
    #fig.set_ylim([0.01484,0.01532])
    #fig.set_yticks([], minor=True)
    #fig.set_xticks([], minor=True)
    #fig.set_xticks([1.75E4, 2E4, 2.25E4, 2.5E4])
    #fig.set_xticklabels([1.75E4, 2E4, 2.25E4, 2.5E4])
    #fig.set_yticks([1.49E-2,1.5E-2,1.51E-2,1.52E-2,1.53E-2])
    #fig.set_yticklabels([1.49E-2,1.5E-2,1.51E-2,1.52E-2,1.53E-2])

    plt.colorbar(label=r"$m^2_{\eta'}~[GeV^2]$")
    fig.set_xlabel(r'$\beta/H$',fontsize=13)
    fig.set_ylabel(r'$\alpha$',fontsize=15)
    plt.tight_layout()
    debug_plot(name="debug", overwrite=False)
    #plt.show()
    

    plt.savefig('DifferencePlotN3F6.pdf',bbox_inches="tight")

def plotArrows(reslN, resN):
    colormap = plt.cm.viridis #or any other colormap
    normalize = matplotlib.colors.Normalize(vmin=np.min(np.concatenate((reslN[reslN[:,10]!=0,1],resN[resN[:,10]!=0,1]))), vmax=np.max(np.concatenate((reslN[reslN[:,10]!=0,1],resN[resN[:,10]!=0,1])))) #Set manually this time.
	#For the colour map
    fig,ax= plt.subplots()
    
    for i in range(len(reslN)):
        if (reslN[i,10]!=0 and reslN[i,11]>0) and (resN[i,10]!=0 and resN[i,11]>0):
            
            dx = reslN[i,11]-resN[i,11] #Arrow length in x-direction
            dy = reslN[i,10]-resN[i,10] #Arrow length in y-direction
            
            #plt.arrow(,dx,dy,head_width=0.0008, head_length=10,length_includes_head=True)
            ax.annotate("HELLO", xytext=(resN[i,11],resN[i,10]), xy=(dx,dy), arrowprops=dict(arrowstyle="->"))

            #large N term has half-transparency.
            ax.scatter(reslN[i,11], reslN[i,10], c = reslN[i,1], cmap=colormap, norm=normalize)
            #Normal term has full-transparency.
            ax.scatter(resN[i,11], resN[i,10], c = resN[i,1], alpha=1/(1), cmap=colormap, norm=normalize)
            

    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ##Set the following plotting parameters as required:
    
    #fig.set_ylim([0.01484,0.01532])
    #fig.set_yticks([], minor=True)
    #fig.set_xticks([], minor=True)
    #fig.set_xticks([1.75E4, 2E4, 2.25E4, 2.5E4])
    #fig.set_xticklabels([1.75E4, 2E4, 2.25E4, 2.5E4])
    #fig.set_yticks([1.49E-2,1.5E-2,1.51E-2,1.52E-2,1.53E-2])
    #fig.set_yticklabels([1.49E-2,1.5E-2,1.51E-2,1.52E-2,1.53E-2])

    #plt.colorbar(label=r"$m^2_{\eta'}$")
    ax.set_xlabel(r'$\beta/H$',fontsize=13)
    ax.set_ylabel(r'$\alpha$',fontsize=15)
    plt.tight_layout()
    debug_plot(name="debug", overwrite=False)
    #plt.show()
    
    
	
#Load up the data.
resN = np.genfromtxt(f'Test_N3F6_Normal.csv', delimiter=',', dtype=float, skip_header=1)
reslN = np.genfromtxt(f'Test_N3F6_largeN.csv', delimiter=',', dtype=float, skip_header=2)

plotDifference(reslN,resN)

#plotArrows(reslN,resN)