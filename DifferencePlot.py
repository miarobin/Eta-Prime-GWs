import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate, optimize
import csv
from multiprocessing import Pool

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"]= 11


def plotDifference(resC, resN):
    #So you can see exactly where the points have moved to
    markers = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","D","d","|"]
	
    colormap = plt.cm.viridis #or any other colormap
    normalize = matplotlib.colors.Normalize(vmin=3, vmax=20)
	#For the colour map
    fig= plt.subplot()

    for i in range(len(resC)):
        if resC[i,2]!=0 and resN[i,2]!=0:
            plt.scatter(resC[i,2], resC[i,1], c = resC[i,3], alpha=1/(3), marker=markers[-1], cmap=colormap, norm=normalize)
            plt.scatter(resN[i,2], resN[i,1], c = resN[i,3], alpha=1/(1), marker=markers[-1], cmap=colormap, norm=normalize)
            markers.pop()

    fig.set_xscale("log")
    fig.set_yscale("log")
    #fig.set_ylim([0.01484,0.01532])
    #fig.set_yticks([], minor=True)
    #fig.set_xticks([], minor=True)
    #fig.set_xticks([1.75E4, 2E4, 2.25E4, 2.5E4])
    #fig.set_xticklabels([1.75E4, 2E4, 2.25E4, 2.5E4])
    #fig.set_yticks([1.49E-2,1.5E-2,1.51E-2,1.52E-2,1.53E-2])
    #fig.set_yticklabels([1.49E-2,1.5E-2,1.51E-2,1.52E-2,1.53E-2])

    plt.colorbar(label=r"Mass Ratio $m_\sigma/m_{\eta'}$")
    fig.set_xlabel(r'$\beta/H$',fontsize=13)
    fig.set_ylabel(r'$\alpha$',fontsize=15)
    plt.tight_layout()
    plt.show()
    
    #plt.savefig('DifferencePlotN3F6.pdf',bbox_inches="tight")

	

resN = np.genfromtxt(f'Test_N3F6_Normal.csv', delimiter=',', dtype=float, skip_header=1)[:,4:]
resC = np.genfromtxt(f'Test_N3F6_Csaki.csv', delimiter=',', dtype=float, skip_header=1)[:,4:]

plotDifference(resC,resN)