import Potential2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate, optimize
import csv


#Plots the potential as a function of temperature

def plotV(V, Ts):
	fSig = V.fSigma()
	for T in Ts:
		plt.plot(np.linspace(-5,fSig*1.2,num=300),V.Vtot(np.linspace(-5,fSig*1.2,num=300),T)-V.Vtot(0,T),label=f"T={T}")
	plt.legend()
	plt.show()
	
def splitV(V, T):
    fSig = V.fSigma()
    plt.plot(np.linspace(-5,fSig*1.2,num=300),V.Vtot(np.linspace(-5,fSig*1.2,num=300),T)-V.Vtot(0,T),label=f"$V_{{tot}}$")
    plt.plot(np.linspace(-5,fSig*1.2,num=300),V.V(np.linspace(-5,fSig*1.2,num=300))-V.V(0),label=f"$V_{{0}}$")
    plt.plot(np.linspace(-5,fSig*1.2,num=300),V.V1T(np.linspace(-5,fSig*1.2,num=300),T)-V.V1T(0,T),label=f"$V_{{th}}$")
    plt.plot(np.linspace(-5,fSig*1.2,num=300),V.VGluonic(np.linspace(-5,fSig*1.2,num=300),T)-V.VGluonic(0,T),label=f"$V_{{Gluonic}}$")
    plt.legend()
    plt.show()
N=3; F=6
'''
m2Sig = [10E3,2*10E5]
m2Eta = [10E3, 2*10E4] 
m2X = [8*10E2,10E4] 
m2Pi = [8*10E2,10E4]
FPI=600'''

#m2Sig, m2Eta, m2X, m2Pi.

m2Sig = np.linspace(10E3, 2*10E5, num=2)
m2Eta = np.linspace(10E3, 2*10E4, num=7)
m2X = np.linspace(8*10E2,10E4, num=10)
m2Pi = np.linspace(8*10E2,10E4, num=10)

data = []
for i in m2Sig:
    for j in m2Eta:
        for k in m2X:
             for l in m2Pi:
                  data.append([i,j,k,l])



i=0
for point in data:
    inputs = Potential2.masses_to_lagrangian_Csaki(*point, N, F)
    if inputs[0] is not None:
        i+=1
        V = Potential2.Potential(*inputs, N, F, CsakiTerm=True)
        print(Potential2.masses_to_lagrangian_Csaki(*point, N, F))
        plotV(V,[495,500,505,510])
        splitV(V,505)
    if i>3:
         break
	