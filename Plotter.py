import Potential
import GravitationalWave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate, optimize
import csv
from multiprocessing import Pool

##Important constants for the potential.
mh = 125.18; mW = 80.385; mZ = 91.1875; mt = 173.1; mb = 4.18; v = 246.; l = mh**2/v**2

def save_arrays_to_csv(file_path, column_titles, *arrays):
    # Transpose the arrays to align them by columns
    transposed_arrays = np.array(list(zip(*arrays)))
    

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        #Writes the column titles.
        writer.writerow(column_titles)
        
        #Writes in data rows.
        for row in transposed_arrays:
            writer.writerow(row)


def S3T(V,*T):
	#Produce a log plot of 3D action against temperature.
	if not T:
		T = np.linspace(v/1.8,v/2., num=25)
		
	highTmin = optimize.minimize(lambda h:V.Vtot(h,400)/v**4,-246,method="Nelder-Mead")
	print(highTmin)
	zeroTmax = optimize.minimize(lambda h:-V.Vtot(h,0)/v**4, -246,method="Nelder-Mead")
	print(zeroTmax)
	print(f"High T Minimum: {highTmin.x[0]}; Zero T Local Max = {zeroTmax.x[0]}")
	Tn,grd,message = GravitationalWave.grid(V,prnt=True,plot=True)
	print(f"message = {message}")
	print(f"alpha = {GravitationalWave.alpha(V,Tn)}, Beta/H = {GravitationalWave.beta_over_H(V,Tn,grd)}, Wall Velocity = {GravitationalWave.wallVelocity(V,GravitationalWave.alpha(V,Tn), Tn)}")
		
	plt.plot(v/T, 140*np.ones((25)))
	plt.plot(v/T, grd(T))
	plt.xlabel(f"$v/T$")
	plt.ylabel(f"Log($S_3/T$)")
	
	return (Tn, GravitationalWave.alpha(V,Tn), GravitationalWave.beta_over_H(V,Tn,grd), GravitationalWave.wallVelocity(V,GravitationalWave.alpha(V,Tn), Tn), message)

def populate(xi, muSig, lmb, kappa, m2Sig, N, F,plot=False):
	#Lambda, Kappa, m^2_Sigma, Mu_Sig, Xi.
	V = Potential.Potential(xi, muSig, lmb, kappa, m2Sig, N, F)
	fSig = V.findminima(0) 
	print(fSig)
	
	if plot:
		#Plots the potential as a function of temperature
		def plotV(V, Ts):
			for T in Ts:
				plt.plot(np.linspace(-5,500,num=100),V.Vtot(np.linspace(-5,500,num=100),T)-V.Vtot(0,T),label=f"T={T}")
			plt.legend()
			plt.show()
		plotV(V,[0,100,200])
	if fSig == None:
		return (0, 0, 0, 0, 15)
	
	print(1)
	
	massRatio = abs(V.mSq['Phi'][0](fSig,0)/V.mSq['Eta'][0](fSig,0))

	print(2)
	Tn, grd, message = GravitationalWave.grid(V,prnt=True,plot=plot)
	
	if Tn is not None:
		alpha = abs(GravitationalWave.alpha(V,Tn)); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = GravitationalWave.wallVelocity(V, alpha, Tn)
		print(f"Tn = {Tn}, alpha = {alpha}, betaH = {betaH}, massRatio = {massRatio}")
		return (Tn, alpha, betaH, massRatio, message)
	else:
		print('CT Returned None')
		return (0, 0, 0, massRatio, message)
	

def plotDifference(data, results, Ns):
	results = np.array(results)
	#So you can see exactly where the points have moved to
	markers = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","D","d","|"]
	
	colormap = plt.cm.plasma #or any other colormap
	normalize = matplotlib.colors.Normalize(vmin=min(np.ravel(results[:,:,3])), vmax=max(np.ravel(results[:,:,3])))
	#For the colour map

	for i, point in enumerate(data):
		if all([results[Ni,i,0]>0 for Ni,N in enumerate(Ns)]):
			for Ni,N in enumerate(Ns):
				plt.scatter(results[Ni,i,2], results[Ni,i,1], c = results[Ni,i,3], alpha=1/(Ni+1), marker=markers[-1], cmap=colormap, norm=normalize)
			markers.pop()

			
	plt.xscale("log")
	plt.yscale("log")
	plt.colorbar()
	plt.xlabel(f'beta/H')
	plt.ylabel(f'alpha')
	plt.show()

	
if __name__ == "__main__":
	print('hello')
	

#Xi, mu_Sigma, Lambda, Kappa, m_Sigma
	dataF3 = np.array([
 [-250000., 100., 0.01, 0.01, 800.],
 [-250000., 100., 0.01, 0.01, 500.],
 [-250000., 1000., 0.01, 0.01, 5000.],
 [-360000., 900., 0.012, 0.03, 4432.79],
 [-250000., 1000., 0.01, 0.01, 2754.03],
 [-250000., 1000., 0.1, 0.1, 2500.],
 [-1.69E6, 2630., 3., 6., 706.415],
 [-2.5E7, 2000., 0.04, 0.1, 4341.02],
 [-9.0E6, 2000., 1., 1., 600.],
 [-490000., 1000., 1.2, 1., 831.363],
 [-1.225E7, 2000., 0.04, 0.1, 4158.],
 [-1.225E7, 2000., 0.04, 0.1, 4158.],
 [-250000., 1000., 0.1, 0.1, 1793.59],
 [-490000., 1000., 1.2, 2., 534.703],
 [-640000., 1000., 0.8, 2., 674.2],
 [-250000., 1000., 0.1, 0.1, 1444.88],
 [-250000., 1000., 0.1, 0.1, 1444.88],
 [-250000., 3000., 0.1, 0.1, 4335.],
 [-1.E6, 1000., 1., 1., 268.361],
 [-250000., 1000., 1., 1., 500.],
 [-250000., 2000., 0.1, 0.1, 2067.37],
 [-250000., 2000., 0.1, 0.1, 1832.19],
 [-250000., 3000., 0.1, 0.1, 2748.29],
 [-250000., 3000., 0.1, 0.1, 2748.29],
 [-250000., 1000., 0.1, 0.1, 870.901],
 [-250000., 1000., 1., 2., 233.785],
 [-250000., 1000., 1., 1., 275.403],
 [-250000., 1000., 0.1, 0.1, 700.],
 [-250000., 1000., 0.1, 0.1, 697.217],
 [-250000., 1000., 0.5, 2., 488.458],
 [-250000., 2000., 0.1, 0.1, 1394.43],
 [-250000., 1000., 0.1, 1., 1000.],
 [-1.E6, 3000., 1., 1., 1000.],
 [-250000., 1000., 0.5, 0.5, 350.],
 [-1.44E6, 2000., 0.1, 1., 1286.36],
 [-4.E6, 4000., 0.4, 0.3, 1043.29],
 [-250000., 9000., 1., 1., 3000.],
 [-90000., 1000., 0.3, 2., 481.234],
 [-9.E6, 3000., 0.1, 1., 20.],
 [-250000., 1000., 0.1, 2., 353.903],
 [-250000., 3000., 0.5, 0.01, 20.],
 [-9.E6, 3000., 0.01, 1., 10.],
 [-2.5E7, 7000., 0.1, 1.2, 5.],
 [-2.5E7, 7000., 0.1, 0.8, 5.],
 [-2.5E7, 8000., 0.1, 0.5, 5.],
 [-2.5E7, 9000., 0.1, 1., 5.],
 [-250000., 9000., 0.1, 1., 3000.],
 [-1.E6, 6000., 0.1, 1., 5.],
 [-250000., 9000., 0.1, 1., 100.],
 [-250000., 9000., 0.1, 1., 10.]
	])
	
	dataF3 = dataF3[:20]
	
	dataF4 = np.array(
	[[1., 0.0159439, 1., 279.801],
 [0.1832, 0.0400589, 0.0340461, 184.187],
 [0.0782609, 0.0152908, 0.0264089, 406.113],
 [0.40943, 0.145084, 0.0147786, 1822.93],
 [7.11314, 1.29084, 3.11112, 2500.31],
 [4.80576, 0.482538, 3.21777, 781.017],
 [2., 0.308642, 2., 481.431],
 [1., 0.288504, 0.4, 2117.03],
 [0.5, 0.12716, 0.3, 2788.96],
 [1., 0.059453, 1., 547.209],
 [3., 0.333554, 2.8, 549.451],
 [0.5, 0.0784602, 0.3, 2000.],
 [2., 0.252551, 1.5, 614.103],
 [3., 0.408554, 2.5, 480.177],
 [1., 0.279321, 0.5, 719.534],
 [0.0716646, 0.0134228, 0.0206629, 586.858],
 [0.5, 0.090625, 0.2, 1000.],
 [1., 0.094518, 1., 1503.06],
 [0.7, 0.0634564, 0.5, 497.923],
 [3., 0.0877929, 2.8, 648.089],
 [1., 0.146701, 0.5, 447.919],
 [0.5, 0.0794444, 0.2, 1354.35],
 [1., 0.08, 1., 2032.78],
 [0.566423, 0.139436, 0.0522319, 2460.73],
 [4., 0.195313, 4., 404.754],
 [7.40275, 1.78942, 0.887677, 2331.09],
 [0.916804, 0.0513227, 0.81068, 1409.14],
 [1., 0.0635802, 0.9, 604.448],
 [1.8, 0.262327, 1., 1531.72],
 [0.7, 0.067284, 0.5, 433.415],
 [0.886234, 0.152521, 0.322259, 604.578],
 [0.873178, 0.182685, 0.167786, 681.383],
 [0.448218, 0.100183, 0.0611488, 138.485],
 [4., 0.059453, 4., 1511.87],
 [1., 0.0236295, 1., 1481.69],
 [0.8, 0.193036, 0.05, 800.],
 [6.96888, 1.62827, 0.627876, 2554.57],
 [0.5, 0.00510204, 0.5, 1000.],
 [4., 0.034626, 4., 1778.36],
 [1., 0.0171468, 1., 1860.15],
 [1., 0.0388889, 0.9, 685.706],
 [1.54993, 0.235733, 0.672107, 1615.88],
 [0.374256, 0.0495923, 0.185553, 1616.35],
 [1., 0.0111383, 1., 1193.82],
 [0.5, 0.0294444, 0.4, 391.322],
 [1., 0.0118343, 1., 1207.9],
 [1., 0.0118343, 1., 1207.9],
 [1.05, 0.0187027, 1., 268.541],
 [3., 0.0193698, 3., 719.424],
 [1., 0.00617284, 1., 295.559]])
	

	#dataF4 = dataF4[:5]; F = 4

	#N=1; F=4
	#res = [populate(0., row[0],row[1],row[2],row[3]**2, N=N, F=4,plot=False) for row in dataF4[2:3]]
	res = populate(0,386.79**2/2,+31.51/2,-82.77/4,263.83**2, N=2, F=4, plot=True)
	print(res)

	'''
	Ns = [1, 2]
	fPi = 2000
	results = []
	
	def fSig1_function(muSig, lmb, kappa, m2Sig): return 2*(2*m2Sig)**0.5 / (kappa + 4*lmb - muSig)**0.5

	for N in Ns:
		#Changing the power of the breaking term (1 is from Rachel's paper, 1/N is from Csaba's paper)
		#, xi, muSig, lmb, kappa, m2Sig, N, F, muSSI=0
		results_N = np.array([populate(0., row[0]*fPi**(4-4/N),row[1],row[2],row[3]**2, N=N, F=4) for row in dataF4])
		
		results.append(results_N)

		column_titles = ['mu_Sigma', 'Lambda', 'Kappa', 'm^2_Sigma', 'Tn', 'Alpha', 'Beta', 'MassRatio']
		# File path to save the CSV
		file_path = f'Test_N{N}F{F}_ASBPower.csv'
		save_arrays_to_csv(file_path, column_titles, dataF4[:,0], dataF4[:,1], dataF4[:,2], dataF4[:,3], results_N[:,0], results_N[:,1], results_N[:,2], results_N[:,3])

	plotDifference(dataF4, results, Ns)'''
	


