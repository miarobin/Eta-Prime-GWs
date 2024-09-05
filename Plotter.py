import Potential
import GravitationalWave
import numpy as np
import matplotlib.pyplot as plt
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
	Tn,grd,message = GravitationalWave.grid(V,prnt=True)
	print(f"message = {message}")
	print(f"alpha = {GravitationalWave.alpha(V,Tn)}, Beta/H = {GravitationalWave.beta_over_H(V,Tn,grd)}, Wall Velocity = {GravitationalWave.wallVelocity(V,GravitationalWave.alpha(V,Tn), Tn)}")
		
	plt.plot(v/T, 140*np.ones((25)))
	plt.plot(v/T, grd(T))
	plt.xlabel(f"$v/T$")
	plt.ylabel(f"Log($S_3/T$)")
	
	return (Tn, GravitationalWave.alpha(V,Tn), GravitationalWave.beta_over_H(V,Tn,grd), GravitationalWave.wallVelocity(V,GravitationalWave.alpha(V,Tn), Tn), message)


def populate(lmb, kappa, m2Sig, muSig, xi):
	#Lambda, Kappa, m^2_Sigma, Mu_Sig, Xi.
	V = Potential.Potential(lmb, kappa, m2Sig, muSig, xi)

	Tn, grd, message = GravitationalWave.grid(V)
	
				
	if Tn is not None:
		alpha = GravitationalWave.alpha(V,Tn); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = GravitationalWave.wallVelocity(V, alpha, Tn)
		return (Tn, alpha, betaH, vw, message)
	else:
		return (0, 0, 0, 0, message)


	
if __name__ == "__main__":
	print('hello')
	
#Xi, mu_Sigma, Lambda, Kappa, m_Sigma
	data = np.array([
 [-250000., 100., 0.01, 0.01, 800.],
 [-250000., 100., 0.01, 0.01, 500.],
 [-250000., 1000., 0.01, 0.01, 5000.],
 [-360000., 900., 0.012, 0.03, 4432.79],
 [-250000., 1000., 0.01, 0.01, 2754.03],
 [-250000., 1000., 0.1, 0.1, 2500.],
 [-1.69*10^6, 2630., 3., 6., 706.415],
 [-2.5*10^7, 2000., 0.04, 0.1, 4341.02],
 [-9.*10^6, 2000., 1., 1., 600.],
 [-490000., 1000., 1.2, 1., 831.363],
 [-1.225*10^7, 2000., 0.04, 0.1, 4158.],
 [-1.225*10^7, 2000., 0.04, 0.1, 4158.],
 [-250000., 1000., 0.1, 0.1, 1793.59],
 [-490000., 1000., 1.2, 2., 534.703],
 [-640000., 1000., 0.8, 2., 674.2],
 [-250000., 1000., 0.1, 0.1, 1444.88],
 [-250000., 1000., 0.1, 0.1, 1444.88],
 [-250000., 3000., 0.1, 0.1, 4335.],
 [-1.*10^6, 1000., 1., 1., 268.361],
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
 [-1.*10^6, 3000., 1., 1., 1000.],
 [-250000., 1000., 0.5, 0.5, 350.],
 [-1.44*10^6, 2000., 0.1, 1., 1286.36],
 [-4.*10^6, 4000., 0.4, 0.3, 1043.29],
 [-250000., 9000., 1., 1., 3000.],
 [-90000., 1000., 0.3, 2., 481.234],
 [-9.*10^6, 3000., 0.1, 1., 20.],
 [-250000., 1000., 0.1, 2., 353.903],
 [-250000., 3000., 0.5, 0.01, 20.],
 [-9.*10^6, 3000., 0.01, 1., 10.],
 [-2.5*10^7, 7000., 0.1, 1.2, 5.],
 [-2.5*10^7, 7000., 0.1, 0.8, 5.],
 [-2.5*10^7, 8000., 0.1, 0.5, 5.],
 [-2.5*10^7, 9000., 0.1, 1., 5.],
 [-250000., 9000., 0.1, 1., 3000.],
 [-1.*10^6, 6000., 0.1, 1., 5.],
 [-250000., 9000., 0.1, 1., 100.],
 [-250000., 9000., 0.1, 1., 10.]
	])

	results = np.array([populate(row[2], row[3], row[4]**2, row[1], row[0]) for row in data])


	column_titles = ['Lambda', 'Kappa', 'm^2_Sigma', 'mu_Sigma', 'xi', 'Tn', 'Alpha', 'Beta']
	# File path to save the CSV
	file_path = f'Test_NF3_ASBPower_1.csv'
	save_arrays_to_csv(file_path, column_titles, data[:,0], data[:,1], data[:,2], data[:,3], results[:,0], results[:,1], results[:,2])
