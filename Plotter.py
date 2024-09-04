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
	plt.title(f"$m_h =${mh}, $m_W =${mW}, $m_Z =${mZ}, $m_t =${mt}, $\lambda =${round(l,5)}")
	
	return (Tn, GravitationalWave.alpha(V,Tn), GravitationalWave.beta_over_H(V,Tn,grd), GravitationalWave.wallVelocity(V,GravitationalWave.alpha(V,Tn), Tn), message)


def populate(ep, delta, g4, beta):
	print(f"Epsilon = {ep}, Delta = {delta}")
	V = Potential.Potential(ep, g4, Potential.gammaa(ep,g4,delta),beta)

	lhs,rhs = V.findminima(0); Tn = None; message = 0
	

	if (rhs is None and optimize.minimize(lambda h:V.Vtot(h,400),-246).x[0] < lhs):
		Tn, grd, message = GravitationalWave.grid(V)
	
	if rhs is not None:
		highTmin = optimize.minimize(lambda h:V.Vtot(h,400)/v**4,(lhs-rhs)/2,bounds=[(lhs,rhs)],method='Nelder-Mead')
		zeroTmax = optimize.minimize(lambda h:-V.Vtot(h,0)/v**4, (lhs-rhs)/2,bounds=[(lhs,rhs)],method='Nelder-Mead')
		if highTmin.fun<10 and zeroTmax.fun<10:
			if (highTmin.x[0] < zeroTmax.x[0]): 
				Tn, grd, message = GravitationalWave.grid(V)
			else:
				return (ep, delta, 0, 0, 0, 0, 0)	
		else:
			message = 13
				
	if Tn is not None:
		alpha = GravitationalWave.alpha(V,Tn); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = GravitationalWave.wallVelocity(V, alpha, Tn)
		return (ep, delta, Tn, alpha, betaH, vw, message)
	else:
		return (ep, delta, 0, 0, 0, 0, message)



def parallelScan(eps, deltas,g4,beta):
	#MAKE THE ARRAY
	result=[]
        
	points = [(ep, delta, g4, beta) for ep in eps for delta in deltas]
	#Multithreading with 16 cores.
	with Pool(8) as p:
		   result = p.starmap(populate, points)
		
	
	Eps = []; Deltas = []; Tns = []; Alphas = []; Betas = []; Vws = []; Messages = []
	for item in result:
		Eps.append(item[0]); Deltas.append(item[1]); Tns.append(item[2]); Alphas.append(item[3]); Betas.append(item[4]); Vws.append(item[5]); Messages.append(item[6])

	print(f"eps = {len(Eps)},delta={len(Deltas)},Tns={len(Tns)},Alphas={len(Alphas)},Betas={len(Betas)},Vws={len(Vws)}")
	
	#MAKE THE FILE WRITER
	#Column Titles
	column_titles = ['Epsilon', 'Delta', 'Tn', 'Alpha', 'Beta', 'Vw', 'Message']
	# File path to save the CSV
	file_path = f'newscan_g4_{round(g4,2)}_beta_{round(beta,2)}.csv'
	save_arrays_to_csv(file_path, column_titles, Eps, Deltas, Tns, Alphas, Betas, Vws, Messages)
	
	#MAKE THE PLOT
	_Eps, _Deltas = np.meshgrid(sorted(Eps), sorted(Deltas))
	_Tns = interpolate.griddata((Eps, Deltas), Tns, (_Eps, _Deltas), method='linear')
	_Alphas = interpolate.griddata((Eps, Deltas), Alphas, (_Eps, _Deltas), method='linear')
	_Betas = interpolate.griddata((Eps, Deltas), Betas, (_Eps, _Deltas), method='linear')
	_Vws = interpolate.griddata((Eps, Deltas), Vws, (_Eps, _Deltas), method='linear')
			

	fig, ax = plt.subplots(2,2)
	im0 = ax[0,0].pcolormesh(_Eps, _Deltas, _Tns)
	cbar = plt.colorbar(im0)
	cbar.set_label(r"$T_n$")
	ax[0,0].set_xlabel(r"$\epsilon$")
	ax[0,0].set_ylabel(r"$\delta$")
	
	im1 = ax[0,1].pcolormesh(_Eps, _Deltas, _Alphas)
	cbar = plt.colorbar(im1)
	cbar.set_label(r"$\alpha$")
	ax[0,1].set_xlabel(r"$\epsilon$")
	ax[0,1].set_ylabel(r"$\delta$")
	
	im2 = ax[1,0].pcolormesh(_Eps, _Deltas, _Betas)
	cbar = plt.colorbar(im2)
	cbar.set_label(r"$\beta/H$")
	ax[1,0].set_xlabel(r"$\epsilon$")
	ax[1,0].set_ylabel(r"$\delta$")
	
	im3 = ax[1,1].pcolormesh(_Eps, _Deltas, _Vws)
	cbar = plt.colorbar(im3)
	cbar.set_label(r"$v_w$")
	ax[1,1].set_xlabel(r"$\epsilon$")
	ax[1,1].set_ylabel(r"$\delta$")

	plt.show()
	
	
def scan(eps, deltas, g4, beta):
	Tns = []; Alphas = []; Betas = []; Vws = []; Eps = []; Deltas = []
	
	for ep, delta in zip(eps,deltas):
		Eps.append(ep); Deltas.append(delta)
		print(f"Epsilon = {ep}, Delta = {delta}")
		V = Potential.Potential(ep, g4, Potential.gammaa(ep,g4,delta),beta)
			
		Tn, grd, message = GravitationalWave.grid(V,prnt=False)
		print(f'message = {message}')
				
		if Tn is not None:
			alpha = GravitationalWave.alpha(V,Tn); betaH = GravitationalWave.beta_over_H(V,Tn,grd); vw = GravitationalWave.wallVelocity(V, alpha, Tn)
			Tns.append(Tn); Alphas.append(alpha); Betas.append(betaH); Vws.append(vw)
		else:
			Tns.append(0); Alphas.append(0); Betas.append(0); Vws.append(0)

	column_titles = ['Epsilon', 'Delta', 'Tn', 'Alpha', 'Beta', 'Vw']
	# File path to save the CSV
	file_path = f'fixscan_g4_{round(g4,2)}_beta_{round(beta,2)}.csv'
	save_arrays_to_csv(file_path, column_titles, Eps, Deltas, Tns, Alphas, Betas, Vws)
	
	#MAKE THE PLOT
	_Eps, _Deltas = np.meshgrid(sorted(Eps), sorted(Deltas))
	_Tns = interpolate.griddata((Eps, Deltas), Tns, (_Eps, _Deltas), method='linear')
	_Alphas = interpolate.griddata((Eps, Deltas), Alphas, (_Eps, _Deltas), method='linear')
	_Betas = interpolate.griddata((Eps, Deltas), Betas, (_Eps, _Deltas), method='linear')
	_Vws = interpolate.griddata((Eps, Deltas), Vws, (_Eps, _Deltas), method='linear')
			

	fig, ax = plt.subplots(2,2)
	im0 = ax[0,0].pcolormesh(_Eps, _Deltas, _Tns)
	cbar = plt.colorbar(im0)
	cbar.set_label(r"$T_n$")
	ax[0,0].set_xlabel(r"$\epsilon$")
	ax[0,0].set_ylabel(r"$\delta$")
	
	im1 = ax[0,1].pcolormesh(_Eps, _Deltas, _Alphas)
	cbar = plt.colorbar(im1)
	cbar.set_label(r"$\alpha$")
	ax[0,1].set_xlabel(r"$\epsilon$")
	ax[0,1].set_ylabel(r"$\delta$")
	
	im2 = ax[1,0].pcolormesh(_Eps, _Deltas, _Betas)
	cbar = plt.colorbar(im2)
	cbar.set_label(r"$\beta/H$")
	ax[1,0].set_xlabel(r"$\epsilon$")
	ax[1,0].set_ylabel(r"$\delta$")
	
	im3 = ax[1,1].pcolormesh(_Eps, _Deltas, _Vws)
	cbar = plt.colorbar(im3)
	cbar.set_label(r"$v_w$")
	ax[1,1].set_xlabel(r"$\epsilon$")
	ax[1,1].set_ylabel(r"$\delta$")

	plt.show()
	
def euclideanScan(eps, deltas, g4, beta):
	Actions = []; Eps = []; Deltas = []
	
	for ep in eps:
		for delta in deltas:
			Eps.append(ep); Deltas.append(delta)
			print(f"Epsilon = {ep}, Delta = {delta}")
			V = Potential.Potential(ep, g4, Potential.gammaa(ep,g4,delta),beta)
			
			Actions.append(GravitationalWave.euclidAction(V,prnt=False))
			
	#MAKE THE PLOT
	_Eps, _Deltas = np.meshgrid(sorted(Eps), sorted(Deltas))
	_Actions = interpolate.griddata((Eps, Deltas), Actions, (_Eps, _Deltas), method='linear')

			
	ax = plt.subplot()
	im = ax.pcolormesh(_Eps, _Deltas, np.log10(_Actions))
	cbar = plt.colorbar(im)
	cbar.set_label(r"$S_4$")
	ax.set_xlabel(r"$\epsilon$")
	ax.set_ylabel(r"$\delta$")


	plt.show()
	

def bubbleWrap(ep, delta, g4, beta, prnt=False, plot=False):
	#Define the potential.
	V = Potential.Potential(ep,g4,Potential.gammaa(ep,g4,delta),beta,higgs_corr=True,loop=True,msbar=False)
	
	#Keeping track of progress.
	print(f"epsilon = {ep}; delta = {delta}")
	
	#Checking that the potential is the expected shape.
	lhs,rhs = V.findminima(0); bubble = 0; message = 0
	if prnt: print(f"lhs = {lhs}; rhs = {rhs}; thermal min = {optimize.minimize(lambda h:V.Vtot(h,1e6),-246,method='Nelder-Mead').x[0]}; local max = {optimize.minimize(lambda h:-V.Vtot(h,0), -246).x[0]}")
		

	#PLOTTING THE POTENTIAL
	if plot:
		v_star = V.vStar()
		Ts = [100,200]
		for T in Ts:
			plt.plot(np.linspace(-4.5,0.5, num=100)-v_star/v, V.Vtot(v*np.linspace(-4.5,0.5, num=100),T)/v**4-V.Vtot(v_star,T)/(v**4),label=f"T = {T} GeV")
		plt.legend()
		plt.show()
	
	if (rhs is None and optimize.minimize(lambda h:V.Vtot(h,1e6),-246,method="Nelder-Mead").x[0] > lhs):
		bubble, message = GravitationalWave.BubbleorNot(V,prnt=prnt)
				
	if (rhs is not None and optimize.minimize(lambda h:V.Vtot(h,1e5),-246,method="Nelder-Mead").x[0] > optimize.minimize(lambda h:-V.Vtot(h,0), -246).x[0]): 
		bubble,message = GravitationalWave.BubbleorNot(V,prnt=prnt)
	
	if prnt: print(f"message {message}")
	if bubble == None:
		bubble = 0
	return (ep, delta, bubble, message)

def binaryBubbleScan(eps,deltas,g4,beta):
	#MAKE THE ARRAY
	result=[]
        
	points = [(ep, delta, g4, beta) for ep in eps for delta in deltas]
	#Multithreading with 16 cores.
	with Pool(16) as p:
		   result = p.starmap(bubbleWrap, points)
		
	
	Eps = []; Deltas = []; Bubbles = []; Messages = []
	for item in result:
		Eps.append(item[0]); Deltas.append(item[1]); Bubbles.append(item[2]); Messages.append(item[3])

	print(Messages)

	#MAKE THE PLOT
	_Eps, _Deltas = np.meshgrid(sorted(Eps), sorted(Deltas))
	_Bubbles = interpolate.griddata((Eps, Deltas), Bubbles, (_Eps, _Deltas), method='linear')
	_Messages = interpolate.griddata((Eps, Deltas), Messages, (_Eps, _Deltas), method='linear')

	fig, ax = plt.subplots(1,2)
	ax1 = ax[0]; ax2 = ax[1]
	im = ax1.pcolormesh(_Eps, _Deltas, _Bubbles)
	cbar = plt.colorbar(im)
	cbar.set_label(r"To Bubble or Not to Bubble")
	ax1.set_xlabel(r"$\epsilon$")
	ax1.set_ylabel(r"$\delta$")
	
	im = ax2.pcolormesh(_Eps, _Deltas, _Messages)
	cbar = plt.colorbar(im)
	cbar.set_label(r"Message")
	ax2.set_xlabel(r"$\epsilon$")
	ax2.set_ylabel(r"$\delta$")

	plt.show()


	
if __name__ == "__main__":
	eps =0.06377551020408163; g4 = 1.6; delta=+0.05571428571428570; beta = np.sqrt(0.1)

	#print(GravitationalWave.euclidAction(V,prnt=False))
	epss = [	
0.0570588235294118	
		]
	
	deltas = [
-0.0388235294117647
		]

	V = Potential.Potential(epss[0],g4,Potential.gammaa(epss[0],1.6,deltas[0]),np.sqrt(0.1),higgs_corr=True,loop=True,msbar=False)
	print(f"gamma_eps={V.gammaedelta()}")
	print(f"eps = {V.ep}")
	print(f"gamma_a={V.ga}")
	#V.minmaxPlot(50,200)
	plt.title(f"epsilon = {epss[0]}, delta = {deltas[0]}")
	plt.xlabel('T')
	plt.ylabel('h')
	plt.show()
	scan(epss,deltas,1.6,np.sqrt(0.1))

	for eps, delta in zip(epss,deltas):
		V = Potential.Potential(eps,g4,Potential.gammaa(eps,g4,delta),beta,higgs_corr=True,loop=True,msbar=False)
		#GravitationalWave.plotV(V,[0,50,100,200])
		S3T(V)
		plt.show()

	#eps = np.linspace(-0.02,-0.04,num=10)
	#deltas = np.linspace(0.02,0.08,num=10)
	#euclideanScan(eps, deltas, 0.8, np.sqrt(0.3))
	
	#binaryBubbleScan(eps, deltas, .9, np.sqrt(0.1))
	#binaryBubbleScan([-0.14,-0.02], [0.04,0.14], .8, np.sqrt(0.1))

	
	eps = np.concatenate((np.linspace(0.055,0.065,num=35),[1-np.sqrt(8/9),0.0572,0.05718]))
	deltas = np.linspace(-0.01,-0.15,num=35)
	parallelScan(eps,deltas,1.6,np.sqrt(0.1))
	
	#profiler.disable()
	#profiler.print_stats(sort='pcalls')

