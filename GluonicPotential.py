import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import integrate


def VMedium(phi, l, T):
	if T==0: return 0

	eps_P = lambda x: np.sqrt(x**2 + phi**2)
	
	return -2*3*T/np.pi**2 * integrate.quad(lambda x: x**2 * (np.log(1 + (3*l-1)*np.exp(-eps_P(x)/T)+np.exp(-2*eps_P(x)/T)) + np.log(1 + np.exp(-eps_P(x)/T))),0,np.inf)[0]
	
a0 = 3.72; a1 = -5.73; a2 = 8.49; a3 = -9.29; a4 = 0.27; b3 = 2.4; b4 = 4.53
Tc = 600

def VPoly(l, T):
	if T == 0:
		return 0

	b2T = lambda _T :a0 + a1*(Tc/_T) + a2*(Tc/_T)**2 + a3*(Tc/_T)**3 + a4*(Tc/_T)**4

	return T**4 * (-b2T(T)/2 * l**2 + b4 * l**4 - 2*b3*l**3)



def VGluonic(phi,T):
	gluonic = lambda l,_phi: VPoly(l, T) + VMedium(phi, l, T)
	
	res = optimize.minimize(lambda l: gluonic(l, phi), 0.5, bounds=[(0,1)]).x[0]
	#print(f'sigma = {phi}, l = {res}')
	return VPoly(res, T) + VMedium(phi, res, T)

data = np.genfromtxt(f'gridDataF3N3.csv', delimiter=',', dtype=float, skip_header=0)
plt.plot(data[306:346,1], data[306:346,2],color='green',label='Martha')

print(data[306:346])

plt.plot(data[306:346,1],[VGluonic(sig,76) for sig in data[306:346,1]],color='red',label='Mia')
plt.legend()
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$V_{gluonic}$')
plt.title('T=76 GeV')
plt.show()