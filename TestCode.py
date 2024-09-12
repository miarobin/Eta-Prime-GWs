import unittest
import Potential
import GravitationalWave
import numpy as np

#Terminal Colour Escape Sequences
RESET = "\033[0m"  # Reset all formatting
RED = "\033[31m"   # Red color
GREEN = "\033[32m" # Green color
CYAN = "\033[36m" # Cyan color


#For testing against the results from 1904.10967
#NOTE THE MASS ON THE INPUT MUST BE SQUARED!
input = [[-250000, 9000, 1, 1, 3000**2],
         [-250000., 2000., 0.1, 0.1, 2067.37**2],
         [-9.E6, 3000., 0.1, 1., 20.**2]]

effective_mass = [[6708.2, 9000., 8529.36, 866.025],
                  [4666.75, 6300.11, 5996.93, 866.025],
                  [2631.4, 4557.46, 7176.29, 5196.15]]
GW_params = [[5452.63, 0.00249046, 15762.3],
             [8257.33, 0.00105713, 6481.13],
             [3275.91, 0.00277528, 4398.28]]

class TestPotential(unittest.TestCase):
    #Xi, muSig, lmb, kappa, m2Sig, 


    def test_F3(self, index):
        #Xi, muSig, lmb, kappa, m2Sig, N, F
        V1 = Potential.Potential(*(input[index]),1,3)
        fSig1=V1.findminima(0)
        
        masses_got = [float(np.sqrt(m2(fSig1,0))) for m2, n in [V1.mSq['Phi'],V1.mSq['Eta'],V1.mSq['X'],V1.mSq['Pi']]]
        GW_got = GravitationalWave.gravitationalWave(V1)
        
        #Phi, Eta, X, Pi
        masses_expected = effective_mass[index]
        #Effective Masses:
        np.testing.assert_almost_equal(masses_got, masses_expected, decimal=2, err_msg=RED + f"Error in effective masses index {index}" + RESET)
        
        GW_exp = GW_params[index]
        
        #Tn
        errTn = abs(GW_exp[0] - GW_got[0])/GW_exp[0]
        if errTn > .1: print(CYAN + f"{round(errTn*100,2)}% difference in Tn for test {index}" + RESET)
        np.testing.assert_approx_equal(GW_got[0], GW_exp[0], significant = 2, err_msg=RED + f"Tn Incorrect index {index}" + RESET) 
        #alpha
        errAlp = abs(GW_exp[1] - GW_got[1])/GW_exp[1]
        if errAlp > .1: print(CYAN + f"{round(errAlp*100,2)}% difference in Alpha for test {index}" + RESET)
        np.testing.assert_allclose(GW_got[1], GW_exp[1], rtol=0.5, err_msg=RED + f"Alpha Incorrect index {index}" + RESET) 
        #beta/H
        errbH = abs(GW_exp[2] - GW_got[2])/GW_exp[2]
        if errbH > .1: print(CYAN + f"{round(errbH*100,2)}% difference in b/H for test {index}" + RESET)
        np.testing.assert_approx_equal(GW_got[2], GW_exp[2], significant = 1, err_msg=RED + f"Beta/H Incorrect index {index}" + RESET) #Less accurate due to derivative.



    def test_F4(self, index):
        print('hi')





if __name__ == '__main__':
    unittest.main()