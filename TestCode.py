import unittest
import Potential
import GravitationalWave
import numpy as np

#Terminal Colour Escape Sequences
RESET = "\033[0m"  # Reset all formatting
RED = "\033[31m"   # Red color
GREEN = "\033[32m" # Green color
CYAN = "\033[36m" # Cyan color
ORANGE = "\033[3m" # Orange colour


#For testing against the results from 1904.10967
#NOTE THE MASS ON THE INPUT MUST BE SQUARED!
#Xi, muSig, lmb, kappa, m2Sig
input_F3 = [[-250000, 9000, 1, 1, 3000**2],
         [-250000., 2000., 0.1, 0.1, 2067.37**2],
         [-9.E6, 3000., 0.1, 1., 20.**2]]

effective_mass_F3 = [[6708.2, 9000., 8529.36, 866.025],
                  [4666.75, 6300.11, 5996.93, 866.025],
                  [2631.4, 4557.46, 7176.29, 5196.15]]
GW_params_F3 = [[5452.63, 0.00249046, 15762.3],
             [8257.33, 0.00105713, 6481.13],
             [3275.91, 0.00277528, 4398.28]]

#Xi, muSig, lmb, kappa, m2Sig, muSSI
input_F4 = [[-4.38845E7, 4.80576, 0.482538, 3.21777, 781.017**2, 232.292],
            [-86317.7, 0.7, 0.067284, 0.5, 433.415**2, 43721.9],
            [-575345., 1., 0.0111383, 1., 1193.82**2, 293.149],
            [-3.40724E7, 1., 0.0159439, 1., 279.801**2, 29404.1]]
            

effective_mass_F4 = [[1104.47, 5853.7, 12659.3, 11474., 9333.65, 7649.35, 5348.33, 10.7771],
                     [594.841, 2676.78, 2527.77, 498.019, 2499.15, 322.738, 2482.63, 147.855],
                     [1688.28, 11311.5, 11387.5, 1313.76, 11345.3, 875.816, 11311.5, 12.1068],
                     [376.664, 2109.32, 10327.6, 10109.9, 7062., 6739.63, 2112.8, 121.252]]

GW_params_F4 = [[2125.2, 0.00114566, 86553.3],
                [885.152, 0.028589, 3755.1],
                [2238.37, 0.0872054, 1803.78],
                [1251.95, 0.000408117, 62409.1]]

class TestPotential(unittest.TestCase):
    


    def test_F3(self, index):
        print(f"--- F = 3 Test {index} ---")
        #Xi, muSig, lmb, kappa, m2Sig, N, F
        V1 = Potential.Potential(*(input_F3[index]),1,3)
        fSig1=V1.findminima(0)
 
        
        masses_got = [float(np.sqrt(m2(fSig1,0))) for m2, n in [V1.mSq['Phi'],V1.mSq['Eta'],V1.mSq['X'],V1.mSq['Pi']]]
        GW_got = GravitationalWave.gravitationalWave(V1)
        
        #Phi, Eta, X, Pi
        masses_expected = effective_mass_F3[index]
        #Effective Masses:
        np.testing.assert_almost_equal(masses_got, masses_expected, decimal=2, err_msg=RED + f"Error in effective masses index {index}" + RESET)
        
        GW_exp = GW_params_F3[index]
        
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
        
        if errTn < .1 and errAlp < .1 and errbH < .1:        
            print(GREEN + f"--- Test {index} Passed ---" + RESET)
        else:
            print(ORANGE + f"--- Test {index} Passed with Accuracy Concerns ---" + RESET)






    def test_F4(self, index):
        print(f"--- F = 4 Test {index} ---")
        #Xi, muSig, lmb, kappa, m2Sig, muSSI, N, F
        V1 = Potential.Potential(*(input_F4[index][:-1]),1,4,muSSI = input_F4[index][-1])
        fSig1=V1.findminima(0)
        print(fSig1)
        
        masses_got = [float(np.sqrt(m2(fSig1,0))) for m2, n in [V1.mSq['Phi'],V1.mSq['Eta'],V1.mSq['X8'],V1.mSq['Pi8'],V1.mSq['X3'],V1.mSq['Pi3'],V1.mSq['EtaPsi'], V1.mSq['EtaChi']]]
        GW_got = GravitationalWave.gravitationalWave(V1)
        
        #Phi, Eta, X, Pi
        masses_expected = effective_mass_F4[index]
        #Effective Masses:
        errMass = (np.array(masses_got) - np.array(masses_expected))/np.array(masses_expected)
        np.testing.assert_allclose(masses_got, masses_expected, rtol=0.25, err_msg=RED + f"Error in effective masses index {index}" + RESET)
        if not all([em<.1 for em in errMass]): print(CYAN + f"Maximum of {round(max(errMass)*100,2)}% difference in effective masses for test {index}" + RESET)     

        GW_exp = GW_params_F4[index]
        
        #Tn
        errTn = abs(GW_exp[0] - GW_got[0])/GW_exp[0]
        if errTn > .1: print(CYAN + f"{round(errTn*100,2)}% difference in Tn for test {index}" + RESET)
        np.testing.assert_allclose(GW_got[0], GW_exp[0], rtol = 0.25, err_msg=RED + f"Tn Incorrect index {index}" + RESET) 
        #alpha
        errAlp = abs(GW_exp[1] - GW_got[1])/GW_exp[1]
        if errAlp > .1: print(CYAN + f"{round(errAlp*100,2)}% difference in Alpha for test {index}" + RESET)
        np.testing.assert_allclose(GW_got[1], GW_exp[1], rtol=0.5, err_msg=RED + f"Alpha Incorrect index {index}" + RESET) 
        #beta/H
        errbH = abs(GW_exp[2] - GW_got[2])/GW_exp[2]
        if errbH > .1: print(CYAN + f"{round(errbH*100,2)}% difference in b/H for test {index}" + RESET)
        np.testing.assert_approx_equal(GW_got[2], GW_exp[2], significant = 1, err_msg=RED + f"Beta/H Incorrect index {index}" + RESET) #Less accurate due to derivative.
        
        if errTn < .1 and errAlp < .1 and errbH < .1:        
            print(GREEN + f"--- Test {index} Passed ---" + RESET)
        else:
            print(ORANGE + f"--- Test {index} Passed with Accuracy Concerns ---" + RESET)




if __name__ == '__main__':
    unittest.main()