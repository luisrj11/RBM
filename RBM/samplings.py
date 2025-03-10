# packes
import numpy as np
from joblib import Parallel, delayed

from RBM.algorithms.algorithms import Algorithms

'''
===============================================================================
Class Samplings:

Here it the class to creat the data base to apply the statistic technique

To use thi class aalways has to pass all paramter with exception Step_parameter
becuase it is directly configure:

Step_parameter = 1.0  for the Metropolis
Step_parameter = 0.05 for the Metropolis Hastings

You can change when you call the method samplings

Information about parallel proces:

https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html

https://www.geeksforgeeks.org/python-os-cpu_count-method/
===============================================================================
'''

class Samplings(Algorithms):
    def __init__(self, Number_particles:int,Dimension:int, Number_hidden_layer:int, Interaction = False,\
                     Sigma = 1.0 ,Algorithm:str = None, Number_MC_cycles = None, Step_parameter = None) -> None:
        
        '-----------------------------------------------'
        self.Number_particles = Number_particles
        self.Dimension = Dimension
        self.Number_hidden_layer = Number_hidden_layer
        self.Interaction = Interaction
        self.Sigma = Sigma
        '-----------------------------------------------'
        self.Algorithm = Algorithm
        self.Number_MC_cycles = Number_MC_cycles
        '-----------------------------------------------'
        self.Step_parameter = Step_parameter
        '-----------------------------------------------'

    def samplings(self,Optimal_parameter_a_b_w,Number_samplings,Number_core = 1):

        algorithm = super().set_algorithm(
                               Number_particles = self.Number_particles,
                               Dimension = self.Dimension,
                               Number_hidden_layer = self.Number_hidden_layer,
                               Interaction = self.Interaction,
                               Sigma = self.Sigma,
                               Algorithm = self.Algorithm, 
                               Number_MC_cycles = self.Number_MC_cycles,
                               Step_parameter = self.Step_parameter)
        try:
            a = Optimal_parameter_a_b_w[0]
            b = Optimal_parameter_a_b_w[1]
            w = Optimal_parameter_a_b_w[2]
            Number_samplings = int(Number_samplings)
        except:
            return(f'< ERROR > : The Optimal_parameter_a_b_w {Optimal_parameter_a_b_w} has to be a tuple Optimal_parameter_a_b_w = (a,b,w)')
        
        result = Parallel(n_jobs=Number_core)(delayed(algorithm)(a,b,w) for Ns in range(Number_samplings))

        result = np.array(result)

        Energies = result[:,0]
        Variances = result[:,1]
        errors = result[:,2]
        Time_consuming = result[:,3]

        return Energies, Variances, errors, Time_consuming
 
if __name__ == "__main__":

    from numpy.random import uniform

    print('==================================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('==================================================')

    Number_particles = 1
    Dimension = 1
    Number_hidden_layer = 1
    Number_MC_cycles = 10**3

    Number_samplings = 3
    Number_core = 2

    a = uniform(0.0, 0.001, size=(Number_particles,Dimension))
    b = uniform(0.0, 0.001, size= (Number_hidden_layer))
    w = uniform(0.0, 0.001, size=(Number_particles, Dimension, Number_hidden_layer))

    print(a)

    Optimal_parameter_a_b_w = (a,b,w)    

    print('====================== Energy using Metropolis ======================')
    algorithms_1 = Samplings(Number_particles,Dimension,Number_hidden_layer,Interaction=False,\
                                              Algorithm='Metropolis',Number_MC_cycles= Number_MC_cycles)
    metropolis = algorithms_1.samplings(Optimal_parameter_a_b_w,Number_samplings,Number_core)
    
    Energies, Variances, Errors, Time_CPU = metropolis
    print(f'Energy: {Energies} | Variance: {Variances} | Error: {Errors} | Time CPU: {Time_CPU}')

    print('======================================================================')
    print('================== Energy using Metropolis Hastings ==================')
    algorithms_2 = Samplings(Number_particles,Dimension,Number_hidden_layer,Interaction=False,\
                                                      Algorithm='MetropolisHastings',Number_MC_cycles=Number_MC_cycles) 
    metropolis_hastings = algorithms_2.samplings(Optimal_parameter_a_b_w,Number_samplings,Number_core)
    
    Energies, Variances, Errors, Time_CPU = metropolis_hastings
    print(f'Energy: {Energies} | Variance: {Variances} | Error: {Errors} | Time CPU: {Time_CPU}')

    print('=====================================================================')

    
    '''
    import os
 
    cpuCount = os.cpu_count()   
    
    print("Number of CPUs in the system:", cpuCount)
    
    '''