# Packes
from numpy.random import uniform

#from VMCM.Hamiltonian.LocalEnergy import LocalEnergy
#from VMCM.utils.Type import Type
from RBM.algorithms.metroplis import Metropolis
from RBM.algorithms.metropolis_hastings import MetropolisHastings 
'''
===============================================================================
Algorithm use the class Methoplis and Metropolis Hastings and put together in 
a super class

To instance this class you do not need to pass any arguemnt because all parameter
are configured in the method call set_algorithm.

set_algorithm is used to configure the tipe of algorithm but the other parameter too.

You could not pass Number_MC_cycles, Step_parameter because they can be
configure when you going to use the algorithm.

Step_parameter: can be represent the Step_jumping or Steptime, it depends on which 
algorithms are used.
===============================================================================
'''

class Algorithms(Metropolis,MetropolisHastings):
        
    def __init__(self) -> None:
        pass
    '''
    ===============================================================================
    ------------------
    Set up algorithm :
    ------------------

    ===============================================================================
    '''

    # Setting algorithm and type of calculation
    def set_algorithm(self, Number_particles:int,Dimension:int, Number_hidden_layer:int, Interaction = False,\
                     Sigma = 1.0 ,Algorithm:str = None, Number_MC_cycles = None, Step_parameter = None):
        
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
        self.Step_size_jumping = Step_parameter
        self.Time_step = Step_parameter
        '-----------------------------------------------'
        if type(self.Algorithm) == str : 
            if self.Algorithm == 'Metropolis' :
                metropolis_algorithm = super().metropolis_algorithm
                return metropolis_algorithm
                
            elif self.Algorithm == 'MetropolisHastings':
                metropolis_hastings_algorithm =super().metropolis_hastings_algorithm 
                return metropolis_hastings_algorithm
            else:
                return(
                f'''< ERROR > : {self.Algorithm} it in not set up. 
                ===> Option: algorithm = Metropolis or algorithm = MetropolisHastings'''
                )
        else:
            return(
            f'''< ERROR > : algorithm has to be a string variable.
            ===> Option: algorithm = Metropolis or algorithm = MetropolisHastings'''
            )
    
if __name__ == "__main__":

    print('==================================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('==================================================')

    Number_particles = 5
    Dimension = 2
    Number_hidden_layer = 10
    Number_MC_cycles = 10**4

    a = uniform(0.0, 0.001, size=(Number_particles,Dimension))
    b = uniform(0.0, 0.001, size= (Number_hidden_layer))
    w = uniform(0.0, 0.001, size=(Number_particles, Dimension, Number_hidden_layer))

    print('====================== Energy using Metropolis ======================')
    algorithms_1 = Algorithms()
    metropolis = algorithms_1.set_algorithm(Number_particles,Dimension,Number_hidden_layer,Interaction=True,\
                                              Algorithm='Metropolis',Number_MC_cycles=Number_particles)
    
    Energy, Variance, Error, Time_CPU = metropolis(a,b,w,Number_MC_cycles)
    print(f'Energy: {Energy} | Variance: {Variance} | Error: {Error} | Time CPU: {Time_CPU}')

    print('======================================================================')
    print('================== Energy using Metropolis Hastings ==================')
    algorithms_2 = Algorithms() 
    metropolis_hastings = algorithms_2.set_algorithm(Number_particles,Dimension,Number_hidden_layer,Interaction=True,\
                                                      Algorithm='MetropolisHastings',Number_MC_cycles=Number_particles)
    
    Energy, Variance, Error, Time_CPU = metropolis_hastings(a,b,w,Number_MC_cycles)
    print(f'Energy: {Energy} | Variance: {Variance} | Error: {Error} | Time CPU: {Time_CPU}')

    print('=====================================================================')

