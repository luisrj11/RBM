# Packages
import numpy as np
from numpy.random import normal
from time import time

from RBM.optimizator.energy_its_derivative import EnergyItsDerivative
'''
===============================================================================
Class Optimizator 

To instance this class always has to be necessary pass a NUmber particles,
dimencion and the hidden layer.

for : 
    Interaction : bool 
    Number_MC_cycles : int
    Step_parameter : float 

can be instance when calls a class but it is not necesary becuase you will be
able to configure when call the method metropolis_algorithm
===============================================================================
'''
class Optimizator(EnergyItsDerivative):
    def __init__(self, Number_particles: int, Dimension: int, Number_hidden_layer: int, Interaction=False,\
                Sigma=1, Algorithm:str = None, Number_MC_cycles=None, Step_parameter=None) -> None:
        super().__init__(Number_particles, Dimension, Number_hidden_layer, Interaction, Sigma, Number_MC_cycles, Step_parameter)
        self.Algorithm = Algorithm
        self.Step_parameter = Step_parameter

    '''
    ===============================================================================
    -------------------------
    Gradient descent method :
    -------------------------

    It does not need any parameter by default is confugured learnung rate 
    (learning_rate = 0.001), Number iteration (Maximum_iterations = 50,
    and bounds gues for (a), (b) and (w) bounds_guess = (low-limit, up-limit), but
    it is possible to change it.
    ===============================================================================
    '''
    def gradient_descent(self,
                        learning_rate = 0.001,          # Learning rate (learning_rate),
                        Maximum_iterations = 50,        # Maximum iterations
                        bounds_guess = (0.0 ,0.001)     # Bounds guesses limit
                        ):
        
        # Starting time consuming
        Time_inicio = time()
        
        # Guess for parameters
        a = np.random.normal(loc= bounds_guess[0], scale= bounds_guess[1], size=(self.Number_particles, self.Dimension))
        b = np.random.normal(loc= bounds_guess[0], scale= bounds_guess[1], size=(self.Number_hidden_layer))
        w = np.random.normal(loc= bounds_guess[0], scale= bounds_guess[1], size=(self.Number_particles, self.Dimension, self.Number_hidden_layer))
        
        # Set up iteration using stochastic gradient method
        EDerivative = np.empty((3,),dtype=object)
        EDerivative = [np.copy(a),np.copy(b),np.copy(w)]
        
        # Saves energy in each iterations
        Energies = np.zeros(Maximum_iterations)

        # Choose algorithm 
        if type(self.Algorithm) == str :
            if self.Algorithm == 'Metropolis':
                energy_minimization = super().metropolis_optimizator

            elif self.Algorithm == 'MetropolisHastings':
                energy_minimization = super().metropolis_hastings_optimizator

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

        # To iterate gradiente descent 
        Iteration_number = np.zeros(0)
        iter = 0
        while iter < Maximum_iterations:
            
            Iteration_number = np.append(Iteration_number,iter)
            
            try:
                Energie_its_derivative = energy_minimization(a,b,w, self.Number_MC_cycles)
            except Exception as error:
                print(f'{error} ===> HINT: You should try a smaller learning rate that (learning_rate = {learning_rate})')
                break
            
            Energy = Energie_its_derivative[0]
            EDerivative = Energie_its_derivative[1]

            a_gradient = EDerivative[0]
            b_gradient = EDerivative[1]
            w_gradient = EDerivative[2]

            a -= learning_rate*a_gradient
            b -= learning_rate*b_gradient 
            w -= learning_rate*w_gradient 

            Energies[iter] = Energy

            iter += 1
            
        # Time CPU
        Time_fin = time()
        Time_consuming = Time_fin - Time_inicio
        return Energies, a , b, w , Iteration_number, Time_consuming

if __name__ == "__main__":
    print('==================================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('==================================================')

    Number_particles = 2
    Dimension = 2
    Number_hidden_layer = 2
    Number_MC_cycles = 10**4

    print('========================= Using Metropolis ==========================')
    algorithm_1 = Optimizator(Number_particles, Dimension, Number_hidden_layer, Interaction=False,\
                              Algorithm = 'Metropolis',Number_MC_cycles = Number_MC_cycles)
    Energies, a, b, w, Iteration_number, Time_CPU = algorithm_1.gradient_descent(Maximum_iterations=10)
    print(
    f'''==============================================================================
    Energy: {Energies}  
    ==============================================================================
    Optimization values (a): {a}
    ==============================================================================
    Optimization values (b): {b}
    ==============================================================================
    Optimization values (w): {w}
    ==============================================================================
    Time CPU: {Time_CPU} [seconds] | {Time_CPU/60.0} [minutes]
    ==============================================================================
    ''')
    
    print('===================== Using Metropolis Hastings =====================')
    algorithm_2 = Optimizator(Number_particles, Dimension, Number_hidden_layer, Interaction=False,\
                              Algorithm = 'MetropolisHastings',Number_MC_cycles = Number_MC_cycles)
    Energies, a, b, w, Iteration_number, Time_CPU = algorithm_2.gradient_descent(Maximum_iterations=5)
    print(
    f'''==============================================================================
    Energy: {Energies}  
    ==============================================================================
    Optimization values (a): {a}
    ==============================================================================
    Optimization values (b): {b}
    ==============================================================================
    Optimization values (w): {w}
    ==============================================================================
    Time CPU: {Time_CPU} [seconds] | {Time_CPU/60.0} [minutes]
    ==============================================================================
    ''')