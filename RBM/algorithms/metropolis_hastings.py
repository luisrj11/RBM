# Packes
from random import random, seed, normalvariate
from time import time
from math import exp, sqrt
import numpy as np
from numpy.random import uniform

from RBM.hamiltonian.local_energy import LocalEnergy
from RBM.utils.variable_type import Type


'''
===============================================================================
Class Metropolis Hastings algorithm 

To instance this class always has to be necessary pass a number particles,
dimencion and number of hidden layer.

for : 
    Interaction : bool
    Number_MC_cycles : int
    Time_step : float 

Can be instance when calls a class but it is not necesary becuase you will be
able to configure when call the method metropolis_hastings_algorithm
===============================================================================
'''

class MetropolisHastings(LocalEnergy):

    def __init__(self, Number_particles: int, Dimension: int, Number_hidden_layer: int, Interaction = False, Sigma=1.0,\
                 Number_MC_cycles = None,Time_step = None ) -> None:
        super().__init__(Number_particles, Dimension, Number_hidden_layer, Interaction, Sigma)
        self.Number_MC_cycles = Number_MC_cycles
        self.Time_step = Time_step

    '''
    ===============================================================================
    ------------------------
    Quantum force analytic :
    ------------------------

    It is a function of position (r), a , b and w parameters
    
    r: Positio 

    a : Visible layer

    b : Hidden layer

    w : Weight tensor that related the visible and hidden layer 
    ===============================================================================
    '''

    # Drift force
    def quantum_force(self,
            r: Type.matrix,
            a: Type.matrix,
            b: Type.vector,
            w: Type.tensor
            )-> float:

        # Standar desviation (Sigma = 1.0)
        sigma = self.Sigma

        # Sigma squared
        sigma2 = sigma**2

        qforce = np.zeros((self.Number_particles,self.Dimension), np.double)
        sum1 = np.zeros((self.Number_particles,self.Dimension), np.double)

        # Factor in the exponent of hidden leyer (q_factor)
        Q_fac = self.q_factor(r,b,w)

        for ih in range(self.Number_hidden_layer):
            sum1 += w[:,:,ih]/(1+np.exp(-Q_fac[ih]))

        qforce = 2*(-(r-a)/sigma2 + sum1/sigma2)

        return qforce
    
    '''
    ===============================================================================
    -------------------------------
    Metropolis Hastings algorithm :
    -------------------------------
    It is a function 

    a : Visible layer

    b : Hidden layer

    w : Weight tensor that related the visible and hidden layer 

    Number_MC_cycles : Number Monte Carlos cycles

    Time_step : Time step size jumping the random walker 
    ===============================================================================
    '''
    
    def metropolis_hastings_algorithm(self,  
                             a: Type.matrix,
                             b: Type.vector,
                             w: Type.tensor,        
                             Number_MC_cycles : int = None,               # NUmber of Monte Carlos cycles
                             Time_step = 0.05,                          
                                )-> tuple:
        # Time CPU
        Time_inicio = time()

        '''
        Setting the number particles and dimension and verify if you instance directly
        the configuration of the parameter for metropolis_algorithm method
        '''

        if self.Number_MC_cycles != None and Number_MC_cycles == None :
             Number_MC_cycles = self.Number_MC_cycles

        if self.Time_step != None and Time_step == None :
            Time_step = self.Time_step
        
        if Number_MC_cycles == None or Time_step == None:
            return ('< ERROR > : You have to set up Number_MC_cycles or step paramter')

    
        # Number of particles 
        Number_particles = self.Number_particles

        # Dimention
        Dimension = self.Dimension
       
        # Wave funtion 
        wave_function = super().wave_function

        # Drift force
        quantum_force = self.quantum_force

        # Local energy 
        local_energy = super().local_energy

        # Parameters in the Fokker-Planck simulation of the quantum force
        Dif = 0.5

        # Sava infomations  
        energy = 0.0  
        energy2 = 0.0 

        # Positions 
        Position_old = np.zeros((Number_particles,Dimension), np.double)
        Position_new = np.zeros((Number_particles,Dimension), np.double)

        # Quantum force
        Quantum_force_old = np.zeros((Number_particles,Dimension), np.double)
        Quantum_force_new = np.zeros((Number_particles,Dimension), np.double)

        # seed starts random numbers  
        seed()

        # Initial position
        for i in range(Number_particles):
            for j in range(Dimension):
                Position_old[i,j] = normalvariate(0.0,1.0)*sqrt(Time_step)
        wfold = wave_function(Position_old,a,b,w)
        Quantum_force_old = quantum_force(Position_old,a,b,w)

        
        # Loop over Monte Carlos cicles (MCcycles)
        for MCcycle in range(Number_MC_cycles):
            
            # Trial position moving one particle at the time
            for i in range(Number_particles):
                for j in range(Dimension):
                    Position_new[i,j] = Position_old[i,j] + normalvariate(0.0,1.0)*sqrt(Time_step)+\
                                       Quantum_force_old[i,j]*Time_step*Dif
                wfnew = wave_function(Position_new,a,b,w)
                Quantum_force_new = quantum_force(Position_new,a,b,w)
                
                # Greens function
                GreensFunction = 0.0
                for j in range(Dimension):
                        GreensFunction += 0.5*(Quantum_force_old[i,j]+Quantum_force_new[i,j])*\
	                              (Dif*Time_step*0.5*(Quantum_force_old[i,j]-Quantum_force_new[i,j])-\
                                   Position_new[i,j]+Position_old[i,j])
                
                # Caclulate the Green's function
                GreensFunction = exp(GreensFunction)
                ProbabilityRatio = GreensFunction*(wfnew**2/wfold**2)

                # Metropolis-Hastings test to see whether we accept the move
                if random() <= ProbabilityRatio:
                    for j in range(Dimension):
                        Position_old[i,j] = Position_new[i,j]
                        Quantum_force_old[i,j] = Quantum_force_new[i,j]
                    wfold = wfnew
            
            # Calculate the local energy 
            DeltaE = local_energy(Position_old,a,b,w)
            energy += DeltaE
            energy2 += DeltaE**2
    
        # Calculate mean, variance and error  
        energy /= Number_MC_cycles
        energy2 /= Number_MC_cycles
        variance = energy2 - energy**2
        error = sqrt(variance/Number_MC_cycles)

        # Time CPU
        Time_fin = time()
        Time_consuming = Time_fin - Time_inicio
        
        return energy, variance, error, Time_consuming
    
if __name__ == "__main__":

    print('==================================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('==================================================')

    Number_particles = 5
    Dimension = 2
    Number_hidden_layer = 10
    Number_MC_cycles = 10**4
    
    # Number particle 
    Number_particles = 1

    # Dimension
    Dimension = 1

    # Number Monte Carlos cycles
    Number_MC_cycles = 10**4

    Number_hidden_layer = 10
    
    algorithm = MetropolisHastings(Number_particles, Dimension, Number_hidden_layer, Interaction=True)

    a = uniform(0.0, 0.001, size=(Number_particles,Dimension))
    b = uniform(0.0, 0.001, size= (Number_hidden_layer))
    w = uniform(0.0, 0.001, size=(Number_particles, Dimension, Number_hidden_layer))

    print('================== Energy using Metropolis Hastings ==================')
    Energy, Variance, Error, Time_CPU  = algorithm.metropolis_hastings_algorithm(a,b,w,Number_MC_cycles)
    print(f'Energy: {Energy} | Variance: {Variance} | Error: {Error} | Time CPU: {Time_CPU}')
    print('======================================================================')