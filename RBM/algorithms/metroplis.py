
# Packes
from random import random, seed
from time import time
from math import exp, sqrt
import numpy as np
from numpy.random import uniform

from RBM.hamiltonian.local_energy import LocalEnergy
from RBM.utils.variable_type import Type

'''
===============================================================================
Class Metropolis algorithm

To instance this class always has to be necessary pass a NUmber particles,
dimencion and the hidden layer.

for : 
    Interaction : bool 
    Number_MC_cycles : int
    Step_size_jumping : float

can be instance when calls a class but it is not necesary becuase you will be
able to configure when call the method metropolis_algorithm
===============================================================================
'''

class Metropolis(LocalEnergy):

    def __init__(self, Number_particles: int, Dimension: int, Number_hidden_layer: int, Interaction=False,Sigma = 1.0,\
                 Number_MC_cycles = None,Step_size_jumping = None) -> None:
        super().__init__(Number_particles, Dimension, Number_hidden_layer, Interaction, Sigma) 
        self.Number_MC_cycles = Number_MC_cycles
        self.Step_size_jumping = Step_size_jumping
    '''
    ===============================================================================
    ----------------------
    Metropolis algorithm :
    ----------------------

    It is a function 

    a : Visible layer

    b : Hidden layer

    w : Weight tensor that related the visible and hidden layer 

    Number_MC_cycles : Number Monte Carlos cycles

    Step_size_jumping : Step size jumping the random walker 
    ===============================================================================
    '''
    
    def metropolis_algorithm(self, 
                            a: Type.matrix,
                            b: Type.vector,
                            w: Type.tensor,                                             
                            Number_MC_cycles : int = None,                 # NUmber of Monte Carlos cycles
                            Step_size_jumping : float = 1.0,               # Step size jumping of the randon number(walker)
                            )-> tuple:
        # Starting time consuming
        Time_inicio = time()

        '''
        Setting the number particles and dimension and verify if you instance directly
        the configuration of the parameter for metropolis_algorithm method
        '''

        if  self.Number_MC_cycles != None and Number_MC_cycles == None :
                Number_MC_cycles = self.Number_MC_cycles

        if self.Step_size_jumping != None and Step_size_jumping == None :
                Step_size_jumping = self.Step_size_jumping

        if Number_MC_cycles == None or Step_size_jumping == None :
                return ('< ERROR > : You have to set up Number_MC_cycles or step paramter')


        # Number of particles 
        Number_particles = self.Number_particles

        # Dimention
        Dimension = self.Dimension

        # Wave funtion 
        wave_function = super().wave_function
        
        # Local energy 
        local_energy = super().local_energy

        # Sava infomations  
        energy = 0.0  
        energy2 = 0.0 

        # Positions 
        Position_old = np.zeros((Number_particles,Dimension), np.double)
        Position_new = np.zeros((Number_particles,Dimension), np.double)

        # seed starts random numbers  
        seed()

        # Initial position
        Position_old = Step_size_jumping * uniform(low=-0.5, high=0.5, size=(Number_particles,Dimension))
        wfold = wave_function(Position_old,a,b,w)
        
        # Loop over Monte Carlos cicles (MCcycles)
        for MCcycle in range(Number_MC_cycles):
            
            # Trial position moving one particle at the time
            for ip in range(Number_particles):
                Position_new[ip,:] = Position_old[ip,:] + Step_size_jumping * uniform(low=-0.5, high=0.5, size=(Dimension))
                wfnew = wave_function(Position_new,a,b,w)
                
                # Metropolis test to see whether we accept 
                if random() < wfnew**2 / wfold**2:
                    Position_old[ip,:] = Position_new[ip,:]
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

    Number_particles = 2
    Dimension = 1
    Number_hidden_layer = 1
    Number_MC_cycles = 10**4

    algorithm = Metropolis(Number_particles, Dimension, Number_hidden_layer, Interaction=True)

    a = uniform(0.0, 0.001, size=(Number_particles,Dimension))
    b = uniform(0.0, 0.001, size= (Number_hidden_layer))
    w = uniform(0.0, 0.001, size=(Number_particles, Dimension, Number_hidden_layer))

    print('================== Energy using Metropolis ==================')
    Energy, Variance, Error, Time_CPU = algorithm.metropolis_algorithm(a,b,w,Number_MC_cycles)
    print(f'Energy: {Energy} | Variance: {Variance} | Error: {Error} | Time CPU: {Time_CPU}')
    print('=============================================================')