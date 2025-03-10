# Packes
import numpy as np
from numpy.random import uniform

#import jax
#import jax.numpy as jnp

#jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_platform_name", "cpu")

from RBM.utils.variable_type import Type


# Quantum state
'''
===============================================================================
Class quantum state or just a trial object wave function as 'function' 
of the  particles position and varational parameter a, b and w.

consideraction
- Positions always has to be a matrix [Mumber particles x Dimension]
===============================================================================
'''
class State:

    def __init__(self, Number_particles: int , Dimension :int, Number_hidden_layer:int) -> None:
        self.Number_particles = Number_particles
        self.Dimension = Dimension
        self.Number_hidden_layer = Number_hidden_layer
        self.Sigma = 1.0

    '''
    ===============================================================================
    ----------
    Q fector : 
    ----------
    
    It is a function of position (r), b and w parameters.
    
    This is needed in the wave function definition
    
    r: Position 

    b : Hidden layer

    w : Weight tensor that related the visible and hidden layer 
    ===============================================================================
    '''

    # Factor in the exponent of hidden leyer (q_factor)
    def q_factor(self,
            r: Type.matrix,
            b: Type.vector,
            w:Type.matrix)-> Type.vector:
        
        # Standar desviation (Sigma = 1.0)
        Sigma = self.Sigma
        
        # Number of hidden layer
        Number_hidden_layer = self.Number_hidden_layer

        # Save infomation
        Q_fac = np.zeros((Number_hidden_layer), np.double)
        term_ex = np.zeros((Number_hidden_layer), np.double)

        # Term in the exponent 
        for ih in range(Number_hidden_layer):
           term_ex[ih] = (r*w[:,:,ih]).sum()

        Q_fac = b + (term_ex/Sigma**2)

        return Q_fac
    
    '''
    ===============================================================================
    ---------------
    Wave function :
    ---------------

    It is a function of position (r), a , b and w parameters
    
    r: Position 

    a : Visible layer

    b : Hidden layer

    w : Weight tensor that related the visible and hidden layer 
    ===============================================================================
    '''
    # Trial wave function for the 2-electron quantum dot in two dims
    def wave_function(self,
            r: Type.matrix,
            a: Type.matrix,
            b: Type.vector,
            w: Type.tensor)-> float:
        
        # Standar desviation (Sigma = 1.0)
        Sigma = self.Sigma

        Psi_1 = 0.0
        Psi_2 = 1.0

        # Number of particles 
        Number_particles = self.Number_particles

        # Dimention
        Dimension = self.Dimension

        # Number of hidden layer
        Number_hidden_layer = self.Number_hidden_layer

        # Factor in the exponent of hidden leyer (q_factor)
        Q_fac = self.q_factor(r,b,w)

        # Sum over all visible layer
        for ip in range(Number_particles):
            for id in range(Dimension):
                Psi_1 += (r[ip,id] - a[ip,id])**2

        # Sum over all hidden layer
        for ih in range(Number_hidden_layer):
            
            # Factor inside the squart root
            Psi_2 *= (1.0 + np.exp(Q_fac[ih]))
        
        # Factor out the squart root
        Psi_1 = np.exp(-Psi_1/(2*Sigma**2))
    
        return float(Psi_1*Psi_2)

if __name__ == "__main__":

    print('===========================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('===========================================')

    Number_particles = 2
    Dimension = 2
    Number_hidden_layer = 50

    r = uniform(-0.5,  0.5, size=(Number_particles,Dimension))
    a = uniform(0.0, 0.001, size=(Number_particles,Dimension))
    b = uniform(0.0, 0.001, size= (Number_hidden_layer))
    w = uniform(0.0, 0.001, size=(Number_particles, Dimension, Number_hidden_layer))

    print('================== Q factor ==================')
    print(State(Number_particles,Dimension,Number_hidden_layer).q_factor(r,b,w))
    print('==============================================')
    print('================ wave funtion ================')
    print(State(Number_particles,Dimension,Number_hidden_layer).wave_function(r,a,b,w))
    print('==============================================')


    