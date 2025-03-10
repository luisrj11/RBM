# Packages
import numpy as np
from numpy.random import randn

from RBM.utils.variable_type import Type
from RBM.quantum_state.states import State

# Local energy
'''
===============================================================================
Define the Local energy class this needs a State class

consideraction
- Positions always has to be a matrid [Number particles x Dimention]
- Interaction : It is bool variable  
===============================================================================
'''
class LocalEnergy(State):

        def __init__(self, Number_particles: int, Dimension: int, Number_hidden_layer: int,Interaction = False, Sigma = 1.0) -> None:
            super().__init__(Number_particles, Dimension, Number_hidden_layer)
            self.Interaction = Interaction
            self.Sigma = Sigma 

        '''
        ===============================================================================
        ----------------------
        Anlytic local energy :
        ----------------------

        It is a function of position (r), a , b and w parameters
    
        r: Positio  

        a : Visible layer

        b : Hidden layer

        w : Weight tensor that related the visible and hidden layer 
        ===============================================================================
        '''
        # Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
        def local_energy(self,
            r: Type.matrix,
            a: Type.matrix,
            b: Type.vector,
            w: Type.tensor)-> float:
            
            # Standar desviation (Sigma = 1.0)
            sigma = self.Sigma

            # Sigma squared
            sigma2 = sigma**2

            # Number of particles 
            Number_particles = self.Number_particles

            # Dimention
            Dimension = self.Dimension

            # Number of hidden layer
            Number_hidden_layer = self.Number_hidden_layer

            Locenergy = 0.0
            
            # Factor in the exponent of hidden leyer (q_factor)
            Q_fac = self.q_factor(r,b,w)

            for ip in range(Number_particles):
                for id in range(Dimension):
                    sum1 = 0.0
                    sum2 = 0.0
                    for ih in range(Number_hidden_layer):
                        sum1 += w[ip,id,ih]/(1 + np.exp(-Q_fac[ih]))
                        sum2 += w[ip,id,ih]**2 * np.exp(Q_fac[ih]) / (1.0 + np.exp(Q_fac[ih]))**2

                    d_ln_psi_1 = -(r[ip,id] - a[ip,id]) /sigma2 + sum1/sigma2
                    d_ln_psi_2 = -1/sigma2 + sum2/sigma2**2

                    Locenergy += 0.5*(-d_ln_psi_1*d_ln_psi_1 - d_ln_psi_2 + r[ip,id]**2)
            
            # Turn on the interaction 
            if type(self.Interaction) == bool:
                if (self.Interaction == True):
                    if Number_particles > 1 : 
                        for ip in range(Number_particles):
                            for jp in range(ip):
                                distance = 0.0
                                for id in range(Dimension):
                                    distance += (r[ip,id] - r[jp,id])**2
                                Locenergy += 1/np.sqrt(distance)
                    else:
                        print('< Warning > : There is just one particle, it can not interact with itself')
                        print('===> HINT: Set up the Interaction = False')
            else:
                return (f'< ERROR > : The interaction has to be a bool variable')
          
            
            return float(Locenergy)

if __name__ == "__main__":

    print('==================================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('==================================================')

    Number_particles = 6
    Dimension = 1
    Number_hidden_layer = 2

    Local_energy = LocalEnergy(Number_particles,Dimension,Number_hidden_layer)
    Local_energy.Interaction = True

    r = randn(Number_particles,Dimension)
    a = randn(Number_particles,Dimension)
    b = randn(Number_hidden_layer)
    w = randn(Number_particles,Dimension,Number_hidden_layer)

    print('================== Local energy ==================')
    print(Local_energy.local_energy(r,a,b,w))
    print('==================================================')