# RBM

Currently, restricted Boltzmann machines (RBMs) are a useful tool in the field of machine learning, offering robust capabilities for unsupervised learning tasks. RBMs are of vital importance, since they allow us to study the quantum many-body problem.

In many-body physics, BMs have become a powerful tool for modelling and understanding complex quantum systems. These neural networks are particularly adept at capturing the intricate correlations and interactions among a large number of particles. By leveraging their ability to represent complex probability distributions, BMs can efficiently approximate the ground state wave functions of quantum systems, even in high-dimensional spaces . This is particularly useful in studying phenomena like quantum phase transitions, where traditional methods may struggle due to the exponential growth of the state space with the number of particles. Additionally, BMs can be employed to model thermal states, providing insights into the thermodynamic properties  of many-body systems. Their flexibility and adaptability make them valuable in simulating and  exploring a wide range of physical systems, from strongly correlated electron systems to lattice models in condensed matter physics.

1. **Solving the quantum many-body problem with artificial neural networks** by Giuseppe Carleo and Matthias Troyer. *Science, 355(6325)*, 602-606, 2017. [DOI Link](https://doi.org/10.1126/science.aag2302).

2. **Two electrons in an external oscillator potential: Particular analytic solutions of a Coulomb correlation problem** by M. Taut. *Physical Review A, 48(5)*, 3561-3566, 1993. [DOI Link](https://link.aps.org/doi/10.1103/PhysRevA.48.3561).

3. **Restricted Boltzmann Machines: Introduction and Review** by Guido Montúfar. *ArXiv, abs/1806.07066*, 2018. [Link to Paper](https://api.semanticscholar.org/CorpusID:49309345).

## Code implementation

In the code implementation, we use some approximation to construct the fermions wave function, we drop the antisymmetric properties, that means the result work very well for two electrons, if we want extended for more than two fermions we have to make some modification. On the other hand, for the bosons the perfect can be extended until the computation tool allows it. 