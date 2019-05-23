# Model analysis project

Our project is titled **Solving Stochastic Differential Equations using Monte Carlo Simulation**. In this project we show how the Monte Carlo methods can be applied to solve stochastic differential equations. To do this we consider the stochastic differential equation given by the CIR model since the CIR model has an analytical solution, thus we can compare the numerical estimates to the actual solution, thereby examining the accuracy and convergence rate of the Monte Carlo simulations. First, we implement the simple Euler discretization scheme. However, the Euler scheme leads to discretization error and therefore we extend the analysis and consider the Milstein discretization scheme to improve upon the speed of convergence and accuracy of the Monte Carlo Methods. 

The **results** of the project can be seen from running [modelproject.ipynb](modelproject.ipynb).

**Dependencies:** We suspect computers with low computational power can encounter long running times since we are simulating a lot of sample paths. It took 2 minutes and 20 seconds to run the full program on a fairly decent computer. Apart from that, only the standard Anaconda Python 3 installation is required.
