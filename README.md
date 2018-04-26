# Points On A Sphere
## *Python code used to collect the data given in my report and to produce graphs and images*.


**DiscEnergyFit.py**: 
Fits a function to the set of data collected by my gradient flow and gentic algorithm for points in a disc. 
Creates a plot of the function against the data and a plot of the differences between data and fitted data.


**EnergyFitSphere.py**:
Fits a function to the set of data collected by my gradient flow and gentic algorithm for points on the surface of a sphere. 
Creates a plot of the function against the data and a plot of the differences between data and fitted data.

**GeneticAlgorithm.py**:
Applies the genetic algorithm to n points on the sphere. 

**GradientFlow.py**:
Applies the gradient flow method to n points constrained to  the surface of a unit sphere.

**RelaxBallCharges.py**:
Applies the gradient flow method to points in a ball. Can also choose to make one point have a charge greater than 1.

**RelaxBallGA.py**:
Applies the genetic algorithm to points in a ball, using the gradient flow method to relax the configurations.

**RelaxDisc.py**:
Applies the gradient flow method to points constrained to a unit disc. Can run the program using any of the three different
distribution methods for the starting points detailed in the report.

**RelaxDiscGA.py**:
Applies the genetic algorithm to points constrained to a unit disc, using the gradient flow method to relax the configurations.

**SpherePacking.py**: 
Given a set of points, plots the points with spheres of radius r about each point where r is half the minimum distance between 
any two points. Also can plot the points as spheres when one point has a larger charge than all others. This point is drawn 
with a sphere of radius k where k is the half the smallest distance between this point and the point closest to it.

**Symmetry2.1.py**:
Given a configuration of points on the sphere will output whether the configuration has platonic symmetry, at least C_3 symmetry, 
or at most C_2 symmetry. In the case that at least C_3 symmetry is found the program then searches for which C_k symmetry the shape has,
whether the shape has a vertical or horizontal reflective plane and also if the shape has dihedral symmetry.

**VoronoiPlot.py**:  
Creates a Voronoi plot of a given configuration of points on the surface of a sphere.

**VPDisc.py**:
Produces a Voronoi plot of a given configuration of points in a disc.

**fullmonte**:
Program written by Paul Sutcliffe with changes made by myself. I rewrote the code so that it is split into functions instead of one loop.
Also changed the stopping criteria of the program.

