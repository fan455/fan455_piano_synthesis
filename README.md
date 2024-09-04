# fan455_piano_synthesis
This is an open-source piano sound synthesis project using physical modelling. It currently aims to simulate: (1) linearized strings with vertical, horizontal, longitudinal, rotational vibration; (2) non-linear hammer-string interaction; (3) fully 3D soundboard; (4) coupling between strings and soundboard through the bridge; (5) linearized acoustic wave propagation. 

The core codes are written in Rust, and some pre-processing and post-analysis parts are written in Python. The main external dependencies of the codes are: (1) GMSH (both GUI and SDK version) for mesh generation and processing; (2) Intel MKL for solving generalized eigenvalue problems of sparse matrices in the system of vibration equations, using the FEAST algorithm. 

The repository include 1D and 3D finite element models customized for piano modelling. The finite element approaches used here are maybe more flexible than general-purpose finite element softwares. (1) Different variables may use different element orders to reduce computation costs and also alleviate shear locking. (2) Dirichlet boundary conditions are implemented by excluding relevant DOFs on the boundary. 

The project is in active developing status, so the codes are probably un-runnable for you right now. I will update the documentation when the project is complete enough.
