# iSeeing
iSeeing (read 'icing') is a finite difference python solver for single-phase phase field equation.
The solver is based on the work of Kobayashi [1].

Laplacians use a 2nd order 9-points stencil, gradients a 2nd order 2-points stencil.
Domain assumes periodic boundary contidions.

Explicit time marching using first order Euler is done for both the phase field and temperature field.

iSeeing has plenty of scope for improvement in both performance and accuracy but is nonetheless a nice tool for educational purposes.

# Reference
[1] Kobayashi, Ryo. "Modeling and numerical simulations of dendritic crystal growth." Physica D: Nonlinear Phenomena 63.3-4 (1993): 410-423.
