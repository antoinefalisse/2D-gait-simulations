'''
This function returns the integration error, following a backward Euler
integration scheme. Constraints in the optimization problem impose this
error to be nul.

If x is the state value at time t, then z is the state value at time t+1.
The backward Euler equation can be formulated as:
u(t) = (x(t+1) - x(t))/dt, and therefore the error is given by:
error = (x(t+1) - x(t)) - u(t)dt.
'''

def eulerIntegrator(x,z,u,dt):

    return (z - x) - u*dt
