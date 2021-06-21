'''
This functions returns a CasADi function, which can be called when
solving the NLP to compute the dynamic constraint errors.
'''

import casadi
from getSystemDynamics import getSystemDynamics

def getDynamicConstraintErrors(
    m1, m2, m3, m4, m5, 
    I1, I2, I3, I4, I5, 
    d1, d2, d3, d4, d5, 
    l1, l2, l3, l4, l5, 
    g):

    lc1 = l1 - d1
    lc5 = l5 - d5
    lc2 = l2 - d2
    lc4 = l4 - d4
    lc3 = d3

    # CasADi variables.
    # Joint coordinate values.
    q1_MX = casadi.MX.sym('q1_MX',1)
    q2_MX = casadi.MX.sym('q2_MX',1)      
    q3_MX = casadi.MX.sym('q3_MX',1)
    q4_MX = casadi.MX.sym('q4_MX',1)
    q5_MX = casadi.MX.sym('q5_MX',1)
    # Joint coordinate speeds.
    dq1_MX = casadi.MX.sym('dq1_MX',1)
    dq2_MX = casadi.MX.sym('dq2_MX',1) 
    dq3_MX = casadi.MX.sym('dq3_MX',1)
    dq4_MX = casadi.MX.sym('dq4_MX',1)
    dq5_MX = casadi.MX.sym('dq5_MX',1)
    # Joint coordinate speed derivatives (accelerations).
    ddq1_MX = casadi.MX.sym('ddq1_MX',1)
    ddq2_MX = casadi.MX.sym('ddq2_MX',1)  
    ddq3_MX = casadi.MX.sym('ddq3_MX',1)
    ddq4_MX = casadi.MX.sym('ddq4_MX',1)
    ddq5_MX = casadi.MX.sym('ddq5_MX',1)
    # Joint torques.
    T1_MX = casadi.MX.sym('T1_MX',1)
    T2_MX = casadi.MX.sym('T2_MX',1)     
    T3_MX = casadi.MX.sym('T3_MX',1)
    T4_MX = casadi.MX.sym('T4_MX',1)
    T5_MX = casadi.MX.sym('T5_MX',1)
    
    # The equations of motion were descibed symbolically.
    constraintErrors = getSystemDynamics(
        I1,I2,I3,I4,I5,
        T1_MX,T2_MX,T3_MX,T4_MX,T5_MX,
        ddq1_MX,ddq2_MX,ddq3_MX,ddq4_MX,ddq5_MX,
        dq1_MX,dq2_MX,dq3_MX,dq4_MX,dq5_MX,
        g,
        l1,l2,l4,
        lc1,lc2,lc3,lc4,lc5,
        m1,m2,m3,m4,m5,
        q1_MX,q2_MX,q3_MX,q4_MX,q5_MX)
    
    # CasADi function describing implicitly the dynamic constraint errors.
    # f(T, ddq, dq, q) == 0.
    f_getDynamicConstraintErrors = casadi.Function(
        'f_getDynamicConstraintErrors',[
        T1_MX,T2_MX,T3_MX,T4_MX,T5_MX,
        ddq1_MX,ddq2_MX,ddq3_MX,ddq4_MX,ddq5_MX,
        dq1_MX,dq2_MX,dq3_MX,dq4_MX,dq5_MX,
        q1_MX,q2_MX,q3_MX,q4_MX,q5_MX],constraintErrors)
    
    return f_getDynamicConstraintErrors
