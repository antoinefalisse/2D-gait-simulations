function relativeJointAngularVelocities = getRelativeJointAngularVelocities(dq1,dq2,dq3,dq4,dq5)
%GETRELATIVEJOINTANGULARVELOCITIES
%    RELATIVEJOINTANGULARVELOCITIES = GETRELATIVEJOINTANGULARVELOCITIES(DQ1,DQ2,DQ3,DQ4,DQ5)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    14-Jul-2021 13:11:55

t2 = -dq3;
relativeJointAngularVelocities = [dq1;dq1-dq2;dq2+t2;dq4+t2;-dq4+dq5];
