function jointPositions = getJointPositions(l1,l2,l3,l4,l5,q1,q2,q3,q4,q5)
%GETJOINTPOSITIONS
%    JOINTPOSITIONS = GETJOINTPOSITIONS(L1,L2,L3,L4,L5,Q1,Q2,Q3,Q4,Q5)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    21-Jun-2021 14:24:54

t2 = pi./2.0;
t3 = q1+t2;
t4 = q2+t2;
t5 = q3+t2;
t6 = q4+t2;
t7 = q5+t2;
t8 = cos(t3);
t9 = cos(t4);
t10 = cos(t6);
t11 = sin(t3);
t12 = sin(t4);
t13 = sin(t6);
t14 = l1.*t8;
t15 = l2.*t9;
t16 = l4.*t10;
t17 = l1.*t11;
t18 = l2.*t12;
t19 = l4.*t13;
t20 = -t16;
t21 = -t19;
jointPositions = [t14;t17;t14+t15;t17+t18;t14+t15+l3.*cos(t5);t17+t18+l3.*sin(t5);t14+t15+t20;t17+t18+t21;t14+t15+t20-l5.*cos(t7);t17+t18+t21-l5.*sin(t7)];
