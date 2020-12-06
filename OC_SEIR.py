"""
Solves the Optimal Control of a SEIR (or SEIRN) system of ODEs using Forward - Backward Sweep

Forward - Backward Sweep from [Lenhart and Workman,"Optimal Control Applied to Biological Models"] 


Forward - Backward Sweep: a method used to solve an optimal control problem numerically. 
It is an algorithm that will generate an approximation to the optimality system.


Outline of the F-B Sweep:
Step 1: Make an initial guess for u over the interval.

Step 2: Using the initial condition x0 = x(0) and the values for u, solve x forward in time 
according to its differential equation in the optimality system.

Step 3: Using the transversality condition lambda_{N+1} = lambda(T) = 0 and the values for u
and x, solve lambda backward in time according to its differential equation in the optimality system.

Step 4:Update u by entering the new x and lambda  values into the characterization of the optimal control.
Average this control with the old control from the previous iteration.

Step 5: Check convergence. If values of the variables in this iteration and the last iteration are negligibly close, 
output the current values as solutions. If values are not close, return to Step 2.


SEIRN from [Joshi, Lenhart, Li, and Wang, "Optimal Control Methods Applied to Disease Models"]

"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#Test represents the convergence test variable
test = -1

# Parameters
b = 0.525
d = 0.5
c = 0.001
e = 0.5
g = 0.1
a = 0.2
A = 0.1
T = 20

#Initial Conditions
S0 = 1000
E0 = 2000
I0 = 5000
R0 = 1000
N0 = 9000

delta = 0.001   # Accepted Tolerance
M = 1000

tvec = np.linspace(0,T,M+1) # Time variable
h = 1/M    # Step size for Runge-Kutta 4 sweep
h2 = h/2


u = np.zeros(M+1)     # Initial guess for Control
x = np.zeros((M+1,4)) 

#x = np.zeros((M+1,4))     # State
S = np.zeros(M+1)     # State S
E = np.zeros(M+1)     # State E
I = np.zeros(M+1)     # State I
N = np.zeros(M+1)     # State N

#Lambda = np.zeros((M+1,4)) # Adjoint
Lambda1 = np.zeros(M+1) # Adjoint of  S
Lambda2 = np.zeros(M+1) # Adjoint of E
Lambda3 = np.zeros(M+1) # Adjoint of I
Lambda4 = np.zeros(M+1) # Adjoint of N
Lambda = np.zeros((M+1,4)) 

R = np.zeros(M+1)

S[0] = S0     # Stores initial value of State S.
E[0] = E0     # Stores initial value of State E.
I[0] = I0     # Stores initial value of State I.
N[0] = N0     # Stores initial value of State N.



#The while loop begins with the test function. This loop contains the Forward-Backward Sweep. The loop ends once converegence occurs (t>=0).
while(test < 0):
    
    #Stores the current values of u, S, E, I, N, Lambda1, Lambda2, Lambda3, Lambda4, x, and Lambda as the previous values of u, x, and lambda.
    oldu = u
    oldx = x
    #oldS = S
    #oldE = E
    #oldI = I
    #oldN = N
    #oldLambda1 = Lambda1
    #oldLambda2 = Lambda2
    #oldLambda3 = Lambda3
    #oldLambda4 = Lambda4
    oldLambda = Lambda

    # Generate new values. 

    #Runge-Kutta 4 sweep: Solving x forward in time. x1 is used  to find x2, x2 is used to solve x3 and so on.
    for i in np.arange (0,M):
        k11 = b*N[i] - (d + c*I[i] + u[i])*S[i]
        k12 = c*S[i]*I[i] - (e + d)*E[i]
        k13 = e*E[i] - (g + a + d)*I[i]
        k14 = (b - d)*N[i] - a*I[i]

        k21 = b*(N[i] + h2*k14) - (d + c*(I[i] + h2*k13) + 0.5*(u[i] + u[i+1]))*(S[i] + h2*k11)
        k22 = c*(S[i] + h2*k11)*(I[i] + h2*k13) - (e + d)*(E[i] + h2*k12)
        k23 = e*(E[i] + h2*k12) - (g + a + d)*(I[i] + h2*k13)
        k24 = (b - d)*(N[i] +h2*k14) - a*(I[i] + h2*k13)

        k31 = b*(N[i] + h2*k24) - (d + c*(I[i] + h2*k23) + 0.5*(u[i] + u[i+1]))*(S[i] + h2*k21)
        k32 = c*(S[i] + h2*k21)*(I[i] + h2*k23) - (e + d)*(E[i] + h2*k22)
        k33 = e*(E[i] + h2*k22) - (g + a + d)*(I[i] + h2*k23)
        k34 = (b - d)*(N[i] +h2*k24) - a*(I[i] + h2*k23)

        k41 = b*(N[i] + h*k34) - (d + c*(I[i] + h*k33) + u[i+1])*(S[i] + h*k31)
        k42 = c*(S[i] + h*k31)*(I[i] + h*k33) - (e + d)*(E[i] + h*k32)
        k43 = e*(E[i] + h*k32) - (g + a + d)*(I[i] + h*k33)
        k44 = (b - d)*(N[i] +h*k34) - a*(I[i] + h*k33)

        S[i+1] = S[i] + (h/6)*(k11 + 2*k21 +2*k31 +k41) 
        E[i+1] = E[i] + (h/6)*(k12 + 2*k22 +2*k32 +k42) 
        I[i+1] = I[i] + (h/6)*(k13 + 2*k23 +2*k33 +k43) 
        N[i+1] = N[i] + (h/6)*(k14 + 2*k24 +2*k34 +k44)   
    
    #Runge-Kutta 4 sweep: Solving Lambda backwards in time. Lambda1 is used  to find Lambda2, Lambda2 is used to solve Lambda3 and so on.
    for i in np.arange (0,M):
        j = M + 2 - (i+2)
        k11 = (d + c + u[j])*Lambda1[j] - c*Lambda2[j]*I[j]
        k12 = (e + d)*Lambda2[j] - e*Lambda3[j]
        k13 = -A + (g + a + d)*Lambda3[j] + c*Lambda1[j]*S[j] - c*Lambda2[j]*S[j] + a*Lambda4[j]
        k14 = -(b - d)*Lambda4[j] - b*Lambda1[j]

        k21 = (d + c + 0.5*(u[j] + u[j-1]))*(Lambda1[j] - h2*k11) - c*(Lambda2[j] - h2*k12)*0.5*(I[j] + I[j-1])
        k22 = (e + d)*(Lambda2[j] - h2*k12) - e*(Lambda3[j] - h2*k13)
        k23 = -A + (g + a + d)*(Lambda3[j] - h2*k13) + c*(Lambda1[j] - h2*k11)*0.5*(S[j] + S[j-1]) - c*(Lambda2[j] - h2*k12)*0.5*(S[j] + S[j-1]) + a*(Lambda4[j] - h2*k14)
        k24 = -(b - d)*(Lambda4[j] - h2*k14) - b*(Lambda1[j] - h2*k11)

        k31 = (d + c + 0.5*(u[j] + u[j-1]))*(Lambda1[j] - h2*k21) - c*(Lambda2[j] - h2*k22)*0.5*(I[j] + I[j-1])
        k32 = (e + d)*(Lambda2[j] - h2*k22) - e*(Lambda3[j] - h2*k23)
        k33 = -A + (g + a + d)*(Lambda3[j] - h2*k23) + c*(Lambda1[j] - h2*k21)*0.5*(S[j] + S[j-1]) - c*(Lambda2[j] - h2*k22)*0.5*(S[j] + S[j-1]) + a*(Lambda4[j] - h2*k24)
        k34 = -(b - d)*(Lambda4[j] - h2*k24) - b*(Lambda1[j] - h2*k21)

        k41 = (d + c + u[j-1])*(Lambda1[j] - h*k31) - c*(Lambda2[j] - h*k32)* I[j-1]
        k42 = (e + d)*(Lambda2[j] - h*k32) - e*(Lambda3[j] - h*k33)
        k43 = -A + (g + a + d)*(Lambda3[j] - h*k33) + c*(Lambda1[j] - h*k31)*S[j-1] - c*(Lambda2[j] - h*k32)*S[j-1] + a*(Lambda4[j] - h*k34)
        k44 = -(b - d)*(Lambda4[j] - h*k34) - b*(Lambda1[j] - h*k31)

        Lambda1[j-1] = Lambda1[j] - (h/6)*(k11 + 2*k21 + 2*k31 +k41)
        Lambda2[j-1] = Lambda2[j] - (h/6)*(k12 + 2*k22 + 2*k32 +k42)
        Lambda3[j-1] = Lambda3[j] - (h/6)*(k13 + 2*k23 + 2*k33 +k43)
        Lambda4[j-1] = Lambda4[j] - (h/6)*(k14 + 2*k24 + 2*k34 +k44)
    
    # Make SEIN and Lambda1-4 into transpose matrices
    SEIN = np.array([S,E,I,N])
    x = SEIN.transpose()
    Lam1_4 = np.array([Lambda1,Lambda2,Lambda3,Lambda4])
    Lambda = Lam1_4.transpose()

    #oldx = x
    #oldLambda = Lambda
    y = x[:,0]
    z = Lambda[:,0]

    # Optimal Control
    temp = (y*z)/2       #Represents the characterization of u 


    m = np.zeros(M+1)                 #Represents u and it's bounds i.e 0<= u* <= 0.9
    for i in np.arange (0,M+1):
          m[i] = min(0.9, max(0,temp[i]))
          u1 =m       


          u = 0.5*(u1 + oldu)     # Control 
    
    #Convergence test parameters of each variable.
    temp1 = delta*LA.norm(u) - LA.norm(oldu - u)
    temp2 = delta*LA.norm(x) - LA.norm(oldx - x)
    temp3 = delta*LA.norm(Lambda) - LA.norm(oldLambda - Lambda)
    

    test = min(temp1, min(temp2, temp3))     
    


#Solving for R where R = N - S - E - I
R = x[:,3] - x[:,1] - x[:,2] - x[:,0]

##Plot Results##
plt.figure(figsize=(9,9)) 

#plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)
#This is used to fix the spacing of the subplots
plt.subplots_adjust(None,None,None,None,0.5,0.5)    

plt.subplot(3,2,1);plt.plot(tvec,x[:,0])
plt.subplot(3,2,1);plt.xlabel('Time')
plt.subplot(3,2,1);plt.ylabel('S')
plt.subplot(3,2,1);plt.xlim([0,20])


plt.subplot(3,2,2);plt.plot(tvec,x[:,1])
plt.subplot(3,2,2);plt.xlabel('Time')
plt.subplot(3,2,2);plt.ylabel('E')
plt.subplot(3,2,2);plt.xlim([0,20])


plt.subplot(3,2,3);plt.plot(tvec,x[:,2])
plt.subplot(3,2,3);plt.xlabel('Time')
plt.subplot(3,2,3);plt.ylabel('I')   
plt.subplot(3,2,3);plt.xlim([0,20])


plt.subplot(3,2,4);plt.plot(tvec,R)
plt.subplot(3,2,4);plt.xlabel('Time')
plt.subplot(3,2,4);plt.ylabel('R')
plt.subplot(3,2,4);plt.xlim([0,20])


plt.subplot(3,2,5);plt.plot(tvec,x[:,3])
plt.subplot(3,2,5);plt.xlabel('Time')
plt.subplot(3,2,5);plt.ylabel('N')
plt.subplot(3,2,5);plt.xlim([0,20])


plt.subplot(3,2,6);plt.plot(tvec,u)
plt.subplot(3,2,6);plt.xlabel('Time')
plt.subplot(3,2,6);plt.ylabel('u')   
plt.subplot(3,2,6);plt.xlim([0,20])


plt.show()