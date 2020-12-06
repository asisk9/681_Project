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


def OC_SEIRN(b, d, c, e, g, a, A, T, S0, E0, I0, R0, N0):
 #Test represents the convergence test variable
 test = -1

 delta = 0.001   # Accepted Tolerance
 M = 1000

 t = np.linspace(0,T,M+1) # Time variable
 h = 1/M    # Step size for Runge-Kutta 4 sweep
 h2 = h/2


 u = np.zeros(M+1)     # Initial guess for Control
 x = np.zeros((M+1,4)) # State
 S = np.zeros(M+1)     # State S
 E = np.zeros(M+1)     # State E
 I = np.zeros(M+1)     # State I
 N = np.zeros(M+1)     # State N

 LambdaS = np.zeros(M+1) # Adjoint of S
 LambdaE = np.zeros(M+1) # Adjoint of E
 LambdaI = np.zeros(M+1) # Adjoint of I
 LambdaN = np.zeros(M+1) # Adjoint of N
 Lambda = np.zeros((M+1,4))  # Adjoint

 R = np.zeros(M+1)    # Recovery rate 

 S[0] = S0     # Stores initial value of State S.
 E[0] = E0     # Stores initial value of State E.
 I[0] = I0     # Stores initial value of State I.
 N[0] = N0     # Stores initial value of State N.



 #The while loop begins with the test function. This loop contains the Forward-Backward Sweep. The loop ends once converegence occurs (t>=0).
 while(test < 0):
    
     #Stores the current values of u, S, E, I, N, Lambda1, Lambda2, Lambda3, Lambda4, x, and Lambda as the previous values of u, x, and lambda.
     oldu = u
     oldx = x
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
    
     #Runge-Kutta 4 sweep: Solving Lambda backwards in time. LambdaS is used  to find LambdaE, LambdaE is used to solve LambdaI and so on.
     for i in np.arange (0,M):
         j = M + 2 - (i+2)
         k11 = (d + c + u[j])*LambdaS[j] - c*LambdaE[j]*I[j]
         k12 = (e + d)*LambdaE[j] - e*LambdaI[j]
         k13 = -A + (g + a + d)*LambdaI[j] + c*LambdaS[j]*S[j] - c*LambdaE[j]*S[j] + a*LambdaN[j]
         k14 = -(b - d)*LambdaN[j] - b*LambdaS[j]

         k21 = (d + c + 0.5*(u[j] + u[j-1]))*(LambdaS[j] - h2*k11) - c*(LambdaE[j] - h2*k12)*0.5*(I[j] + I[j-1])
         k22 = (e + d)*(LambdaE[j] - h2*k12) - e*(LambdaI[j] - h2*k13)
         k23 = -A + (g + a + d)*(LambdaI[j] - h2*k13) + c*(LambdaS[j] - h2*k11)*0.5*(S[j] + S[j-1]) - c*(LambdaE[j] - h2*k12)*0.5*(S[j] + S[j-1]) + a*(LambdaN[j] - h2*k14)
         k24 = -(b - d)*(LambdaN[j] - h2*k14) - b*(LambdaS[j] - h2*k11)

         k31 = (d + c + 0.5*(u[j] + u[j-1]))*(LambdaS[j] - h2*k21) - c*(LambdaE[j] - h2*k22)*0.5*(I[j] + I[j-1])
         k32 = (e + d)*(LambdaE[j] - h2*k22) - e*(LambdaI[j] - h2*k23)
         k33 = -A + (g + a + d)*(LambdaI[j] - h2*k23) + c*(LambdaS[j] - h2*k21)*0.5*(S[j] + S[j-1]) - c*(LambdaE[j] - h2*k22)*0.5*(S[j] + S[j-1]) + a*(LambdaN[j] - h2*k24)
         k34 = -(b - d)*(LambdaN[j] - h2*k24) - b*(LambdaS[j] - h2*k21)

         k41 = (d + c + u[j-1])*(LambdaS[j] - h*k31) - c*(LambdaE[j] - h*k32)* I[j-1]
         k42 = (e + d)*(LambdaE[j] - h*k32) - e*(LambdaI[j] - h*k33)
         k43 = -A + (g + a + d)*(LambdaI[j] - h*k33) + c*(LambdaS[j] - h*k31)*S[j-1] - c*(LambdaE[j] - h*k32)*S[j-1] + a*(LambdaN[j] - h*k34)
         k44 = -(b - d)*(LambdaN[j] - h*k34) - b*(LambdaS[j] - h*k31)

         LambdaS[j-1] = LambdaS[j] - (h/6)*(k11 + 2*k21 + 2*k31 +k41)
         LambdaE[j-1] = LambdaE[j] - (h/6)*(k12 + 2*k22 + 2*k32 +k42)
         LambdaI[j-1] = LambdaI[j] - (h/6)*(k13 + 2*k23 + 2*k33 +k43)
         LambdaN[j-1] = LambdaN[j] - (h/6)*(k14 + 2*k24 + 2*k34 +k44)
    
     # Make SEIN and Lambda1-4 into transpose matrices
     SEIN = np.array([S,E,I,N])
     x = SEIN.transpose()
     LamSEIN = np.array([LambdaS,LambdaE,LambdaI,LambdaN])
     Lambda = LamSEIN.transpose()

     # Optimal Control
     w = x[:,0]          #Stores the susceptible
     z = Lambda[:,0]     #Stores the Lambda of the susceptible

     temp = (w*z)/2       #Represents the characterization of u 

     m = np.zeros(M+1)                 
     for i in np.arange (0,M+1):       #Represents u and it's bounds i.e 0<= u* <= 0.9
           m[i] = min(0.9, max(0,temp[i]))
           u1 = m       

           u = 0.5*(u1 + oldu)     # Control 
    
     #Convergence test parameters of each variable.
     temp1 = delta*LA.norm(u) - LA.norm(oldu - u)
     temp2 = delta*LA.norm(x) - LA.norm(oldx - x)
     temp3 = delta*LA.norm(Lambda) - LA.norm(oldLambda - Lambda)
    

     test = min(temp1, min(temp2, temp3))     
    


 #Solving for R where R = N - S - E - I
 R = x[:,3] - x[:,1] - x[:,2] - x[:,0]

 y = [x[:,0], x[:,1], x[:,2], x[:,3], R, u, t]
 return y

y1 = OC_SEIRN(b = 0.525, d = 0.5, c = 0.0001, e = 0.5, g = 0.1, a = 0.2, A = 0.1, T = 20, S0 = 1000, E0 = 100, I0 = 50, R0 = 15, N0 = 1165)

y2 = OC_SEIRN(b = 0.525, d = 0.5, c = 0.001, e = 0.5, g = 0.1, a = 0.2, A = 0.1, T = 20, S0 = 1000, E0 = 100, I0 = 50, R0 = 15, N0 = 1165)

y3 = OC_SEIRN(b = 0.525, d = 0.5, c = 0.001, e = 0.5, g = 0.1, a = 0.2, A = 0.1, T = 20, S0 = 1000, E0 = 1000, I0 = 2000, R0 = 500, N0 = 4500)

y4 = OC_SEIRN(b = 0.525, d = 0.5, c = 0.001, e = 0.5, g = 0.1, a = 0.2, A = 0.1, T = 20, S0 = 1000, E0 = 2000, I0 = 5000, R0 = 1000, N0 = 9000)


##Plot Results##
#Change figure size
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

plt.figure(1)           #Plot of y1

#plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)
#This is used to fix the spacing of the subplots
plt.subplots_adjust(None,None,None,None,0.5,0.5)    

plt.subplot(3,2,1);plt.plot(y1[6], y1[0], 'r-.')
plt.subplot(3,2,1);plt.xlabel('Time')
plt.subplot(3,2,1);plt.ylabel('S')
plt.subplot(3,2,1);plt.xlim([0,20])
plt.subplot(3,2,1);plt.ylim(bottom=1000)

plt.subplot(3,2,2);plt.plot(y1[6], y1[1], 'r-.')
plt.subplot(3,2,2);plt.xlabel('Time')
plt.subplot(3,2,2);plt.ylabel('E')
plt.subplot(3,2,2);plt.xlim([0,20])
plt.subplot(3,2,2);plt.ylim(bottom=39)

plt.subplot(3,2,3);plt.plot(y1[6], y1[2], 'r-.')
plt.subplot(3,2,3);plt.xlabel('Time')
plt.subplot(3,2,3);plt.ylabel('I')   
plt.subplot(3,2,3);plt.xlim([0,20])
plt.subplot(3,2,3);plt.ylim(bottom=43.5)

plt.subplot(3,2,4);plt.plot(y1[6], y1[4], 'r-.')
plt.subplot(3,2,4);plt.xlabel('Time')
plt.subplot(3,2,4);plt.ylabel('R')
plt.subplot(3,2,4);plt.xlim([0,20])
plt.subplot(3,2,4);plt.ylim(bottom=15)

plt.subplot(3,2,5);plt.plot(y1[6], y1[3], 'r-.')
plt.subplot(3,2,5);plt.xlabel('Time')
plt.subplot(3,2,5);plt.ylabel('N')
plt.subplot(3,2,5);plt.xlim([0,20])
plt.subplot(3,2,5);plt.ylim(bottom=1165)

plt.subplot(3,2,6);plt.plot(y1[6], y1[5], 'r-.')
plt.subplot(3,2,6);plt.xlabel('Time')
plt.subplot(3,2,6);plt.ylabel('u')   
plt.subplot(3,2,6);plt.xlim([0,20])
plt.subplot(3,2,6);plt.ylim(bottom=0)
#---------------------------------------------------------------------------------------------------
plt.figure(2)       #Plot of y2
plt.subplots_adjust(None,None,None,None,0.5,0.5)    

plt.subplot(3,2,1);plt.plot(y2[6], y2[0],'g--')
plt.subplot(3,2,1);plt.xlabel('Time')
plt.subplot(3,2,1);plt.ylabel('S')
plt.subplot(3,2,1);plt.xlim([0,20])
plt.subplot(3,2,1);plt.ylim(bottom=990)

plt.subplot(3,2,2);plt.plot(y2[6], y2[1],'g--')
plt.subplot(3,2,2);plt.xlabel('Time')
plt.subplot(3,2,2);plt.ylabel('E')
plt.subplot(3,2,2);plt.xlim([0,20])
plt.subplot(3,2,2);plt.ylim(bottom=69)

plt.subplot(3,2,3);plt.plot(y2[6], y2[2],'g--')
plt.subplot(3,2,3);plt.xlabel('Time')
plt.subplot(3,2,3);plt.ylabel('I')   
plt.subplot(3,2,3);plt.xlim([0,20])
plt.subplot(3,2,3);plt.ylim(bottom=50)

plt.subplot(3,2,4);plt.plot(y2[6], y2[4],'g--')
plt.subplot(3,2,4);plt.xlabel('Time')
plt.subplot(3,2,4);plt.ylabel('R')
plt.subplot(3,2,4);plt.xlim([0,20])
plt.subplot(3,2,4);plt.ylim(bottom=15)

plt.subplot(3,2,5);plt.plot(y2[6], y2[3],'g--')
plt.subplot(3,2,5);plt.xlabel('Time')
plt.subplot(3,2,5);plt.ylabel('N')
plt.subplot(3,2,5);plt.xlim([0,20])
plt.subplot(3,2,5);plt.ylim(bottom=1165)

plt.subplot(3,2,6);plt.plot(y2[6], y2[5],'g--')
plt.subplot(3,2,6);plt.xlabel('Time')
plt.subplot(3,2,6);plt.ylabel('u')   
plt.subplot(3,2,6);plt.xlim([0,20])
plt.subplot(3,2,6);plt.ylim(bottom=0)
#---------------------------------------------------------------------------------------------------
plt.figure(3)          #Plot of y3
plt.subplots_adjust(None,None,None,None,0.5,0.5)    

plt.subplot(3,2,1);plt.plot(y3[6], y3[0], 'm_')
plt.subplot(3,2,1);plt.xlabel('Time')
plt.subplot(3,2,1);plt.ylabel('S')
plt.subplot(3,2,1);plt.xlim([0,20])
plt.subplot(3,2,1);plt.ylim(bottom=800)

plt.subplot(3,2,2);plt.plot(y3[6], y3[1], 'm_')
plt.subplot(3,2,2);plt.xlabel('Time')
plt.subplot(3,2,2);plt.ylabel('E')
plt.subplot(3,2,2);plt.xlim([0,20])
plt.subplot(3,2,2);plt.ylim(bottom=1000)

plt.subplot(3,2,3);plt.plot(y3[6], y3[2], 'm_')
plt.subplot(3,2,3);plt.xlabel('Time')
plt.subplot(3,2,3);plt.ylabel('I')   
plt.subplot(3,2,3);plt.xlim([0,20])
plt.subplot(3,2,3);plt.ylim(bottom=1307)

plt.subplot(3,2,4);plt.plot(y3[6], y3[4], 'm_')
plt.subplot(3,2,4);plt.xlabel('Time')
plt.subplot(3,2,4);plt.ylabel('R')
plt.subplot(3,2,4);plt.xlim([0,20])
plt.subplot(3,2,4);plt.ylim(bottom=500)

plt.subplot(3,2,5);plt.plot(y3[6], y3[3], 'm_')
plt.subplot(3,2,5);plt.xlabel('Time')
plt.subplot(3,2,5);plt.ylabel('N')
plt.subplot(3,2,5);plt.xlim([0,20])
plt.subplot(3,2,5);plt.ylim(bottom=4289)

plt.subplot(3,2,6);plt.plot(y3[6], y3[5], 'm_')
plt.subplot(3,2,6);plt.xlabel('Time')
plt.subplot(3,2,6);plt.ylabel('u')   
plt.subplot(3,2,6);plt.xlim([0,20])
plt.subplot(3,2,6);plt.ylim(bottom=0)
#------------------------------------------------------------------------------------------------------
plt.figure(4)            #Plot of y4
plt.subplots_adjust(None,None,None,None,0.5,0.5)    

plt.subplot(3,2,1);plt.plot(y4[6], y4[0])
plt.subplot(3,2,1);plt.xlabel('Time')
plt.subplot(3,2,1);plt.ylabel('S')
plt.subplot(3,2,1);plt.xlim([0,20])
plt.subplot(3,2,1);plt.ylim(bottom=800)

plt.subplot(3,2,2);plt.plot(y4[6], y4[1])
plt.subplot(3,2,2);plt.xlabel('Time')
plt.subplot(3,2,2);plt.ylabel('E')
plt.subplot(3,2,2);plt.xlim([0,20])
plt.subplot(3,2,2);plt.ylim(bottom=2000)

plt.subplot(3,2,3);plt.plot(y4[6], y4[2])
plt.subplot(3,2,3);plt.xlabel('Time')
plt.subplot(3,2,3);plt.ylabel('I')   
plt.subplot(3,2,3);plt.xlim([0,20])
plt.subplot(3,2,3);plt.ylim(bottom=3160)

plt.subplot(3,2,4);plt.plot(y4[6], y4[4])
plt.subplot(3,2,4);plt.xlabel('Time')
plt.subplot(3,2,4);plt.ylabel('R')
plt.subplot(3,2,4);plt.xlim([0,20])
plt.subplot(3,2,4);plt.ylim(bottom=1000)

plt.subplot(3,2,5);plt.plot(y4[6], y4[3])
plt.subplot(3,2,5);plt.xlabel('Time')
plt.subplot(3,2,5);plt.ylabel('N')
plt.subplot(3,2,5);plt.xlim([0,20])
plt.subplot(3,2,5);plt.ylim(bottom=8431)

plt.subplot(3,2,6);plt.plot(y4[6], y4[5])
plt.subplot(3,2,6);plt.xlabel('Time')
plt.subplot(3,2,6);plt.ylabel('u')   
plt.subplot(3,2,6);plt.xlim([0,20])
plt.subplot(3,2,6);plt.ylim(bottom=0)
#-----------------------------------------------------------------------------------------
plt.figure(5)      #Plot of y1 - y4 
plt.subplots_adjust(None,None,None,None,0.5,0.5)    

plt.subplot(3,2,1);plt.plot(y1[6], y1[0], 'r-.', y2[6], y2[0],'g--', y3[6], y3[0], 'm_', y4[6], y4[0])
plt.subplot(3,2,1);plt.xlabel('Time')
plt.subplot(3,2,1);plt.ylabel('S')
plt.subplot(3,2,1);plt.xlim([0,20])
plt.subplot(3,2,1);plt.ylim(bottom=800)

plt.subplot(3,2,2);plt.plot(y1[6], y1[1], 'r-.', y2[6], y2[1],'g--', y3[6], y3[1], 'm_', y4[6], y4[1])
plt.subplot(3,2,2);plt.xlabel('Time')
plt.subplot(3,2,2);plt.ylabel('E')
plt.subplot(3,2,2);plt.xlim([0,20])
plt.subplot(3,2,2);plt.ylim(bottom=39)

plt.subplot(3,2,3);plt.plot(y1[6], y1[2], 'r-.', y2[6], y2[2],'g--',  y3[6], y3[2], 'm_', y4[6], y4[2])
plt.subplot(3,2,3);plt.xlabel('Time')
plt.subplot(3,2,3);plt.ylabel('I')   
plt.subplot(3,2,3);plt.xlim([0,20])
plt.subplot(3,2,3);plt.ylim(bottom=43.5)

plt.subplot(3,2,4);plt.plot(y1[6], y1[4], 'r-.', y2[6], y2[4],'g--', y3[6], y3[4], 'm_', y4[6], y4[4])
plt.subplot(3,2,4);plt.xlabel('Time')
plt.subplot(3,2,4);plt.ylabel('R')
plt.subplot(3,2,4);plt.xlim([0,20])
plt.subplot(3,2,4);plt.ylim(bottom=15)

plt.subplot(3,2,5);plt.plot(y1[6], y1[3], 'r-.', y2[6], y2[3],'g--', y3[6], y3[3], 'm_', y4[6], y4[3])
plt.subplot(3,2,5);plt.xlabel('Time')
plt.subplot(3,2,5);plt.ylabel('N')
plt.subplot(3,2,5);plt.xlim([0,20])
plt.subplot(3,2,5);plt.ylim(bottom=1165)

plt.subplot(3,2,6);plt.plot(y1[6], y1[5], 'r-.', y2[6], y2[5],'g--', y3[6], y3[5], 'm_', y4[6], y4[5])
plt.subplot(3,2,6);plt.xlabel('Time')
plt.subplot(3,2,6);plt.ylabel('u')   
plt.subplot(3,2,6);plt.xlim([0,20])
plt.subplot(3,2,6);plt.ylim(bottom=0)

plt.show()