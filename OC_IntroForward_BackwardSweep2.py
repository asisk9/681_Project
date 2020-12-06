"""
Solving a basic ODE Optimal Control using the Forward - Backward Sweep Algorithm

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
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0,1,N+1) # Time variable

def OC_F_B_Sweep(A,B,C):

 #Test represents the convergence test variable
 test = -1
 

 x0 = 1 # Initial value of x.
 delta = 0.001   # Accepted Tolerance

 h = 1/N    # Step size for Runge-Kutta 4 sweep
 h2 = h/2

 u = np.zeros(N+1)     # Initial guess for Control

 x = np.zeros(N+1)     # State
 x[0] = x0     # Stores initial value of State x.

 Lambda = np.zeros(N+1) # Adjoint

 #The while loop begins with the test function. This loop contains the Forward-Backward Sweep. The loop ends once converegence occurs (t>=0).
 while(test < 0):
    
     #Stores the current values of u, x, and lambda as the previous values of u, x, and lambda.
     oldu = u
     oldx = x
     oldLambda = Lambda

     # Generate new values. 

     #Runge-Kutta 4 sweep: Solving x forward in time. x1 is used  to find x2, x2 is used to solve x3 and so on.
     for i in np.arange (0,N):
         k1 = -0.5*x[i]**2 + C*u[i]
         k2 = -0.5*(x[i] + h2*k1)**2 + C*0.5*(u[i] + u[i+1])
         k3 = -0.5*(x[i] + h2*k2)**2 + C*0.5*(u[i] + u[i+1])
         k4 = -0.5*(x[i] + h*k3)**2 + C*u[i+1]
         x[i+1] = x[i] + (h/6)*(k1 + 2*k2 +2*k3 +k4)   
     
     #Runge-Kutta 4 sweep: Solving Lambda backwards in time. Lambda1 is used  to find Lambda2, Lambda2 is used to solve Lambda3 and so on.
     for i in np.arange (0,N):
         j = N + 2 - (i+2)
         k1 = -A + Lambda[j]*x[j]
         k2 = -A + (Lambda[j] - h2*k1)*0.5*(x[j]+x[j-1])
         k3 = -A + (Lambda[j] - h2*k2)*0.5*(x[j]+x[j-1])
         k4 = -A + (Lambda[j] - h*k3)*x[j-1]
         Lambda[j-1] = Lambda[j] - (h/6)*(k1 + 2*k2 + 2*k3 +k4)
        
    
    
     u1 = C*Lambda/(2*B)    #Represents u using the new values  for lambda
     u = 0.5*(u1 + oldu)     # Control 
    
     #Convergence test parameters of each variable.
     temp1 = delta*sum(abs(u)) - sum(abs(oldu - u))
     temp2 = delta*sum(abs(x)) - sum(abs(oldx - x))
     temp3 = delta*sum(abs(Lambda)) - sum(abs(oldLambda - Lambda))
    
     test = min(temp1, min(temp2, temp3))

 #Stores the value of the final vectors.
 y = [x, Lambda, u]
 #y[0] = x
 #y[1] = Lambda
 #y[2] = u
 return y


y1 = OC_F_B_Sweep(A = 1,B = 1,C = 4)

y2 = OC_F_B_Sweep(A = 2,B = 1,C = 4)

y3 = OC_F_B_Sweep(A = 2,B = 4,C = 4)


# Plot Results
plt.figure(figsize=(9,7))  
plt.subplot(3,1,1);plt.plot( t, y1[0], 'r-.', t, y2[0],'g--',  t, y3[0])
plt.subplot(3,1,1);plt.xlabel('Time')
plt.subplot(3,1,1);plt.ylabel('State')
plt.subplot(3,1,1);plt.xlim([0,1])
plt.subplot(3,1,1);plt.ylim(bottom=1)

plt.subplot(3,1,2);plt.plot( t, y1[1], 'r-.', t, y2[1],'g--',  t, y3[1])
plt.subplot(3,1,2);plt.xlabel('Time')
plt.subplot(3,1,2);plt.ylabel('Adjoint')
plt.subplot(3,1,2);plt.xlim([0,1])
plt.subplot(3,1,2);plt.ylim(bottom=0)

plt.subplot(3,1,3);plt.plot( t, y1[2], 'r-.', t, y2[2],'g--',  t, y3[2])
plt.subplot(3,1,3);plt.xlabel('Time')
plt.subplot(3,1,3);plt.ylabel('Control')   
plt.subplot(3,1,3);plt.xlim([0,1])
plt.subplot(3,1,3);plt.ylim(bottom=0)

plt.show()