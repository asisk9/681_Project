"""
MATH 681 project: Change in parameters during a simulation

This builds on the SIR.py code given in class. 

It solves and plots an SEIRN epidemic model and includes the ability to
change parameters during a run.

First run: no vaccination
Second run: constant vaccination from the beginning
Third run: Vaccination is introduced at t=j
Forth run: Vaccination is introduced at t=k and the removed at t=2*k

The model and parameters come from "Optimal Control Methods Applied to Disease Models"
by H.R.Joshi, S. Lenhart, M.Y. Li, L. Wang

Parameter meaning:
a = death rate caused by the disease
b = natural birth rate
c = adequate contact rate between susceptible and infectious individuals
d = natural death rate
e = the rate exposed individuals become infectious
g = recovery rate
v = vaccination rate

Created: Nov. 2020
Code modified by Anna Sisk
Original code by Christopher Strickland
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

#time points to solve at
final_time = 500
tpts = np.linspace(0, final_time,1001)

#initial values as for each class
S0 = 500
E0 = 10
I0 = 150
R0 = 140
N0 = S0+E0+I0+R0

#parameter values stored in a dictionary
params = {}
for i in range(0,4):
    if i==0: #No vaccine
       params['a'] = .03
       params['b'] = .008
       params['c'] = .4
       params['d'] = .005
       params['e'] = .05
       params['g'] = .04
       params['v'] = 0 
    if i==1: #Vaccination from the beginning
       params['a'] = .03
       params['b'] = .008
       params['c'] = .4
       params['d'] = .005
       params['e'] = .05
       params['g'] = .04
       params['v'] = .02
    if i==2: #Vaccination introduced later
       params['a'] = .03
       params['b'] = .008
       params['c'] = .4
       params['d'] = .005
       params['e'] = .05
       params['g'] = .04
       params['v'] = 0 
    if i==3: #Vaccination introduced later and then removed
       params['a'] = .03
       params['b'] = .008
       params['c'] = .4
       params['d'] = .005
       params['e'] = .05
       params['g'] = .04
       params['v'] = 0 

    ##################################

    #vectorize initial conditions
    x0 = np.array([S0,E0,I0,R0,N0])
        # define ode equations
    def SEI_ODEs(t,x,params):
        '''This function returns the time derivates of S,E,I,R,N.

        The ode solver expects the first two arguments to be t and x
        NOTE: This is the OPPPOSITE order from scipy.integrate.odeint!!

        The params argument should be a dict with a,b,c,d,e,g, and v as keys.
        It must be passed into the solver using the set_f_params method
        '''

        S = x[0]; E = x[1]; I = x[2]; R = x[3]; N = x[4]
        dx = np.zeros(5)

        dx[0] = params['b']*N-params['d']*S-(params['c']*S*I)/N-params['v']*S
        dx[1] = (params['c']*S*I)/N-params['e']*E-params['d']*E
        dx[2] = params['e']*E-params['g']*I-params['a']*I-params['d']*I
        dx[3] = params['g']*I-params['d']*R+params['v']*S
        dx[4] = params['b']*N-params['d']*N-params['a']*I

        return dx

    ###########################
    ##### Solve procedure #####
    ###########################
    Ssol = []; Esol = []; Isol = []; Rsol = []; Nsol = []

    # create solver object
    solver = ode(SEI_ODEs)
        # set the solver method to RK 4(5), Dormand/Prince
        # see the docs for scipy.integrate.ode for the options with this solver
        # here we'll use the defaults, but pay attention to atol/rtol especially
    solver.set_integrator('dopri5')
        # set the initial conditions
    solver.set_initial_value(x0,0)
        # pass in the solver parameters
    solver.set_f_params(params)

        # the solver will solve (integrate) up to a given time.
        # we want a mesh of times, so we will integrate up to each time in our
        # mesh in a loop
    for t in tpts:
        k = 170 #switch time for parameters during the fourth run
        j = 250 #switch time for parameters during the third run

        if t == 0: #records the initial conditions
            Ssol.append(x0[0])
            Esol.append(x0[1])
            Isol.append(x0[2])
            Rsol.append(x0[3])
            Nsol.append(x0[4])

        if 0<t<=k: #integrates up to the smaller switch time and records solution
            solver.integrate(t)
            assert solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(solver.y[0])
            Esol.append(solver.y[1])
            Isol.append(solver.y[2])
            Rsol.append(solver.y[3])
            Nsol.append(solver.y[4])

        if t>k and i!=2 and i!=3: #if we aren't in the third or fourth run then, the simulation continues unchanged
            solver.integrate(t)
            assert solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(solver.y[0])
            Esol.append(solver.y[1])
            Isol.append(solver.y[2])
            Rsol.append(solver.y[3])
            Nsol.append(solver.y[4])
        
        if k<t<=j and i==2: 
        #for the third run it continues to integrate up to its switch time without parameter changes and records solution
            solver.integrate(t)
            assert solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(solver.y[0])
            Esol.append(solver.y[1])
            Isol.append(solver.y[2])
            Rsol.append(solver.y[3])
            Nsol.append(solver.y[4])

        if k<t<=2*k and i==3:
        #for the fourth run once we hit its switch time (k) the parameters change (for the first time), 
        #then the solution is recorded with the new parameters
            params['v'] = .02

            solver.integrate(t)
            assert solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(solver.y[0])
            Esol.append(solver.y[1])
            Isol.append(solver.y[2])
            Rsol.append(solver.y[3])
            Nsol.append(solver.y[4])

        if t>2*k and i==3: #Second parameter change for the fourth run

            params['v'] = 0

            solver.integrate(t)
            assert solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(solver.y[0])
            Esol.append(solver.y[1])
            Isol.append(solver.y[2])
            Rsol.append(solver.y[3])
            Nsol.append(solver.y[4])
        
        if t>j and i==2: 
        #for the third run once we hit its switch time (j) the parameters change, 
        #then the solution is recorded with the new parameters
            params['v'] = .02

            solver.integrate(t)
            assert solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(solver.y[0])
            Esol.append(solver.y[1])
            Isol.append(solver.y[2])
            Rsol.append(solver.y[3])
            Nsol.append(solver.y[4])


#######################
##### Plot result ######
########################
    plt.figure()
    plt.title("Plot of $S,E,I$ vs. time {}".format(i+1))
    plt.plot(tpts, Ssol,tpts, Esol,tpts, Isol, tpts, Rsol)
    plt.legend(['S','$E$','$I$','$R$'])
    plt.xlabel("Time (months)")
    plt.ylabel("Population")
    plt.ylim(0,700)
    plt.xlim(0,final_time)

plt.show()
