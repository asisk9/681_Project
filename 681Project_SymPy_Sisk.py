"""
MATH 681 project: Exploring the SymPy library through epidemic models

This code includes three functions: eq_points, Stability, NextGen
eq_points: finds equilibrium points
Arguments: the LHS of the model, the state variables

Stability: determines the eigenvalues of the jacobian evaluated at an 
equilibrium point (to help determine conditions for stability)
Arugments: the LHS of the model, the state variables, equilibrium point 


NextGen: finds the basic reproductive number using the next generation method
Arguments: State variable for the infected class(es), state variables, terms that 
represent inflow into the infected classes, other terms in the infected class(es), 
disease free equilibrium

Also includes a main script where two models are defined, the functions are used,
and the results are printed

Model 1 comes from "An Introduction to Mathematical Biology" Ch. 6 Pg. 273 by L. Allen

Model 2 comes from "Mathematical Models for Communicable Diseased" Lecture 2.2 Pg. 35 
by F. Brauer, and C. Castillo-Chavez

Created: Nov. 2020
Author: Anna Sisk
"""

import sympy as sym
from sympy.solvers.solveset import nonlinsolve

#Function definitions
def eq_points(system,state_var):
    [eq1,eq2]=nonlinsolve(system, state_var)
    eq_pts=[eq1,eq2]
    return eq_pts
def stability(system,state_var,eq):
    #Find the jacobian
    X=sym.Matrix(system) 
    Y= sym.Matrix(state_var)   
    jacobian=X.jacobian(Y)

    #Evaluate Jacobian at equilibrium point
    jacobian_eq=jacobian
    for i in range(len(state_var)):
        jacobian_eq=jacobian_eq.subs(state_var[i],eq[i])
    
    #Find the eigenvalues of the jacobian evaluated at equilibrium point
    eigenval=[]
    for i in jacobian_eq.eigenvals():
        eigenval.append(i)
    for i in range(len(eigenval)):
        if eigenval[i]==0:
            warning="Stability cannot be determine due to zero eigenvalue"
            zeoeig=True
            break
        else:
            zeoeig=False
    if zeoeig==True:
        return warning
    else:
        return eigenval
def NextGen(InfectVar,state_var,InFlow,OFlow,DFE):
    f=sym.Matrix(InFlow)
    v=sym.Matrix(OFlow)

    F=f.jacobian(InfectVar)
    V=v.jacobian(InfectVar)

    for i in range(len(InfectVar)):
        F=F.subs(state_var[i],DFE[i])

    for i in range(len(InfectVar)):
        V=V.subs(state_var[i],DFE[i])   
    InverseV=sym.Inverse(V)
    NextGen=F*InverseV
    return NextGen.eigenvals()




######################
# Main Script
######################

#pick which model to run by changing the i value
i=1

if i==1:
#SI model
    #State variables
    S = sym.Symbol('S')
    I = sym.Symbol('I')
    state_var=[S,I]
    #Parameters
    b = sym.Symbol('b')
    g = sym.Symbol('g')
    N = sym.Symbol('N')
    #Model definition
    Sdot=(-b/N)*S*I+g*I
    Idot=(b/N)*S*I-g*I
    system=[Sdot,Idot]
    
if i==2:
# SI model
    #State variables
    S = sym.Symbol('S')
    I = sym.Symbol('I')
    state_var=[S,I]

    #Parameters
    a = sym.Symbol('a')
    b = sym.Symbol('b')
    u = sym.Symbol('u')
    h = sym.Symbol('h')

    #Model definition
    Sdot=h-b*S*I-u*S
    Idot=b*S*I-u*I-a*I
    system=[Sdot, Idot]
  

######################
# calling functions
# ######################
eq=eq_points(system,state_var)
stability=stability(system,state_var,eq[0])

if i==1:
    basic_repro=NextGen([I],state_var,[(b/N)*S*I],[g*I],eq[0])
if i==2:
    basic_repro=NextGen([I],state_var,[b*S*I],[I*(u+a)],eq[0])


######################## 
# printing 
# ######################
sym.pprint('The equilibrium points are {}.'.format(eq))
sym.pprint('Stability of the equilibrium: {}.'.format(stability))
print('The basic reproductive number is' )
for x in basic_repro:
    print('{}.'.format(x))


