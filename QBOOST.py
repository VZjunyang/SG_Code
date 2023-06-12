import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp  
import math

def Corr(l1,l2):
    '''
    ## Description:
        Correlation function between two lists of the same length.

    ## Returns:
        corr (float): correlation between the two lists.

    ## Parameters:
        l1 (list): list of predictions of the first classifier (1 or 0)
        l2 (list): list of predicions of the second classifier / true values (1 or 0)
    '''

    N = len(l1)
    # Get the correlation
    corr = 0
    for i in range(len(l1)):
        corr += (2*l1[i]-1)*(2*l2[i]-1)/N**2 # Scaling to -1/N,1/N
    return corr

def QUBO(L,x,y,lambda_):
    '''
    ## Description:
        Defines QUBO problem

    ## Returns:
        Q: QUBO matrix


    ## Parameters:
        L: list of weak classifiers (0 and 1)
        x: data
        y: labels
        lambda_: regularization parameter
    '''
    N = len(L)
    S = len(x)


    # Define QUBO matrix
    Q = np.zeros((N,N))

    for i in range(N):
        for j in range(i,N):
            if j != i:
                Q[i,j] = Corr(L[i].predict(x),L[j].predict(x))
                Q[j,i] = Q[i,j]
            else:
                Q[i,i] = lambda_ - 2*Corr(np.array(L[i].predict(x)), np.array(y))
    
    return Q

def simulated_annealing(func, init_state, T, reanneal, max_iter):
    """
        ## Description:
            Simulated annealing algorithm.
            Finds the minimum of a function func using simulated annealing.

        ## Returns:
            current_state: final solution
            current_energy: final energy

        ## Parameters:
            func: function to minimize
            init_state (array of 0 and 1): initial solution
            T: initial temperature
            reanneal: number of iterations before reannealing
            max_iter: maximum number of iterations
    """

    # Initialize
    current_state = init_state
    current_energy = func(current_state)

    for iteration in range(max_iter):
        new_state = current_state.copy()

        # Flip one bit
        index = np.random.randint(0,len(new_state))
        new_state[index] = 1 - new_state[index]

        # Compute new energy
        new_energy = func(new_state)

        # If new energy is lower, accept new state
        if new_energy < current_energy:
            current_state = new_state
            current_energy = new_energy
        
        # If new energy is higher, accept new state with probability
        else:
            p = np.exp(-(new_energy - current_energy)/T)      # Compute probability
            if np.random.rand() < p:
                current_state = new_state
                current_energy = new_energy
        
        # Update temperature
        if iteration % reanneal == 0:
            T = T*0.95
    
    return current_state, current_energy 

def RGS(L,x,y,lambda_):
    '''
    ## Description:
        Implements RGS method using weak classifiers and Simulated Annealing

    ## Returns:
    w (array): solution of the QUBO problem

    ## Parameters:
        L (list): list of weak classifiers
        x (array): data
        y (array): labels
        lambda_ (scalar): regularization parameter
    '''

    # Define QUBO matrix
    Q = QUBO(L,x,y,lambda_)

    # Define QUBO problem
    
    def f(w):
        s=0
        for i in range(len(w)):
            for j in range(len(w)):
                s+= Q[i,j]*w[i]*w[j]
        return s
    
    # Define initial state
    init_state = np.zeros(len(L))
    
    # Run simulated annealing
    w, energy = simulated_annealing(f,init_state,1,100,1000)

    return w, energy

def PredictRGS(L,w,x):
    """
    ## Description:
        Predicts the labels of the data x using QBoost method (see paper for formula)

    ## Returns:
        List of predictions of the set of weak classifiers (0 or 1)

    ## Parameters:
        L (list): list of weak classifiers
        w (array): solution of the QUBO problem
        x (array): data
    """
    
    N = len(L)
    S = len(x)
    
    predictions = []

    for i in range(N):
        predictions.append(2* L[i].predict(x) - 1 ) # Scaling to -1,1

    predictions = np.array(predictions)

    # Calculate T
    T = 0
    for i in range(S):
        for j in range(N):
            T+= w[j]*predictions[j][i]
    T = np.sign(T)

    # Calculate predictions 
    C = []
    for i in range(S):
        p = 0
        for j in range(N):
            p += w[j]*predictions[j][i]
        p = np.sign(p-T)
        C.append(p)
    
    # Replace -1 by 0
    for i in range(len(C)):
        if C[i] == -1:
            C[i] = 0
    return C

def PredictRGS_probas(L,w,x,p):
    """
    ## Description:
        Predicts the labels of the data x using QBoost method (see paper for formula) with a probability threshold

    ## Returns:
        List of predictions of the set of weak classifiers (0 or 1) using the threshold p

    ## Parameters:
        L (list): list of weak classifiers
        w (array): solution of the QUBO problem
        x (array): data
        p (scalar): probability threshold bewteen 0 and 1
    """

    N = len(L)
    S = len(x)
    
    predictions = []

    for i in range(N):
        predictions.append([])
        proba = L[i].predict_proba(x)
        for j in range(S):          
            if proba[j][0] > p:
                predictions[i].append(0)
            else:
                predictions[i].append(1)

    predictions = np.array(predictions)

    # Calculate T
    T = 0
    for i in range(S):
        for j in range(N):
            T+= w[j]*predictions[j][i]
    T = np.sign(T)

    # Calculate predictions 
    C = []
    for i in range(S):
        p = 0
        for j in range(N):
            p += w[j]*predictions[j][i]
        p = np.sign(p-T)
        C.append(p)
    
    # Replace -1 by 0
    for i in range(len(C)):
        if C[i] == -1:
            C[i] = 0
    return C