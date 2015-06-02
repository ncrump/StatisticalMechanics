"""
Created on Sun Feb 16 16:36:14 2014
PHYS 780, Assignment 3
Nick Crump
"""

# Metropolis Monte Carlo Simulation of Ising Model Atoms Spins
"""
Simulation of 6x6 lattice of particle spins that interact through the 
Ising Model of magnetization using the Metropolis Monte Carlo Method.  
"""

import numpy as np
import matplotlib.pyplot as plt


# define function that sums nearest neighbors over the lattice
# ----------------------------------------------------------
def sumNN(m):
    N = len(m)-2
    snn = 0
    for i in range(1,N+1):
        for j in range(1,N+1):
            nn = m[i,j]*(m[i+1,j] + m[i-1,j] + m[i,j+1] + m[i,j-1])
            snn = snn + nn    
    return snn*-1
# ----------------------------------------------------------
            

# define number of particles in lattice and number of runs for averaging
N = 6
runs = 1000
temps = np.arange(1,5.5,0.5)
spins = [-1,1]

# define arrays to store potential energy and magnetization
U = []
M = []

# loop through temperatures for calculating magnetization
for T in temps:    
    
    # temp arrays for storing values
    Ut = []
    Mt = []
    
    # initialize lattice of spins
    # NOTE: a larger lattice is used with zeros at boundaries
    #       this avoids having to check lattice boundaries
    lat = np.zeros((N+2,N+2))
    for i in range(1,N+1):
        for j in range(1,N+1):
            # randomly initialize spins of -1 or 1
            rand = np.random.randint(0,2)
            lat[i,j] = spins[rand]
            
    # get initial potential energy and magnetization
    pe = sumNN(lat)
    mag = np.abs(np.sum(lat))/(N*N)
    Ut.append(pe)
    Mt.append(mag)    
    
    # flip spin states & calculate new energy based on Metropolis rules
    # loop through runs for averaging
    for run in range(runs):
        
                
        # loop through lattice to get spin sites
        for i in range(1,N+1):
            for j in range(1,N+1):
                
                # flip spin for site i,j
                lat[i,j] = lat[i,j]*-1
                peNew = sumNN(lat)
                
                # check first Metropolis acceptance criteria
                delta = peNew - pe
                if delta < 0:
                    pe = peNew
                    
                else: 
                    # check second Metropolis acceptance criteria
                    prob = np.exp(-delta/T)
                    rand = np.random.uniform(0,1)
                    
                    # accept flip with this probability
                    if rand < prob:
                        pe = peNew
                        
                    # reject flip otherwise (undo flip)
                    else: 
                        lat[i,j] = lat[i,j]*-1
        
        # append results for each pass through lattice
        Ut.append(sumNN(lat))
        Mt.append(np.abs(np.sum(lat))/(N*N))
    
    
    # append average results for each temp
    U.append(np.average(Ut))
    M.append(np.average(Mt))
    
    
# plot magnetization vs temperature
plt.figure()
plt.subplot(211)
plt.plot(temps,M,'bo--', label='<M> vs T')
plt.ylabel('Magnetization')
plt.legend(loc=3)

# plot potential energy vs temperature
plt.subplot(212)
plt.plot(temps,U,'ro--', label='<U> vs T')
plt.ylabel('Potential Energy')
plt.xlabel('Temperature')
plt.legend(loc=2)