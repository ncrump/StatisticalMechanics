"""
Created on Fri Feb 21 21:08:07 2014
PHYS 780, Assignment 4
Nick Crump
"""

# Metropolis Monte Carlo Simulation with Free Boundary Conditions
# Canonical Ensemble (NVT)
"""
Simulation of 13 point particles that interact through the Lennard-Jones
potential in 3D with free boundary conditions.
"""

import numpy as np
import RandomSphere as RS
import LennardJones as LJ
import matplotlib.pyplot as plt


# define initial simulation values
N = 13                    # number of atoms
R = 1.3                   # radius of sphere
dr = 0.02                 # total displacement step
runs = 1000               # number of runs

temps = [0.55,0.5,0.4,0.3,0.2,0.1,0.05]
# NOTE: first temperature run is to allow system to settle to equilibrium
#       results of first temperature are thrown out

# define arrays to store potential energy and standard deviation
U = []
std = []

# initialize atom positions randomly on surface of sphere
px,py,pz = RS.RandomSphere(N-1,R)
px = np.append(px,0)
py = np.append(py,0)
pz = np.append(pz,0)

print 'initial U =',LJ.LJ(px,py,pz,1.0,1.0)

# loop through temperatures for calculating energy
for T in temps:

    # temp array for storing values
    Ut = []

    # get initial potential energy
    pe = LJ.LJ(px,py,pz,1.0,1.0)
    Ut.append(pe)

    # loop through atoms for averaging
    for run in range(runs):

        # loop through atoms for Monte Carlo sim
        for i in range(N):
            # store old position of atom i
            xi,yi,zi = px[i], py[i], pz[i]

            # get random 3D displacement and move atom
            dx,dy,dz = RS.RandomSphere(1,dr)
            px[i],py[i],pz[i] = xi+float(dx), yi+float(dy), zi+float(dz)

            # calculate new energy and delta
            peNew = LJ.LJ(px,py,pz,1.0,1.0)
            delta = peNew - pe

            # accept move if new energy is lower
            if delta < 0:
                pe = peNew
                Ut.append(pe)

            else:
                # check second condition if new energy is not lower
                prob = np.exp(-delta/T)
                rand = np.random.uniform(0,1)

                # accept move with this probability
                if rand < prob:
                    pe = peNew
                    Ut.append(pe)

                # reject move otherwise (undo move)
                else:
                    px[i],py[i],pz[i] = xi, yi, zi
                    Ut.append(pe)

    # store average results for each temp
    aveU = np.average(Ut)
    U.append(aveU)

    # calculate and store variance for each temp
    std0 = np.std(Ut)
    std.append(std0)
    print 'T =','%1.2f' % T, ' <U> =','%1.4f' % aveU, 'std =','%1.4f' % std0

# plot potential energy vs temperature
plt.figure()
plt.errorbar(temps[:0:-1],U[:0:-1],yerr=std[:0:-1],fmt='bo--', label='<U> vs T')
plt.xlabel('Temperature')
plt.ylabel('Potential Energy')
plt.legend(loc=2)