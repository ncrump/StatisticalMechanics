"""
Created on Sat Feb 08 12:47:22 2014
PHYS 780, Assignment 2
Nick Crump
"""

# Problem 2
"""
Simulation of a random walk in 3-Dimensions on a cube of evenly spaced points.  
"""

import RandomWalk as walk
import numpy as np
import matplotlib.pyplot as plt


# calculate mean displacement of random walker on a 3-D cube
#*******************************************************************
# set number of walks to average over and range of steps per walk
Nwalks = 100
Nsteps = range(1,100000,5000)

# define array to store displacements
dMean = []

# loop through step range
for i in Nsteps:
    d = []
    
    # loop through number of walks and steps per walk
    for j in range(Nwalks):
        # call RandomWalk3D from my random walk module
        xp,yp,zp,dist = walk.RandomWalk3D(i)
        d.append(dist)
        
    # get average displacement for N walks
    dAve = np.average(d)
    dMean.append(dAve)


# relation should be a power function so take logs of both sides
x = np.log(np.array(Nsteps))
y = np.log(np.array(dMean))
# calculate slope of log-log plot to get power coefficient
rise = y[len(y)-1] - y[0]
run = x[len(x)-1] - x[0]
slope = round(rise/run,3)

# plot mean displacement vs N steps
plt.figure()
plt.subplot(211)
plt.plot(Nsteps,dMean,label='$<d>$')  
plt.xlabel('N Steps')
plt.ylabel('$<d>$')
plt.annotate('100 Walks',fontsize=14,xy=(0.15,0.77),xycoords='figure fraction')
plt.legend(loc=2)

# plot log-log of mean displacement vs N steps 
plt.subplot(212)
plt.plot(x,y,label='Log-Log $<d>$')
plt.xlabel('log N Steps')
plt.ylabel('log $<d>$')
plt.annotate('Slope '+str(slope),fontsize=14,xy=(0.15,0.33),xycoords='figure fraction')
plt.legend(loc=2)