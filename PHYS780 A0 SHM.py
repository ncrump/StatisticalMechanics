"""
Created on Mon Jan 27 22:32:12 2014
PHYS 780, Assignment 0 
Nick Crump
"""

# Practice problem
"""
Plots force F(t), kinetic energy KE(t), potential energy PE(t) for 
simple harmonic motion as a function of time.
"""

import numpy as np
from math import sin,cos
import matplotlib.pyplot as plt

# define variables
C = 1
m = 1
phi = 0
omega = 1

# define arrays
F = []
KE = []
PE = []
t = np.arange(0,20,0.1)

# loop to calculate F(t), KE(t), PE(t)
for i in t:
    Fi = -m*(omega**2)*C*sin(omega*i + phi)
    KEi = 0.5*m*(C**2)*(omega**2)*(cos(omega*i + phi))**2
    PEi = 0.5*m*(C**2)*(omega**2)*(sin(omega*i + phi))**2
    
    F.append(Fi)
    KE.append(KEi)
    PE.append(PEi)

# plot F,KE,PE vs time
plt.figure()
plt.plot(t,F,'g--', label='F')
plt.plot(t,KE,'b-', label='KE')
plt.plot(t,PE,'r-', label='PE')
plt.xlabel('Time')
plt.legend(loc=2)