"""
Created on Sat Feb 08 12:47:22 2014
PHYS 780, Assignment 2
Nick Crump
"""

# Problem 1
"""
Implement random number generating algorithms to obtain uniformly distributed and
normally distributed random number sequences and plot their distributions. 
"""

import RandomNumber as RN
import numpy as np
import matplotlib.pyplot as plt


# generate 10**6 random numbers using the Lehmer modulo generator
# make a histogram to check it is a uniform distribution
#*******************************************************************
# initial seed to start random number generator
seed = 112183.0

# call randUniform Lehmer method from my random number module
lehmerU = RN.randUniform(seed,10**6)

# generate random numbers using numpy for comparison
numpyU = np.random.uniform(0,1,10**6)

# plot histogram of randUniform distribution
plt.figure()
plt.hist(lehmerU,bins=20,label='Lehmer Modulo Generator')
plt.xlabel('Random Number')
plt.ylabel('Frequency')
plt.legend()

# plot histogram of numpy uniform distribution
plt.figure()
plt.hist(numpyU,bins=20,label='Numpy Random Generator')
plt.xlabel('Random Number')
plt.ylabel('Frequency')
plt.legend()
#*******************************************************************


# implement Polar method for 10**6 random normal numbers
# make a histogram to check it is a normal distribution
#*******************************************************************
# call randNormal1 Polar method from my random number module
polarN = RN.randNormal1(10**6)

# generate random numbers using numpy for comparison
numpyN = np.random.normal(0,1,10**6)

# plot histogram of randNormal1 (Polar) distribution
plt.figure()
plt.hist(polarN,bins=20,label='Polar Method')
plt.xlabel('Random Number')
plt.ylabel('Frequency')
plt.legend()

# plot histogram of numpy normal distribution
plt.figure()
plt.hist(numpyN,bins=20,label='Numpy Random Generator')
plt.xlabel('Random Number')
plt.ylabel('Frequency')
plt.legend()
#*******************************************************************