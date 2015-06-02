"""
This rotates a vector in the xy-plane by a 2D rotation matrix
"""

import numpy as np
from math import sin,cos,pi
import matplotlib.pyplot as plt

p1 = np.array([0,0])            # origin point
p2 = np.array([5,3])            # second point
v = p2 - p1                     # vector between points
v = np.transpose(np.matrix(v))  # turn into column vector

# plot original vector
plt.figure()
plt.plot([p1[0],v[0]],[p1[1],v[1]],'b-')

# loop through angles to apply rotation matrix to vector
for t in np.linspace(0,2*pi,10):
    m = np.matrix(([cos(t),-sin(t)],[sin(t),cos(t)]))
    vv = m*v
    plt.plot([p1[0],vv[0]],[p1[1],vv[1]],'--')