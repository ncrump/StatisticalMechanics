"""
This rotates a unit vector in 3D about an arbitrary axis by vector scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

p1 = np.array([0,0,0])          # origin point
p2 = np.array([5,3,1])          # second point

v = p2 - p1                     # vector between points
magv = (np.sum(v**2))**0.5      # vector magnitude
uv = v/magv                     # unit vector

# generate random numbers to construct random unit vector
s = 1
while s >= 1:
    rand = np.random.uniform(-1,1,2)
    s = np.sum(rand**2)

# construct unit vector with random orientation
z = 2*((1-s)**0.5)
ur = np.array([z*rand[0],z*rand[1],1-2*s])

# plot original & random unit vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# original unit vector
ax.plot([p1[0],uv[0]],[p1[1],uv[1]],[p1[2],uv[2]],'bo-')
# random unit vector
ax.plot([p1[0],ur[0]],[p1[1],ur[1]],[p1[2],ur[2]],'r-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# loop through scale factors (controls magnitude of rotation)
for gam in np.arange(0,1,0.2):
    t = gam*ur + uv             # rotate original vector by scaling
    magt = (np.sum(t**2))**0.5  # rotated vector magnitude
    ut = t/magt                 # normalized rotated vector
    ax.plot([p1[0],ut[0]],[p1[1],ut[1]],[p1[2],ut[2]],'bo--')