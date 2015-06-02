"""
This rotates a water molecule in 3D by a unit quaternion about a random axis
"""

import numpy as np
from math import pi,sin,cos,acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# water molecule bond length & bond angle
L = 1.0
A = 109.47*(pi/180)

# oxygen & hydrogen bonded sites
O = np.array([0,0,0])
H1 = np.array([O[0]+L*cos(A/2), O[1]+L*sin(A/2), O[2]])
H2 = np.array([O[0]+L*cos(A/2), O[1]-L*sin(A/2), O[2]])

# get molecule orientation vectors
vH1 = (H1-O)                                   # O to H1 vector at origin
vH2 = (H2-O)                                   # O to H2 vector at origin
magH1 = (np.sum(vH1**2))**0.5                  # H1 vector magnitude
magH2 = (np.sum(vH2**2))**0.5                  # H2 vector magnitude
uH1 = vH1/magH1                                # unit vector H1
uH2 = vH2/magH2                                # unit vector H2
angle = acos((np.sum(vH1*vH2))/(magH1*magH2))  # angle between vectors

# check bond length & angle before rotation
print ''
print 'bond length & angle before rotation: ', magH1,magH2,angle*(180/pi)

# matrix of bonded site orientation unit vectors
Vmol = np.matrix(([uH1[0],uH2[0]],[uH1[1],uH2[1]],[uH1[2],uH2[2]]))

# generate parameters for unit quaternion
# represents random orientation vector on surface of 4D unit sphere
s1 = 1
while s1 >= 1:
    r1 = np.random.uniform(-1,1,2)
    s1 = np.sum(r1**2)

s2 = 1
while s2 >= 1:
    r2 = np.random.uniform(-1,1,2)
    s2 = np.sum(r2**2)

# get parameters of random unit quaternion (4-vector)
z = ((1-s1)/s2)**0.5
Q = [r1[0],r1[1],r2[0]*z,r2[1]*z]
q0 = Q[0]
q1 = Q[1]
q2 = Q[2]
q3 = Q[3]
sq0 = q0**2
sq1 = q1**2
sq2 = q2**2
sq3 = q3**2

# form rotation matrix from unit quaternion parameters
R1 = [sq0+sq1-sq2-sq3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)]
R2 = [2*(q1*q2+q0*q3), sq0-sq1+sq2-sq3, 2*(q2*q3-q0*q1)]
R3 = [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), sq0-sq1-sq2+sq3]
Rot = np.matrix(([R1,R2,R3]))

# apply rotation matrix to molecule orientation unit vectors
Vlab = Rot*Vmol

# get rotated molecule orientation vectors
vH1p = np.array([Vlab[0,0],Vlab[1,0],Vlab[2,0]])      # rotated H1 vector
vH2p = np.array([Vlab[0,1],Vlab[1,1],Vlab[2,1]])      # rotated H2 vector
magvH1p = (np.sum(vH1p**2))**0.5                      # rotated H1 vector magnitude
magvH2p = (np.sum(vH2p**2))**0.5                      # rotated H2 vector magnitude
H1p = vH1p+O                                          # new H1 position
H2p = vH2p+O                                          # new H2 position
anglep = acos((np.sum(vH1p*vH2p))/(magvH1p*magvH2p))  # angle between vectors

# check bond length & angle after rotation
print 'bond length & angle after rotation:  ', magvH1p,magvH2p,anglep*(180/pi)

# plot molecule sites before & after rotation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot random unit quaternion
ax.plot([O[0],q1],[O[1],q2],[O[2],q3],'r-')
# plot original molecule sites
ax.plot([O[0],H1[0]],[O[1],H1[1]],[O[2],H1[2]],'bo-',label='Before',markersize=10)
ax.plot([O[0],H2[0]],[O[1],H2[1]],[O[2],H2[2]],'bo-',markersize=10)
ax.legend()
# plot rotated molecule sites
ax.plot([O[0],H1p[0]],[O[1],H1p[1]],[O[2],H1p[2]],'go--',label='After',markersize=10)
ax.plot([O[0],H2p[0]],[O[1],H2p[1]],[O[2],H2p[2]],'go--',markersize=10)
ax.legend()
ax.set_xlim(O[0]-1,O[0]+1)
ax.set_ylim(O[1]-1,O[1]+1)
ax.set_zlim(O[2]-1,O[2]+1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')