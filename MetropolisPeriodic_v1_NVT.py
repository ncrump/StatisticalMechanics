"""
Created on Sat Mar 08 18:00:56 2014
PHYS 780, Assignment 6
Nick Crump
"""

# Metropolis Monte Carlo Simulation with Periodic Boundary Conditions
# Canonical Ensemble (NVT)

# v1 updated and faster
"""
Simulation of 256 point particles that interact through the Lennard-Jones
potential in a cube with periodic boundary conditions.
"""

import numpy as np
import RandomSphere as RS
import CubicLattice as CL
from datetime import datetime
import matplotlib.pyplot as plt

t0 = datetime.now()

# function to calculate positional order parameter
#----------------------------------------------------
def orderFCC(x,y,z,a,N):
    ordX = sum(np.cos(4*np.pi*x/a))
    ordY = sum(np.cos(4*np.pi*y/a))
    ordZ = sum(np.cos(4*np.pi*z/a))
    ordP = (ordX + ordY + ordZ)/(3.0*N)
    return ordP
#----------------------------------------------------

# function to implement minimum image convention
#----------------------------------------------------
def minImage(dX,dY,dZ,L):

    if dX > 0.5*L: dX = dX - L
    elif dX < -0.5*L: dX = dX + L

    if dY > 0.5*L: dY = dY - L
    elif dY < -0.5*L: dY = dY + L

    if dZ > 0.5*L: dZ = dZ - L
    elif dZ < -0.5*L: dZ = dZ + L

    imgR = (dX**2 + dY**2 + dZ**2)**0.5
    return imgR
#----------------------------------------------------


#----------------------------------------------------
# define initial simulation values
dr = 0.02                   # total displacement step
runs = 1000                # number of runs

# define FCC lattice size (this specifies number of atoms N)
trans = 3                   # translations of 4-atom FCC unit cell

# define periodic boundary box values
d = 1.26417                 # distance between atoms
#d = 1.12246
a = d*(2**0.5)              # side length of FCC unit cube
L = (trans+1)*a             # side length of boundary box
Rc = 0.98*0.5*L             # cutoff distance bewteen atoms

# define temperatures
temps = [0.1,0.5,1.0,1.5,2.0]
#----------------------------------------------------


# define arrays to store results
U = []                      # potential energy
var = []                    # variance of energy
acc = []                    # acceptance probability
order = []                  # FCC order parameter
Uhist = []

# print number of atoms N in FCC lattice
N = 4*(trans+1)**3
print 'Number of atoms = ',N

# loop through temperatures for calculating energy
for T in temps:

    # temp arrays for storing values
    UT = []
    ordT = []
    accT = 0

    # initialize atom positions in FCC lattice
    px,py,pz = CL.FCC(d,trans)

    # calculate initial order parameter of FCC lattice
    ord0 = orderFCC(px,py,pz,a,N)
    ordT.append(ord0)

    # initialize matrix to store LJ potential between each atom
    LJm = np.zeros((N,N))

    #----------------------------------------------------
    # get initial energy of system
    for i in xrange(N-1):
        for j in xrange(i+1,N):

            # implement mimimum image convention
            dX = px.item(i) - px.item(j)
            dY = py.item(i) - py.item(j)
            dZ = pz.item(i) - pz.item(j)
            imgR = minImage(dX,dY,dZ,L)

            # implement cut & shift of potential
            if imgR > Rc:
                LJm[i][j] = 0.0
            else:
                LJm[i][j] = 4.0*(imgR**-12 - imgR**-6 - Rc**-12 + Rc**-6)
    #----------------------------------------------------

    # get initial potential energy
    pe = np.sum(LJm)
    UT.append(pe)

    # loop through atoms for averaging
    for run in xrange(runs):

        # generate displacements and random numbers for this run
        dx0,dy0,dz0 = RS.RandomSphere(N,dr)
        rand0 = np.random.uniform(0,1,N)

        # loop through atoms for Monte Carlo sim
        for i in xrange(N):
            # store old values for atom i
            xi,yi,zi = px.item(i), py.item(i), pz.item(i)
            LJiRow = np.copy(LJm[i,:])
            LJiCol = np.copy(LJm[:,i])

            # get random 3D displacement and move atom
            xnew, ynew, znew = xi+dx0.item(i), yi+dy0.item(i), zi+dz0.item(i)

            #----------------------------------------------------
            # implement periodic boundary conditions
            px[i] = xnew % L
            py[i] = ynew % L
            pz[i] = znew % L

            # get new energies only between moved atom to neighbors
            for j in xrange(N):
                if i != j:

                    # implement mimimum image convention
                    dX = px.item(i) - px.item(j)
                    dY = py.item(i) - py.item(j)
                    dZ = pz.item(i) - pz.item(j)
                    imgR = minImage(dX,dY,dZ,L)

                    # implement cut & shift of potential
                    if imgR > Rc:
                        LJm[i][j] = 0.0
                        LJm[j][i] = 0.0
                    else:
                        LJm[i][j] = 4.0*(imgR**-12 - imgR**-6 - Rc**-12 + Rc**-6)
                        LJm[j][i] = 0.0
            #----------------------------------------------------

            # calculate new system energy and delta
            peNew = np.sum(LJm)
            delta = peNew - pe

            # accept move if new energy is lower
            if delta < 0:
                pe = peNew
                UT.append(pe)
                accT += 1.0

            else:
                # check second condition if new energy is not lower
                prob = np.exp(-delta/T)
                rand = rand0.item(i)

                # accept move with this probability
                if rand < prob:
                    pe = peNew
                    UT.append(pe)
                    accT += 1.0

                # reject move otherwise (undo move)
                else:
                    px[i],py[i],pz[i] = xi, yi, zi
                    LJm[i,:] = LJiRow
                    LJm[:,i] = LJiCol
                    UT.append(pe)

        # store order parameter after each pass through system
        ord0 = orderFCC(px,py,pz,a,N)
        ordT.append(ord0)

    # store average energy for each temp
    aveU = np.average(UT)
    U.append(aveU)
    Uhist.append(UT)

    # store variance of energy for each temp
    var0 = np.var(UT)
    var.append(var0)

    # store acceptance probability for each temp
    acc0 = accT/(N*runs)
    acc.append(acc0)

    # store order parameters for each temp
    order.append(ordT)

    print 'T = %1.2f <U> = %1.4f var = %1.4f acc = %1.2f' % (T,aveU,var0,acc0)


# stack and store atom positions for visualization in Avogadro
#----------------------------------------------------
atoms = np.ones(len(px))*7
stack = np.column_stack((atoms,px,py,pz))
np.savetxt('Positions_256Atoms_Rho0.7_T2_Runs1k.txt',stack,fmt=('%i','%f4','%f4','%f4'))
#----------------------------------------------------

# stack and store results to output file
#----------------------------------------------------
xOrd = range(runs+1)
stack1 = np.column_stack((temps,U,var,acc))
stack2 = np.column_stack((xOrd,order[0],order[1],order[2],order[3],order[4]))
stack3 = np.column_stack((Uhist[0],Uhist[1],Uhist[2],Uhist[3],Uhist[4]))
np.savetxt('256atom_energy.txt',stack1,fmt='%1.12f',delimiter=' ')
np.savetxt('256atom_order.txt',stack2,fmt='%1.12f',delimiter=' ')
np.savetxt('256atom_uhist.txt',stack3,fmt='%1.12f',delimiter=' ')
#----------------------------------------------------

# plot results
#----------------------------------------------------
# plot potential energy vs temperature
plt.figure()
plt.subplot(311)
plt.plot(temps,U,'b.-',label='$<U>$')
plt.ylabel('Energy')
plt.legend(loc=4)

# plot variance vs temperature
plt.subplot(312)
plt.plot(temps,var,'r.-',label='$\sigma^2$')
plt.ylabel('Variance')
plt.legend(loc=4)

# plot acceptance probability vs temperature
plt.subplot(313)
plt.plot(temps,acc,'g.-',label='$P_{acc}$')
plt.ylabel('Acceptance Probability')
plt.xlabel('Temperature')
plt.legend(loc=4)

# plot order parameters for selected temperatures
indx = [0,1,2,3,4]
xOrd = range(runs+1)
plt.figure()
for i in indx:
    label = '$T$='+np.str(temps[i])
    plt.plot(xOrd,order[i],label=label)
    plt.legend(loc=1)
plt.ylabel('FCC Order')
plt.xlabel('MMC Step')
#----------------------------------------------------

t1 = datetime.now()
print 'runtime =',t1-t0