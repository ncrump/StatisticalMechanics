"""
Created on Fri Mar 21 20:26:05 2014
PHYS 780, Assignment 7
Nick Crump
"""

# Metropolis Monte Carlo Simulation with Periodic Boundary Conditions
# Canonical Ensemble (NVT)

# v2 updated and faster, plots for multiple densities
# implemented in Cython
"""
Simulation of 256 point particles that interact through the Lennard-Jones
potential in a cube with periodic boundary conditions.
"""

import cython
import numpy as np
cimport numpy as np
import FCC_Cython as FCC
import RandomSphere_Cython as RS
import matplotlib.pyplot as plt
from datetime import datetime

plt.ioff()                   # suppresses plot windows to not display
#@cython.boundscheck(False)  # turn off array bounds checking for speed
#@cython.wraparound(False)   # turn off reverse indexing for speed

cpdef SimRun():

    t0 = datetime.now()

    # setup plot windows
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)

    # define C-type variables
    cdef float pi = 3.14159
    cdef float dr,drr,d,a,L,Rc,Lhalf
    cdef float ordX,ordY,ordZ,ord0
    cdef float xi,yi,zi,xnew,ynew,znew
    cdef float dX,dY,dZ,imgR
    cdef float pe,peNew,delta,accT
    cdef float T,aveU,var0,acc0
    cdef int N,i,j,k,run,trans,indx,hlf

    # define numpy C-type arrays
    cdef np.ndarray[dtype=double,ndim=1] dens,temps
    cdef np.ndarray[dtype=double,ndim=1] d_rho,a_rho,Rc_rho
    cdef np.ndarray[dtype=double,ndim=1] L_rho,Lhalf_rho
    cdef np.ndarray[dtype=double,ndim=1] px,py,pz
    cdef np.ndarray[dtype=double,ndim=1] dx0,dy0,dz0
    cdef np.ndarray[dtype=double,ndim=1] rand0
    cdef np.ndarray[dtype=double,ndim=1] UT,ordT
    cdef np.ndarray[dtype=double,ndim=1] LJRow,LJCol
    cdef np.ndarray[dtype=double,ndim=2] LJm

    #----------------------------------------------------
    # define initial simulation values
    dr = 0.02                  # initial step size
    runs = 1000               # number of runs

    # define FCC lattice size (this specifies number of atoms N)
    trans = 3                  # translations of 4-atom FCC unit cell

    # define densities and temperatures
    dens = np.array([0.9])
    temps = np.array([0.1,0.5,1.0,1.5,2.0])
    nRho = len(dens)
    nTmp = len(temps)

    d_rho = 1.12246*dens**-0.33333  # distance between atoms
    a_rho = d_rho*(2**0.5)          # side length of FCC unit cube
    L_rho = (trans+1)*a_rho         # side length of boundary box
    Rc_rho = 0.98*0.5*L_rho         # cutoff distance bewteen atoms
    Lhalf_rho = 0.5*L_rho           # half length of boundary box
    #----------------------------------------------------

    # print number of atoms N in FCC lattice
    N = 4*(trans+1)**3
    print 'Number of atoms = ',N

    for k in range(nRho):

        # define arrays to store results
        U = []                      # potential energy
        var = []                    # variance of energy
        acc = []                    # acceptance probability
        order = []                  # FCC order parameter

        # define parameters for this density
        Rho = np.float(dens[k])
        d = d_rho[k]
        a = a_rho[k]
        L = L_rho[k]
        Rc = Rc_rho[k]
        Lhalf = Lhalf_rho[k]
        hlf = int(0.5*N*runs)
        drr = 1

        # loop through temperatures for calculating energy
        for T in temps:

            # initialize atom positions in FCC lattice
            px,py,pz = FCC.FCC(d,trans)

            # temp arrays for storing values
            UT = np.zeros(N*runs+1,dtype=float)
            ordT = np.zeros(runs+1,dtype=float)
            accT = 0

            # calculate initial order parameter of FCC lattice
            ordX = np.sum(np.cos(4*pi*px/a))
            ordY = np.sum(np.cos(4*pi*py/a))
            ordZ = np.sum(np.cos(4*pi*pz/a))
            ord0 = (ordX + ordY + ordZ)/(3.0*N)
            ordT[0] = ord0

            # initialize array to store LJ potential between atoms
            LJm = np.zeros((N,N),dtype=float)

            #----------------------------------------------------
            # get initial energy of system
            for i in range(N-1):
                for j in range(i+1,N):

                    # implement mimimum image convention
                    dX = px[i] - px[j]
                    dY = py[i] - py[j]
                    dZ = pz[i] - pz[j]

                    if dX > Lhalf: dX = dX - L
                    elif dX < -Lhalf: dX = dX + L
                    if dY > Lhalf: dY = dY - L
                    elif dY < -Lhalf: dY = dY + L
                    if dZ > Lhalf: dZ = dZ - L
                    elif dZ < -Lhalf: dZ = dZ + L

                    imgR = (dX*dX + dY*dY + dZ*dZ)**0.5

                    # implement cut & shift of potential
                    if imgR > Rc:
                        LJm[i,j] = 0.0
                        LJm[j,i] = 0.0

                    else:
                        LJm[i,j] =  4.0*(imgR**-12 - imgR**-6 - Rc**-12 + Rc**-6)
                        LJm[j,i] = 0.0
            #----------------------------------------------------

            # get initial potential energy
            pe = np.sum(LJm)
            UT[0] = pe

            # generate displacements and random numbers for this run
            indx = 0
            dx0,dy0,dz0 = RS.RandomSphere(N*runs,dr)
            rand0 = np.random.uniform(0,1,N*runs)

            # loop through atoms for averaging
            for run in range(runs):

                # loop through atoms for Monte Carlo sim
                for i in range(N):

                    # store old values for atom i
                    xi = px[i]
                    yi = py[i]
                    zi = pz[i]

                    LJRow = np.copy(LJm[i,:])
                    LJCol = np.copy(LJm[:,i])

                    # get random 3D displacement and move atom
                    xnew = xi + dx0[indx]*drr
                    ynew = yi + dy0[indx]*drr
                    znew = zi + dz0[indx]*drr

                    #----------------------------------------------------
                    # implement periodic boundary conditions
                    xnew = xnew % L
                    ynew = ynew % L
                    znew = znew % L

                    px[i] = xnew
                    py[i] = ynew
                    pz[i] = znew

                    # get new energies only between moved atom to neighbors
                    for j in range(N):
                        if i != j:

                            # implement mimimum image convention
                            dX = xnew - px[j]
                            dY = ynew - py[j]
                            dZ = znew - pz[j]

                            if dX > Lhalf: dX = dX - L
                            elif dX < -Lhalf: dX = dX + L
                            if dY > Lhalf: dY = dY - L
                            elif dY < -Lhalf: dY = dY + L
                            if dZ > Lhalf: dZ = dZ - L
                            elif dZ < -Lhalf: dZ = dZ + L

                            imgR = (dX*dX + dY*dY + dZ*dZ)**0.5

                            # implement cut & shift of potential
                            if imgR > Rc:
                                LJm[i,j] = 0.0
                                LJm[j,i] = 0.0

                            else:
                                LJm[i,j] =  4.0*(imgR**-12 - imgR**-6 - Rc**-12 + Rc**-6)
                                LJm[j,i] = 0.0
                    #----------------------------------------------------

                    # calculate new system energy and delta
                    peNew = np.sum(LJm)
                    delta = peNew - pe

                    # accept move if new energy is lower
                    if delta < 0:
                        pe = peNew
                        UT[indx+1] = pe
                        accT += 1.0
                        acc0 = accT/(indx+1)

                    else:
                        # check second condition if new energy is not lower
                        prob = np.exp(-delta/T)
                        rand = rand0[indx]

                        # accept move with this probability
                        if rand < prob:
                            pe = peNew
                            UT[indx+1] = pe
                            accT += 1.0
                            acc0 = accT/(indx+1)

                        # reject move otherwise (undo move)
                        else:
                            px[i],py[i],pz[i] = xi, yi, zi
                            LJm[i,:] = LJRow
                            LJm[:,i] = LJCol
                            UT[indx+1] = pe
                            acc0 = accT/(indx+1)

                    # check step size & increment counter
                    # want to keep acceptance between 40% & 60%
                    if acc0 < 0.4: drr = 0.5
                    if acc0 > 0.6: drr = 2.0
                    indx += 1

                # store order parameter after each pass through system
                ordX = np.sum(np.cos(4*pi*px/a))
                ordY = np.sum(np.cos(4*pi*py/a))
                ordZ = np.sum(np.cos(4*pi*pz/a))
                ord0 = (ordX + ordY + ordZ)/(3.0*N)
                ordT[run+1] = ord0

            # store average energy for each temp
            # average only last half, first half is equilibration
            aveU = np.average(UT[hlf::])
            U.append(aveU)

            # store variance of energy for each temp
            # average only last half, first half is equilibration
            var0 = np.var(UT[hlf::])
            var.append(var0)

            # store acceptance probability for each temp
            acc.append(acc0)

            # store order parameters for each temp
            order.append(ordT)

            print 'Rho = %1.2f T = %1.2f <U> = %1.4f var = %1.4f acc = %1.2f' % (Rho,T,aveU,var0,acc0)


#        # stack and store results to output file
#        #----------------------------------------------------
#        fname = '256atom_energy_Rho'+np.str(Rho)+'.txt'
#        stack = np.column_stack((temps,U))
#        np.savetxt(fname,stack,fmt='%1.12f',delimiter=' ')
#        #----------------------------------------------------
#
#        # plot results
#        #----------------------------------------------------
#        # plot potential energy vs temperature
#        label = r'$\rho$ = '+np.str(Rho)
#        ax1.plot(temps,U,'.-',label=label)
#        ax1.set_xlabel('Temperature')
#        ax1.set_ylabel('Average Energy')
#        ax1.legend(loc=2)
#
#        # plot variance vs temperature
#        label = r'$\rho$ = '+np.str(Rho)
#        ax2.plot(temps,var,'.-',label=label)
#        ax2.set_xlabel('Temperature')
#        ax2.set_ylabel('Variance')
#        ax2.legend(loc=2)
#
#        # plot acceptance probability vs temperature
#        label = r'$\rho$ = '+np.str(Rho)
#        ax3.plot(temps,acc,'.-',label=label)
#        ax3.set_xlabel('Temperature')
#        ax3.set_ylabel('Acceptance Probability')
#        ax3.legend(loc=2)
#
#        # plot order parameters for selected temperatures
#        ndxO = [0,5,10,15]
#        xOrd = range(runs+1)
#        plt.figure()
#        for i in ndxO:
#            label = '$T$ = '+np.str(temps[i])
#            plt.plot(xOrd,order[i],label=label)
#            plt.legend(loc=1)
#        plt.title(r'$\rho$ = '+np.str(Rho))
#        plt.ylabel('FCC Order')
#        plt.xlabel('MMC Step')
#        plt.savefig('256atom_order_Rho'+np.str(Rho)+'.png')
#        #----------------------------------------------------
#
        t1 = datetime.now()
        print 'runtime =',t1-t0
#
#    # save plots to files
#    fig1.savefig('256atom_energy.png')
#    fig2.savefig('256atom_variance.png')
#    fig3.savefig('256atom_acceptance.png')