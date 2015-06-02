"""
Created on Fri Mar 21 20:26:05 2014
PHYS 780, Assignment 7
Nick Crump
"""

# Metropolis Monte Carlo Simulation with Periodic Boundary Conditions
# Canonical Ensemble (NVT)

# v3 new calculations & running averages
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

    # define C-type variables
    cdef float pi = 3.1415926535
    cdef float dr,drr,d,a,L,Lhalf
    cdef float xi,yi,zi,xnew,ynew,znew
    cdef float dX,dY,dZ,rand
    cdef float imgR6,imgR12,Rc6,Rc12
    cdef float pe,peNew,delta,T
    cdef float aveU,varU,accP,accT,Ucorr
    cdef float aveRunU,aveOldU,varRunU
    cdef float D,aveRunD,aveOldD,varRunD
    cdef int N,runs,trans,hlf,hlfRun,nRho,nTmp
    cdef int i,j,k,t,run,indx

    # define numpy C-type arrays
    cdef np.ndarray[dtype=double,ndim=1] dens,temps
    cdef np.ndarray[dtype=double,ndim=1] d_rho,a_rho,Rc_rho
    cdef np.ndarray[dtype=double,ndim=1] L_rho,Lhalf_rho
    cdef np.ndarray[dtype=double,ndim=1] px,py,pz,px0,py0,pz0
    cdef np.ndarray[dtype=double,ndim=1] dx0,dy0,dz0,rand0
    cdef np.ndarray[dtype=double,ndim=1] UT,aveRunUarr,varRunUarr
    cdef np.ndarray[dtype=double,ndim=1] DT,aveRunDarr,varRunDarr
    cdef np.ndarray[dtype=double,ndim=1] LJRow,LJCol
    cdef np.ndarray[dtype=double,ndim=2] LJm

    #----------------------------------------------------
    # define initial simulation values
    dr = 0.04                      # initial step size
    runs = 1000                   # number of runs

    # define FCC lattice size (this specifies number of atoms N)
    trans = 3                      # translations of 4-atom FCC unit cell

    # define densities and temperatures
    dens = np.array([0.9])
    temps = np.array([0.5,1.0,1.5,2.0])
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
    print 'number of atoms = ',N

    for k in range(nRho):

        # define arrays to store results
        aveUarr = []                   # potential energy
        varUarr = []                   # variance of energy
        accParr = []                   # acceptance probability

        # define parameters for this density
        Rho = np.float(dens[k])
        d = d_rho[k]
        a = a_rho[k]
        L = L_rho[k]
        Rc = Rc_rho[k]
        Lhalf = Lhalf_rho[k]

        Ucorr = 2.66666*pi*Rho*(0.33333*Rc**-9 - Rc**-3)
        Rc6 = 1.0/(Rc*Rc*Rc*Rc*Rc*Rc)
        Rc12 = Rc6*Rc6
        hlf = int(0.5*N*runs)
        hlfRun = int(0.5*runs)
        drr = 1.0

        # loop through temperatures for calculating energy
        for t in range(nTmp):
            T = temps[t]

            # temp arrays for storing values
            UT = np.zeros(hlfRun,dtype=float)          # accepted energy
            DT = np.zeros(hlfRun,dtype=float)          # accepted diplacement
            aveRunUarr = np.zeros(hlfRun,dtype=float)  # running avg energy
            varRunUarr = np.zeros(hlfRun,dtype=float)  # running var energy
            aveRunDarr = np.zeros(hlfRun,dtype=float)  # running avg displacement
            varRunDarr = np.zeros(hlfRun,dtype=float)  # running var displacement

            # initialize atom positions in FCC lattice
            px,py,pz = FCC.FCC(d,trans)

            # initialize matrix to store LJ potential between atoms
            LJm = np.zeros((N,N),dtype=float)

            # get initial energy of system
            #----------------------------------------------------
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

                    else:
                        imgR6 = 1.0/(imgR*imgR*imgR*imgR*imgR*imgR)
                        imgR12 = imgR6*imgR6
                        LJm[i,j] = 4.0*(imgR12 - imgR6 - Rc12 + Rc6)
            #----------------------------------------------------

            # get initial potential energy
            pe = np.sum(LJm)

            # initialize running averages
            aveRunU = pe
            varRunU = 0.0
            aveRunD = 0.0
            varRunD = 0.0
            accT = 0.0
            indx = 0

            # generate displacements and random numbers for this run
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

                    # calculate new energy
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
                                imgR6 = 1.0/(imgR*imgR*imgR*imgR*imgR*imgR)
                                imgR12 = imgR6*imgR6
                                LJm[i,j] = 4.0*(imgR12 - imgR6 - Rc12 + Rc6)
                                LJm[j,i] = 0.0
                    #----------------------------------------------------

                    # calculate new system energy and delta
                    peNew = np.sum(LJm)
                    delta = peNew - pe

                    # accept move if new energy is lower
                    if delta < 0:
                        pe = peNew
                        accT += 1.0

                    else:
                        # check second condition if new energy is not lower
                        prob = np.exp(-delta/T)
                        rand = rand0[indx]

                        # accept move with this probability
                        if rand < prob:
                            pe = peNew
                            accT += 1.0

                        # reject move otherwise (undo move)
                        else:
                            px[i],py[i],pz[i] = xi, yi, zi
                            LJm[i,:] = LJRow
                            LJm[:,i] = LJCol

                    # store energy
                    if run >= hlfRun:
                        UT[run-hlfRun] = pe

                    # update acceptance rate and increment counter
                    accP = accT/(indx+1)
                    indx += 1

                # adjust step to keep acceptance between 40%-60%
                if run % 1000 == 0:
                    if accP < 0.4: drr = 0.5
                    if accP > 0.6: drr = 2.0

                # keep initial positions to calculate mean displacement
                if run == hlfRun:
                    px0 = np.copy(px)
                    py0 = np.copy(py)
                    pz0 = np.copy(pz)

                # calculate MSD, running averages and variances
                #----------------------------------------------------
                if run >= hlfRun:
                    # calculate mean square displacement
                    D = np.sum((px-px0)**2 + (py-py0)**2 + (pz-pz0)**2)/N
                    DT[run-hlfRun] = D

                    # update running averages
                    aveOldU = aveRunU
                    aveOldD = aveRunD
                    aveRunU = aveRunU + (pe-aveRunU)/(run-hlfRun+1)
                    varRunU = varRunU + (pe-aveOldU)*(pe-aveRunU)
                    aveRunD = aveRunD + ((D-aveRunD)/(run-hlfRun+1))
                    varRunD = varRunD + (D-aveOldD)*(D-aveRunD)

                    # store running averages
                    aveRunUarr[run-hlfRun] = aveRunU
                    varRunUarr[run-hlfRun] = varRunU/(run-hlfRun+1)
                    aveRunDarr[run-hlfRun] = aveRunD
                    varRunDarr[run-hlfRun] = varRunD/(run-hlfRun+1)
                #----------------------------------------------------

            # store avg energy and variance for each temp
            # average only last half, first half is equilibration
            aveU = aveRunU
            varU = varRunU/(run-hlfRun+1)
            aveUarr.append(aveU)
            varUarr.append(varU)
            accParr.append(accP)

            print 'Rho = %1.2f T = %1.2f <U> = %1.2f Ucorr = %.2f varU = %1.2f accP = %1.2f' \
            % (Rho,T,aveU,Ucorr,varU,accP)

#            # plot results
#            #----------------------------------------------------
#            # set plot axis for last half of runs
#            xRuns = np.arange(hlfRun)+hlfRun
#
#            # plot running average potential energy
#            title1 = 'Energy, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
#            fname1 = np.str(N)+'atom_RunAveU_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
#            plt.figure()
#            plt.subplot(211)
#            plt.plot(xRuns,UT,'b.',label='$Accepted$')
#            plt.plot(xRuns,aveRunUarr,'r.-')
#            plt.ylabel('Running Average')
#            plt.title(title1)
#            plt.legend(loc=2)
#            plt.subplot(212)
#            plt.plot(xRuns,varRunUarr**0.5,'g.-')
#            plt.ylabel('Running Std Deviation')
#            plt.xlabel('MMC Step')
#            plt.savefig(fname1)
#
#            # plot running average mean squared displacement
#            title2 = 'MSD, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
#            fname2 = np.str(N)+'atom_RunAveD_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
#            plt.figure()
#            plt.subplot(211)
#            plt.plot(xRuns,DT,'b.',label='$Accepted$')
#            plt.plot(xRuns,aveRunDarr,'r.-')
#            plt.ylabel('Running Average')
#            plt.title(title2)
#            plt.legend(loc=2)
#            plt.subplot(212)
#            plt.plot(xRuns,varRunDarr**0.5,'g.-')
#            plt.ylabel('Running Std Deviation')
#            plt.xlabel('MMC Step')
#            plt.savefig(fname2)
#            #----------------------------------------------------
#
#        # plot potential energy, variance, acceptance rate
#        title3 = r'$\rho$ = '+np.str(Rho)+', NVT'
#        fname3 = np.str(N)+'atom_energy_var_acc_Rho'+np.str(Rho)+'_NVT.png'
#        plt.figure()
#        plt.subplot(311)
#        plt.plot(temps,aveUarr,'b.-')
#        plt.ylabel('Average Energy')
#        plt.title(title3)
#        plt.subplot(312)
#        plt.plot(temps,varUarr,'r.-')
#        plt.ylabel('Variance')
#        plt.subplot(313)
#        plt.plot(temps,accParr,'g.-')
#        plt.ylabel('Acceptance')
#        plt.xlabel('Temperature')
#        plt.savefig(fname3)
#
#        # stack and store results to output file
#        #----------------------------------------------------
#        fname = np.str(N)+'atom_energy_var_acc_Rho'+np.str(Rho)+'_NVT.txt'
#        stack = np.column_stack((temps,aveUarr,varUarr,accParr))
#        np.savetxt(fname,stack,fmt='%1.12f',delimiter=' ')
#        #----------------------------------------------------

        print 'runtime =',datetime.now()-t0