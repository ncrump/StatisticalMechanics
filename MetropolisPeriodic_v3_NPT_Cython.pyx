"""
Created on Fri Mar 21 20:26:05 2014
PHYS 780, Assignment 7
Nick Crump
"""

# Metropolis Monte Carlo Simulation with Periodic Boundary Conditions
# Isobaric-Isothermal Ensemble (NPT)

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
@cython.boundscheck(False)  # turn off array bounds checking for speed
@cython.wraparound(False)   # turn off reverse indexing for speed

# function for minimum image
#----------------------------------------------------
cdef minImg(dX,dY,dZ,L):
    Lhalf = 0.5*L
    if dX > Lhalf: dX = dX - L
    elif dX < -Lhalf: dX = dX + L
    if dY > Lhalf: dY = dY - L
    elif dY < -Lhalf: dY = dY + L
    if dZ > Lhalf: dZ = dZ - L
    elif dZ < -Lhalf: dZ = dZ + L
    return (dX*dX + dY*dY + dZ*dZ)**0.5
#----------------------------------------------------

cpdef SimRun():

    t0 = datetime.now()

    # define C-type variables
    cdef float pi = 3.1415926535
    cdef float dr,drr,d,a,L,Lhalf
    cdef float dv,dvv,v0,dvMax,VNew,LNew
    cdef float scale,deltaU,deltaV
    cdef float xi,yi,zi,xnew,ynew,znew
    cdef float dX,dY,dZ,rand
    cdef float Rc,RcNew,Rc6New,Rc12New
    cdef float imgR6,imgR12,Rc6,Rc12
    cdef float pe,peNew,peV,T,expArg
    cdef float aveU,varU,lnVNew
    cdef float accU,accU0,accV,accV0
    cdef float aveRunU,aveOldU,varRunU
    cdef float D,aveRunD,aveOldD,varRunD
    cdef int N,runs,trans,hlf,hlfRun,nRho,nPrs
    cdef int i,j,k,p,run,indx

    # define numpy C-type arrays
    cdef np.ndarray[dtype=double,ndim=1] dens,pres
    cdef np.ndarray[dtype=double,ndim=1] d_rho,a_rho,Rc_rho,L_rho
    cdef np.ndarray[dtype=double,ndim=1] px,py,pz,px0,py0,pz0
    cdef np.ndarray[dtype=double,ndim=1] pxVi,pyVi,pzVi
    cdef np.ndarray[dtype=double,ndim=1] dx0,dy0,dz0,rand0,rand1,randv
    cdef np.ndarray[dtype=double,ndim=1] UP,aveRunUarr,varRunUarr
    cdef np.ndarray[dtype=double,ndim=1] DP,aveRunDarr,varRunDarr
    cdef np.ndarray[dtype=double,ndim=1] LJRow,LJCol
    cdef np.ndarray[dtype=double,ndim=2] LJm,LJv

    #----------------------------------------------------
    # define initial simulation values
    dr = 0.04                      # initial atom step size
    dv = 0.1                       # initial volume step
    runs = 10000                   # number of runs

    # define FCC lattice size (this specifies number of atoms N)
    trans = 3                      # translations of 4-atom FCC unit cell

    # define densities and pressures
    dens = np.array([0.9])
    pres = np.array([1.0,10.0,20.0])
    T = 1.0
    nRho = len(dens)
    nPrs = len(pres)

    d_rho = 1.12246*dens**-0.33333  # distance between atoms
    a_rho = d_rho*(2**0.5)          # side length of FCC unit cube
    L_rho = (trans+1)*a_rho         # side length of boundary box
    Rc_rho = 0.98*0.5*L_rho         # cutoff distance bewteen atoms
    #----------------------------------------------------

    # print number of atoms N in FCC lattice
    N = 4*(trans+1)**3
    print 'number of atoms = ',N

    for k in range(nRho):

        # define arrays to store results
        aveUarr = []                   # potential energy
        varUarr = []                   # variance of energy
        accUarr = []                   # acceptance probability

        # define parameters for this density
        Rho = np.float(dens[k])
        d = d_rho[k]
        a = a_rho[k]
        L = L_rho[k]
        Rc = Rc_rho[k]

        Rc6 = 1.0/(Rc*Rc*Rc*Rc*Rc*Rc)
        Rc12 = Rc6*Rc6

        hlf = int(0.5*N*runs)
        hlfRun = int(0.5*runs)
        v0 = L**3
        dvMax = 0.5*dv
        dvv = 1.0
        drr = 1.0

        print 'initial V = %1.2f' % v0

        # loop through pressures for calculating energy
        for p in range(nPrs):
            P = pres[p]

            # temp arrays for storing values
            UP = np.zeros(hlfRun,dtype=float)          # accepted energy
            DP = np.zeros(hlfRun,dtype=float)          # accepted diplacement
            aveRunUarr = np.zeros(hlfRun,dtype=float)  # running avg energy
            varRunUarr = np.zeros(hlfRun,dtype=float)  # running var energy
            aveRunDarr = np.zeros(hlfRun,dtype=float)  # running avg displacement
            varRunDarr = np.zeros(hlfRun,dtype=float)  # running var displacement

            # initialize atom positions in FCC lattice
            px,py,pz = FCC.FCC(d,trans)

            # initialize array to store LJ potential between atoms
            LJm = np.zeros((N,N),dtype=float)
            LJv = np.zeros((N,N),dtype=float)

            # get initial energy of system
            #----------------------------------------------------
            for i in range(N-1):
                for j in range(i+1,N):

                    # implement mimimum image convention
                    dX = px[i] - px[j]
                    dY = py[i] - py[j]
                    dZ = pz[i] - pz[j]
                    imgR = minImg(dX,dY,dZ,L)

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
            accU0 = 0.0
            accV0 = 0.0
            indx = 0

            # generate displacements and random numbers for this run
            dx0,dy0,dz0 = RS.RandomSphere(N*runs,dr)
            rand0 = np.random.uniform(0,1,N*runs)
            rand1 = np.random.uniform(0,1,runs)
            randv = np.random.uniform(-dvMax,dvMax,runs)

            # loop through atoms for averaging
            for run in range(runs):


                # THIS MOVES ATOMS IN LATTICE
                # **************************************************
                # loop through atoms for Monte Carlo sim
                for i in range(N):
                    LJsum = 0

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
                            imgR = minImg(dX,dY,dZ,L)

                            # implement cut & shift of potential
                            if imgR > Rc:
                                LJm[i,j] = 0.0
                                LJm[j,i] = 0.0

                            else:
                                imgR6 = 1.0/(imgR*imgR*imgR*imgR*imgR*imgR)
                                imgR12 = imgR6*imgR6
                                LJm[i,j] = 4.0*(imgR12 - imgR6 - Rc12 + Rc6)
                                LJm[j,i] = 0.0

                    # calculate new system energy and delta
                    peNew = np.sum(LJm)
                    deltaU = peNew - pe

                    # accept move if new energy is lower
                    if deltaU < 0:
                        pe = peNew
                        accU0 += 1.0
                    else:
                        # check second condition if new energy is not lower
                        prob = np.exp(-deltaU/T)
                        rand = rand0[indx]

                        # accept move with this probability
                        if rand < prob:
                            pe = peNew
                            accU0 += 1.0

                        # reject move otherwise (undo move)
                        else:
                            px[i],py[i],pz[i] = xi, yi, zi
                            LJm[i,:] = LJRow
                            LJm[:,i] = LJCol

                    # store energy
                    if run >= hlfRun:
                        UP[run-hlfRun] = pe

                    # update acceptance rate and increment counter
                    accU = accU0/(indx+1)
                    indx += 1
                # **************************************************
                # END MOVING ATOMS IN LATTICE


                # adjust step to keep acceptance between 40%-60%
                if run % 100 == 0:
                    if accU < 0.4: drr = 0.5
                    if accU > 0.6: drr = 2.0

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
                    DP[run-hlfRun] = D

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


                # THIS MOVES VOLUME OF BOX
                # **************************************************
                # move volume after each cycle through lattice

                # get new volume and box length
                lnVNew = np.log(v0) + randv[run]*dvv
                VNew = np.exp(lnVNew)
                LNew = VNew**0.33333
                scale = LNew/L

                RcNew = Rc*scale
                Rc6New = 1.0/(RcNew*RcNew*RcNew*RcNew*RcNew*RcNew)
                Rc12New = Rc6New*Rc6New

                # store old atom positions
                pxVi = np.copy(px)
                pyVi = np.copy(py)
                pzVi = np.copy(pz)

                # scale new atom positions
                px = px*scale
                py = py*scale
                pz = pz*scale

                #----------------------------------------------------
                # calculate new energy after volume move
                for i in range(N-1):
                    for j in range(i+1,N):

                        # implement mimimum image convention
                        dX = px[i] - px[j]
                        dY = py[i] - py[j]
                        dZ = pz[i] - pz[j]
                        imgR = minImg(dX,dY,dZ,LNew)

                        # implement cut & shift of potential
                        if imgR > RcNew:
                            LJv[i,j] = 0.0

                        else:
                            imgR6 = 1.0/(imgR*imgR*imgR*imgR*imgR*imgR)
                            imgR12 = imgR6*imgR6
                            LJv[i,j] = 4.0*(imgR12 - imgR6 - Rc12New + Rc6New)
                #----------------------------------------------------

                # calculate new system energy and delta
                peV = np.sum(LJv)
                deltaU = peV - pe
                deltaV = VNew - v0

                # accept move if new energy is lower
                if deltaU < 0:
                    pe = peNew
                    accV0 += 1.0
                    L = LNew
                    v0 = VNew
                    Rc = RcNew
                    Rc6 = Rc6New
                    Rc12 = Rc12New

                else:
                    # check second condition if new energy is not lower
                    expArg = (deltaU + P*deltaV - T*(N+1)*np.log(VNew/v0))/T
                    prob = np.exp(-expArg)
                    rand = rand1[run]

                    # accept move with this probability
                    if rand < prob:
                        pe = peNew
                        accV0 += 1.0
                        L = LNew
                        v0 = VNew
                        Rc = RcNew
                        Rc6 = Rc6New
                        Rc12 = Rc12New

                    # reject move otherwise (undo move)
                    else:
                        px = pxVi
                        py = pyVi
                        pz = pzVi

                # update volume acceptance rate
                accV = accV0/(run+1)

                # adjust step to keep acceptance between 40%-60%
                if run % 100 == 0:
                    if accV < 0.4: dvv = 0.5
                    if accV > 0.6: dvv = 2.0
                # **************************************************
                # END MOVING VOLUME OF BOX


            # store avg energy and variance for each pressure
            # average only last half, first half is equilibration
            aveU = aveRunU
            varU = varRunU/(run-hlfRun+1)
            aveUarr.append(aveU)
            varUarr.append(varU)
            accUarr.append(accU)

            print 'Rho = %1.2f T = %1.2f P = %1.2f <U> = %1.2f varU = %1.2f accU = %1.2f accV = %1.2f final V = %1.2f' \
            % (Rho,T,P,aveU,varU,accU,accV,v0)

            # plot results
            #----------------------------------------------------
            # set plot axis for last half of runs
            xRuns = np.arange(hlfRun)+hlfRun

            # plot running average potential energy
            title1 = 'Energy, '+r'$\rho$ = '+np.str(Rho)+', $P$ = '+np.str(P)+', $T$ = '+np.str(T)+', NPT'
            fname1 = np.str(N)+'atom_RunAveU_Rho'+np.str(Rho)+'_P'+np.str(P)+'_T'+np.str(T)+'_NPT.png'
            plt.figure()
            plt.subplot(211)
            plt.plot(xRuns,UP,'b.',label='$Accepted$')
            plt.plot(xRuns,aveRunUarr,'r.-')
            plt.ylabel('Running Average')
            plt.title(title1)
            plt.legend(loc=2)
            plt.subplot(212)
            plt.plot(xRuns,varRunUarr**0.5,'g.-')
            plt.ylabel('Running Std Deviation')
            plt.xlabel('MMC Step')
            plt.savefig(fname1)

            # plot running average mean squared displacement
            title2 = 'MSD, '+r'$\rho$ = '+np.str(Rho)+', $P$ = '+np.str(P)+', $T$ = '+np.str(T)+', NPT'
            fname2 = np.str(N)+'atom_RunAveD_Rho'+np.str(Rho)+'_P'+np.str(P)+'_T'+np.str(T)+'_NPT.png'
            plt.figure()
            plt.subplot(211)
            plt.plot(xRuns,DP,'b.',label='$Accepted$')
            plt.plot(xRuns,aveRunDarr,'r.-')
            plt.ylabel('Running Average')
            plt.title(title2)
            plt.legend(loc=2)
            plt.subplot(212)
            plt.plot(xRuns,varRunDarr**0.5,'g.-')
            plt.ylabel('Running Std Deviation')
            plt.xlabel('MMC Step')
            plt.savefig(fname2)
            #----------------------------------------------------

        # plot potential energy, variance, acceptance rate
        title3 = r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NPT'
        fname3 = np.str(N)+'atom_energy_var_acc_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NPT.png'
        plt.figure()
        plt.subplot(311)
        plt.plot(pres,aveUarr,'b.-')
        plt.ylabel('Average Energy')
        plt.title(title3)
        plt.subplot(312)
        plt.plot(pres,varUarr,'r.-')
        plt.ylabel('Variance')
        plt.subplot(313)
        plt.plot(pres,accUarr,'g.-')
        plt.ylabel('Acceptance')
        plt.xlabel('Pressure')
        plt.savefig(fname3)

        # stack and store results to output file
        #----------------------------------------------------
        fname = np.str(N)+'atom_energy_var_acc_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NPT.txt'
        stack = np.column_stack((pres,aveUarr,varUarr,accUarr))
        np.savetxt(fname,stack,fmt='%1.12f',delimiter=' ')
        #----------------------------------------------------

        print 'runtime =',datetime.now()-t0