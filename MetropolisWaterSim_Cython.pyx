"""
Created on Wed Apr 23 08:55:46 2014
PHYS 780, Final Project
Nick Crump
"""

# Metropolis Monte Carlo Simulation with Periodic Boundary Conditions
# Canonical Ensemble (NVT)
"""
Simulation of 3-site SPC water molecules with trial translations and rotations
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
    cdef float dr,d,a,L,Lhalf,sig,kc,kb,beta
    cdef float Lbond,Abond,ALJ,BLJ,qO,qH,qOO,qOH,qHH
    cdef float OO,OH1,OH2,H1O,H1H1,H1H2,H2O,H2H1,H2H2
    cdef float q0,q1,q2,q3,sq0,sq1,sq2,sq3,rand,s1,s2
    cdef float magvH1,magvH2,dxr,dyr,dzr
    cdef float Oxnew,Oynew,Oznew
    cdef float H1xnew,H1ynew,H1znew
    cdef float H2xnew,H2ynew,H2znew
    cdef float ordX,ordY,ordZ,ord0
    cdef float dXO,dYO,dZO,dXH,dYH,dZH
    cdef float imgR6,imgR12,Rc6,Rc12
    cdef float pe,peNew,delta,Gdr,LJE,CoulE
    cdef float aveU,varU,accP,accT,Cv
    cdef float aveRunU,aveOldU,varRunU
    cdef float D,aveRunD,aveOldD,varRunD
    cdef int N,runs,trans,hlf,hlfRun
    cdef int i,j,run,indx

    # define numpy C-type arrays
    cdef np.ndarray[dtype=double,ndim=1] Ox,Oy,Oz,Ox0,Oy0,Oz0
    cdef np.ndarray[dtype=double,ndim=1] UT,aveRunUarr,varRunUarr
    cdef np.ndarray[dtype=double,ndim=1] DT,aveRunDarr,varRunDarr
    cdef np.ndarray[dtype=double,ndim=1] H1x,H1y,H1z
    cdef np.ndarray[dtype=double,ndim=1] H2x,H2y,H2z
    cdef np.ndarray[dtype=double,ndim=1] Oi,H1i,H2i
    cdef np.ndarray[dtype=double,ndim=1] vH1,vH2,uH1,uH2
    cdef np.ndarray[dtype=double,ndim=1] R1,R2,R3
    cdef np.ndarray[dtype=double,ndim=1] dx0,dy0,dz0,rand0
    cdef np.ndarray[dtype=double,ndim=1] r1,r2,q0A,q1A,q2A,q3A
    cdef np.ndarray[dtype=double,ndim=1] sq0A,sq1A,sq2A,sq3A
    cdef np.ndarray[dtype=double,ndim=1] HHx,HHy,HHz
    cdef np.ndarray[dtype=double,ndim=1] GROO,GRHH,GROH
    cdef np.ndarray[dtype=double,ndim=1] RadOO,RadHH,RadOH
    cdef np.ndarray[dtype=double,ndim=1] GRbins,binOO,binHH,binOH
    cdef np.ndarray[dtype=double,ndim=1] ERow,ECol,Ord
    cdef np.ndarray[dtype=double,ndim=2] E,Rot,Vmol,Vlab
    cdef np.ndarray[dtype=long,ndim=1] histOO,histHH,histOH

    #----------------------------------------------------
    # define initial simulation values
    dr = 0.15                 # displacement step (angstrom)
    runs = 100000             # number of runs

    # define FCC lattice size
    trans = 1                  # translations of 4-atom FCC unit cell

    # define number density and temperature
    Rho = 0.03343              # (atoms/angstrom)
    T = 300                    # (kelvin)

    # define lattice parameters
    d = 1.12246*Rho**-0.33333  # distance between oxygen atoms (angstrom)
    a = d*(2**0.5)             # side length of FCC unit cube  (angstrom)
    L = (trans+1)*a            # side length of boundary box   (angstrom)
    Rc = 0.98*0.5*L            # cutoff distance bewteen atoms (angstrom)
    Lhalf = 0.5*L              # half length of boundary box   (angstrom)

    # define water molecule bond length & bond angle
    Lbond = 1.0                # (angstrom)
    Abond = 109.47*(pi/180)    # (radians)

    # define water molecule interaction parameters
    ALJ  = 629.4e3             # LJ term (kcal*A12/mol)
    BLJ  = 625.5               # LJ term (kcal*A6/mol)
    qO   = -0.8476             # oxygen partial charge (e)
    qH   = 0.4238              # hydrogen partial charge (e)
    kc   = 332.064             # Coulomb constant (ang*kcal/mol)
    kb   = 0.001987            # Boltzmann constant (kcal/mol K)
    beta = 1.0/(kb*T)          # Boltzmann factor (mol/kcal)
    sig  = 1.58                # radius of molecule (angstrom)
    #----------------------------------------------------

    # print number of molecules N in FCC lattice
    N = 4*(trans+1)**3
    print 'number of water molecules = ',N

    # define radius bins for g(r)
    Gdr = 0.06
    GRbins = np.arange(Gdr,Lhalf,Gdr)
    Nbins = len(GRbins)-1

    # define constants
    Rc6 = 1.0/(Rc*Rc*Rc*Rc*Rc*Rc)
    Rc12 = Rc6*Rc6
    hlf = int(0.5*N*runs)
    hlfRun = int(0.5*runs)
    qOO = qO*qO
    qOH = qO*qH
    qHH = qH*qH

    # define arrays to store results
    aveUarr = []                               # potential energy
    varUarr = []                               # variance of energy
    accParr = []                               # acceptance probability
    RadOO = np.zeros(N,dtype=float)            # radius array for gOO
    RadHH = np.zeros(2*N,dtype=float)          # radius array for gHH
    RadOH = np.zeros(2*N,dtype=float)          # radius array for gOH
    UT = np.zeros(hlfRun,dtype=float)          # accepted energy
    DT = np.zeros(hlfRun,dtype=float)          # accepted diplacement
    Ord = np.zeros(runs+1,dtype=float)         # FCC order parameter
    q0A = np.zeros(N*runs,dtype=float)         # unit quaternion parameter
    q1A = np.zeros(N*runs,dtype=float)         # unit quaternion parameter
    q2A = np.zeros(N*runs,dtype=float)         # unit quaternion parameter
    q3A = np.zeros(N*runs,dtype=float)         # unit quaternion parameter
    GROO = np.zeros(Nbins,dtype=float)         # g(r) for OO
    GRHH = np.zeros(Nbins,dtype=float)         # g(r) for HH
    GROH = np.zeros(Nbins,dtype=float)         # g(r) for OH
    aveRunUarr = np.zeros(hlfRun,dtype=float)  # running avg energy
    varRunUarr = np.zeros(hlfRun,dtype=float)  # running var energy
    aveRunDarr = np.zeros(hlfRun,dtype=float)  # running avg displacement
    varRunDarr = np.zeros(hlfRun,dtype=float)  # running var displacement

    # initialize oxygen atom positions in FCC lattice
    Ox,Oy,Oz = FCC.FCC(d,trans)

#    # simple 4-site FCC unit cell test
#    Ox = np.array([0,0.5*a,0,0.5*a])
#    Oy = np.array([0,0.5*a,0.5*a,0])
#    Oz = np.array([0,0,0.5*a,0.5*a])

    # initialize hydrogen 1 positions
    H1x = Ox + Lbond*np.cos(0.5*Abond)
    H1y = Oy + Lbond*np.sin(0.5*Abond)
    H1z = np.copy(Oz)

    # initialize hydrogen 2 positions
    H2x = Ox + Lbond*np.cos(0.5*Abond)
    H2y = Oy - Lbond*np.sin(0.5*Abond)
    H2z = np.copy(Oz)

    # stack and store initial atom positions for import to Avogadro
    #----------------------------------------------------
    O  = np.ones(len(Ox))*8
    H1 = np.ones(len(H1x))*1
    H2 = np.ones(len(H2x))*1
    nums = np.concatenate([O,H1,H2])
    molx = np.concatenate([Ox,H1x,H2x])
    moly = np.concatenate([Oy,H1y,H2y])
    molz = np.concatenate([Oz,H1z,H2z])
    stack = np.column_stack((nums,molx,moly,molz))
    np.savetxt('H2O_'+str(N)+'_InitialPos.txt',stack,fmt=('%i','%f4','%f4','%f4'))
    #----------------------------------------------------

    # initialize energy matrix to store potential between molecules
    E = np.zeros((N,N),dtype=float)

    # get initial energy of system
    #----------------------------------------------------
    for i in range(N-1):
        for j in range(i+1,N):

            # get oxygen-oxygen distances for LJ
            dXO = Ox[i] - Ox[j]
            dYO = Oy[i] - Oy[j]
            dZO = Oz[i] - Oz[j]

            # get all site distances for Coulomb
            OO   = (dXO*dXO + dYO*dYO + dZO*dZO)**0.5
            OH1  = ((Ox[i]-H1x[j])**2 + (Oy[i]-H1y[j])**2 + (Oz[i]-H1z[j])**2)**0.5
            OH2  = ((Ox[i]-H2x[j])**2 + (Oy[i]-H2y[j])**2 + (Oz[i]-H2z[j])**2)**0.5
            H1O  = ((H1x[i]-Ox[j])**2 + (H1y[i]-Oy[j])**2 + (H1z[i]-Oz[j])**2)**0.5
            H1H1 = ((H1x[i]-H1x[j])**2 + (H1y[i]-H1y[j])**2 + (H1z[i]-H1z[j])**2)**0.5
            H1H2 = ((H1x[i]-H2x[j])**2 + (H1y[i]-H2y[j])**2 + (H1z[i]-H2z[j])**2)**0.5
            H2O  = ((H2x[i]-Ox[j])**2 + (H2y[i]-Oy[j])**2 + (H2z[i]-Oz[j])**2)**0.5
            H2H1 = ((H2x[i]-H1x[j])**2 + (H2y[i]-H1y[j])**2 + (H2z[i]-H1z[j])**2)**0.5
            H2H2 = ((H2x[i]-H2x[j])**2 + (H2y[i]-H2y[j])**2 + (H2z[i]-H2z[j])**2)**0.5

            # get Coulomb potential
            CoulE = (qOO/OO)+(qOH/OH1)+(qOH/OH2)+(qOH/H1O)+(qHH/H1H1)+(qHH/H1H2)+(qOH/H2O)+(qHH/H2H1)+(qHH/H2H2)

            # implement mimimum image convention only on oxygen (LJ)
            if dXO > Lhalf: dXO = dXO - L
            elif dXO < -Lhalf: dXO = dXO + L
            if dYO > Lhalf: dYO = dYO - L
            elif dYO < -Lhalf: dYO = dYO + L
            if dZO > Lhalf: dZO = dZO - L
            elif dZO < -Lhalf: dZO = dZO + L
            imgR = (dXO*dXO + dYO*dYO + dZO*dZO)**0.5

            # implement cut & shift of potential only on oxygen (LJ)
            if imgR > Rc:
                LJE = 0.0

            else:
                imgR6 = 1.0/(imgR*imgR*imgR*imgR*imgR*imgR)
                imgR12 = imgR6*imgR6
                LJE = (ALJ*imgR12) - (BLJ*imgR6) - (ALJ*Rc12) + (BLJ*Rc6)

            # write potential to energy matrix
            E[i,j] = kc*CoulE + LJE
    #----------------------------------------------------

    # get initial potential energy
    pe = np.sum(E)

    # initialize running averages
    aveRunU = pe
    varRunU = 0.0
    aveRunD = 0.0
    varRunD = 0.0
    accT = 0.0
    indx = 0

    # get initial order parameter
    ordX = np.sum(np.cos(4*pi*Ox/a))
    ordY = np.sum(np.cos(4*pi*Oy/a))
    ordZ = np.sum(np.cos(4*pi*Oz/a))
    ord0 = (ordX + ordY + ordZ)/(3.0*N)
    Ord[0] = ord0

    # generate displacements and random numbers for this run
    dx0,dy0,dz0 = RS.RandomSphere(N*runs,dr)
    rand0 = np.random.uniform(0,1,N*runs)

    # generate random numbers for quaternion parameters
    #----------------------------------------------------
    for i in range(N*runs):
        s1 = 1
        s2 = 1

        # get first set of random numbers
        while s1 >= 1:
            r1 = np.random.uniform(-1,1,2)
            s1 = np.sum(r1**2)

        # get second set of random numbers
        while s2 >= 1:
            r2 = np.random.uniform(-1,1,2)
            s2 = np.sum(r2**2)

        # get unit quaternion parameters
        z = ((1-s1)/s2)**0.5
        q0A[i] = r1[0]
        q1A[i] = r1[1]
        q2A[i] = r2[0]*z
        q3A[i] = r2[1]*z

    # get squared unit quaternion parameters
    sq0A = q0A**2
    sq1A = q1A**2
    sq2A = q2A**2
    sq3A = q3A**2
    #----------------------------------------------------

    # loop through atoms for averaging
    for run in range(runs):

        # loop through atoms for Monte Carlo sim
        for i in range(N):

            # store old values for molecule i
            Oi  = np.array([Ox[i],Oy[i],Oz[i]],dtype=float)
            H1i = np.array([H1x[i],H1y[i],H1z[i]],dtype=float)
            H2i = np.array([H2x[i],H2y[i],H2z[i]],dtype=float)

            # store old values from energy matrix
            ERow = np.copy(E[i,:])
            ECol = np.copy(E[:,i])

            # get molecule orientation unit vectors to hydrogens
            vH1 = H1i-Oi
            vH2 = H2i-Oi
            magvH1 = (np.sum(vH1**2))**0.5
            magvH2 = (np.sum(vH2**2))**0.5
            uH1 = vH1/magvH1
            uH2 = vH2/magvH2

            # construct orientation matrix of hydrogen unit vectors
            Vmol = np.matrix(([uH1[0],uH2[0]],[uH1[1],uH2[1]],[uH1[2],uH2[2]]),dtype=float)

            # get random trial rotation from unit quaternion
            q0 = q0A[indx]
            q1 = q1A[indx]
            q2 = q2A[indx]
            q3 = q3A[indx]
            sq0 = sq0A[indx]
            sq1 = sq1A[indx]
            sq2 = sq2A[indx]
            sq3 = sq3A[indx]

            # form unit quaternion rotation matrix
            R1 = np.array([sq0+sq1-sq2-sq3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],dtype=float)
            R2 = np.array([2*(q1*q2+q0*q3), sq0-sq1+sq2-sq3, 2*(q2*q3-q0*q1)],dtype=float)
            R3 = np.array([2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), sq0-sq1-sq2+sq3],dtype=float)
            Rot = np.matrix(([R1,R2,R3]),dtype=float)

            # apply rotation matrix to molecule orientation unit vectors
            Vlab = Rot*Vmol

            # get random trial translation of molecule
            dxr = dx0[indx]
            dyr = dy0[indx]
            dzr = dz0[indx]

            # get new translated oxygen positions
            Oxnew = Oi[0] + dxr
            Oynew = Oi[1] + dyr
            Oznew = Oi[2] + dzr

            # get new rotated and translated hydrogen positions
            H1xnew = Vlab[0,0] + Oi[0] + dxr
            H1ynew = Vlab[1,0] + Oi[1] + dyr
            H1znew = Vlab[2,0] + Oi[2] + dzr
            H2xnew = Vlab[0,1] + Oi[0] + dxr
            H2ynew = Vlab[1,1] + Oi[1] + dyr
            H2znew = Vlab[2,1] + Oi[2] + dzr

            # calculate new energy
            #----------------------------------------------------
            # implement periodic boundary conditions on molecule
            if Oxnew+sig > L:
                Oxnew  = Oxnew - L
                H1xnew = H1xnew - L
                H2xnew = H2xnew - L
            if Oxnew+sig < L:
                Oxnew  = Oxnew + L
                H1xnew = H1xnew + L
                H2xnew = H2xnew + L
            if Oynew+sig > L:
                Oynew  = Oynew - L
                H1ynew = H1ynew - L
                H2ynew = H2ynew - L
            if Oynew+sig < L:
                Oynew  = Oynew + L
                H1ynew = H1ynew + L
                H2ynew = H2ynew + L
            if Oznew+sig > L:
                Oznew  = Oznew - L
                H1znew = H1znew - L
                H2znew = H2znew - L
            if Oznew+sig < L:
                Oznew  = Oznew + L
                H1znew = H1znew + L
                H2znew = H2znew + L

            # store new positions
            Ox[i] = Oxnew
            Oy[i] = Oynew
            Oz[i] = Oznew
            H1x[i] = H1xnew
            H1y[i] = H1ynew
            H1z[i] = H1znew
            H2x[i] = H2xnew
            H2y[i] = H2ynew
            H2z[i] = H2znew

            # get new energies only between moved atom to neighbors
            for j in range(N):
                if i != j:

                    # get oxygen-oxygen distances for LJ
                    dXO = Ox[i] - Ox[j]
                    dYO = Oy[i] - Oy[j]
                    dZO = Oz[i] - Oz[j]

                    # get all site distances for Coulomb
                    OO   = (dXO*dXO + dYO*dYO + dZO*dZO)**0.5
                    OH1  = ((Ox[i]-H1x[j])**2 + (Oy[i]-H1y[j])**2 + (Oz[i]-H1z[j])**2)**0.5
                    OH2  = ((Ox[i]-H2x[j])**2 + (Oy[i]-H2y[j])**2 + (Oz[i]-H2z[j])**2)**0.5
                    H1O  = ((H1x[i]-Ox[j])**2 + (H1y[i]-Oy[j])**2 + (H1z[i]-Oz[j])**2)**0.5
                    H1H1 = ((H1x[i]-H1x[j])**2 + (H1y[i]-H1y[j])**2 + (H1z[i]-H1z[j])**2)**0.5
                    H1H2 = ((H1x[i]-H2x[j])**2 + (H1y[i]-H2y[j])**2 + (H1z[i]-H2z[j])**2)**0.5
                    H2O  = ((H2x[i]-Ox[j])**2 + (H2y[i]-Oy[j])**2 + (H2z[i]-Oz[j])**2)**0.5
                    H2H1 = ((H2x[i]-H1x[j])**2 + (H2y[i]-H1y[j])**2 + (H2z[i]-H1z[j])**2)**0.5
                    H2H2 = ((H2x[i]-H2x[j])**2 + (H2y[i]-H2y[j])**2 + (H2z[i]-H2z[j])**2)**0.5

                    # get Coulomb potential
                    CoulE = (qOO/OO)+(qOH/OH1)+(qOH/OH2)+(qOH/H1O)+(qHH/H1H1)+(qHH/H1H2)+(qOH/H2O)+(qHH/H2H1)+(qHH/H2H2)

                    # implement mimimum image convention only on oxygen (LJ)
                    if dXO > Lhalf: dXO = dXO - L
                    elif dXO < -Lhalf: dXO = dXO + L
                    if dYO > Lhalf: dYO = dYO - L
                    elif dYO < -Lhalf: dYO = dYO + L
                    if dZO > Lhalf: dZO = dZO - L
                    elif dZO < -Lhalf: dZO = dZO + L
                    imgR = (dXO*dXO + dYO*dYO + dZO*dZO)**0.5

                    # implement cut & shift of potential only on oxygen (LJ)
                    if imgR > Rc:
                        LJE = 0.0

                    else:
                        imgR6 = 1.0/(imgR*imgR*imgR*imgR*imgR*imgR)
                        imgR12 = imgR6*imgR6
                        LJE = (ALJ*imgR12) - (BLJ*imgR6) - (ALJ*Rc12) + (BLJ*Rc6)

                    # write potential to energy matrix
                    E[i,j] = kc*CoulE + LJE
                    E[j,i] = 0.0
            #----------------------------------------------------

            # calculate new system energy and delta
            peNew = np.sum(E)
            delta = peNew - pe

            # accept move if new energy is lower
            if delta < 0:
                pe = peNew
                accT += 1.0

            else:
                # check second condition if new energy is not lower
                prob = np.exp(-beta*delta)
                rand = rand0[indx]

                # accept move with this probability
                if rand < prob:
                    pe = peNew
                    accT += 1.0

                # reject move otherwise (undo move)
                else:
                    Ox[i],Oy[i],Oz[i] = Oi[0], Oi[1], Oi[2]
                    H1x[i],H1y[i],H1z[i] = H1i[0], H1i[1], H1i[2]
                    H2x[i],H2y[i],H2z[i] = H2i[0], H2i[1], H2i[2]
                    E[i,:] = ERow
                    E[:,i] = ECol

            # store energy
            if run >= hlfRun:
                UT[run-hlfRun] = pe

            # update acceptance rate and increment counter
            accP = accT/(indx+1)
            indx += 1

        # store order parameter after each pass through system
        ordX = np.sum(np.cos(4*pi*Ox/a))
        ordY = np.sum(np.cos(4*pi*Oy/a))
        ordZ = np.sum(np.cos(4*pi*Oz/a))
        ord0 = (ordX + ordY + ordZ)/(3.0*N)
        Ord[run+1] = ord0

        # keep initial positions to calculate mean sq displacement
        if run == hlfRun:
            Ox0 = np.copy(Ox)
            Oy0 = np.copy(Oy)
            Oz0 = np.copy(Oz)

        # calculate MSD, running averages and variances, radial distr func
        #----------------------------------------------------
        if run >= hlfRun:
            # calculate mean square displacement
            D = np.sum((Ox-Ox0)**2 + (Oy-Oy0)**2 + (Oz-Oz0)**2)/N
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

            # concatenate H1 and H2 positions for g(r) calculation
            HHx = np.concatenate([H1x,H2x])
            HHy = np.concatenate([H1y,H2y])
            HHz = np.concatenate([H1z,H2z])

            # calculate OO g(r)
            for i in range(N):
                for j in range(N):
                    dXO = Ox[i] - Ox[j]
                    dYO = Oy[i] - Oy[j]
                    dZO = Oz[i] - Oz[j]
                    if dXO > Lhalf: dXO = dXO - L
                    elif dXO < -Lhalf: dXO = dXO + L
                    if dYO > Lhalf: dYO = dYO - L
                    elif dYO < -Lhalf: dYO = dYO + L
                    if dZO > Lhalf: dZO = dZO - L
                    elif dZO < -Lhalf: dZO = dZO + L
                    imgR = (dXO*dXO + dYO*dYO + dZO*dZO)**0.5
                    RadOO[j] = imgR
                histOO,binOO = np.histogram(RadOO,bins=GRbins+Gdr)
                GROO = GROO + histOO

            # calculate HH g(r)
            for i in range(2*N):
                for j in range(2*N):
                    dXH = HHx[i] - HHx[j]
                    dYH = HHy[i] - HHy[j]
                    dZH = HHz[i] - HHz[j]
                    if dXH > Lhalf: dXH = dXH - L
                    elif dXH < -Lhalf: dXH = dXH + L
                    if dYH > Lhalf: dYH = dYH - L
                    elif dYH < -Lhalf: dYH = dYH + L
                    if dZH > Lhalf: dZH = dZH - L
                    elif dZH < -Lhalf: dZH = dZH + L
                    imgR = (dXH*dXH + dYH*dYH + dZH*dZH)**0.5
                    RadHH[j] = imgR
                histHH,binHH = np.histogram(RadHH,bins=GRbins+Gdr)
                GRHH = GRHH + histHH

            # calculate OH g(r)
            for i in range(N):
                for j in range(2*N):
                    dXH = Ox[i] - HHx[j]
                    dYH = Oy[i] - HHy[j]
                    dZH = Oz[i] - HHz[j]
                    if dXH > Lhalf: dXH = dXH - L
                    elif dXH < -Lhalf: dXH = dXH + L
                    if dYH > Lhalf: dYH = dYH - L
                    elif dYH < -Lhalf: dYH = dYH + L
                    if dZH > Lhalf: dZH = dZH - L
                    elif dZH < -Lhalf: dZH = dZH + L
                    imgR = (dXH*dXH + dYH*dYH + dZH*dZH)**0.5
                    RadOH[j] = imgR
                histOH,binOH = np.histogram(RadOH,bins=GRbins+Gdr)
                GROH = GROH + histOH
        #----------------------------------------------------

    # store avg energy and variance for each temp
    # average only last half, first half is equilibration
    aveU = aveRunU
    varU = varRunU/(run-hlfRun+1)
    aveUarr.append(aveU)
    varUarr.append(varU)
    accParr.append(accP)

    # normalize radial distr func g(r)
    GROO = GROO / (hlfRun*N*Rho*4*pi*Gdr*GRbins[0:Nbins]**2)
    GRHH = GRHH / (hlfRun*N*Rho*18*pi*Gdr*GRbins[0:Nbins]**2)
    GROH = GROH / (hlfRun*N*Rho*8*pi*Gdr*GRbins[0:Nbins]**2)

    # calculate Cv from variance of energy
    Cv = varU / (kb*T**2)

    print 'Rho = %1.4f T = %1.2f <U> = %1.4f varU = %1.4f accP = %1.2f Cv = %1.4f' \
    % (Rho,T,aveU/N,varU,accP,Cv)

    # plot results
    #----------------------------------------------------
    # set plot axis for last half of runs
    xRuns = np.arange(hlfRun)+hlfRun
    UT = np.array(UT)/N
    aveRunUarr = aveRunUarr/N
    indxHH = np.int(1.8/Gdr) - 1
    indxOH = np.int(1.2/Gdr) - 1
    GRHH[0:indxHH] = 0.0
    GROH[0:indxOH] = 0.0

    # plot running average potential energy
    title1 = 'Energy, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
    fname1 = np.str(N)+'_RunAveU_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
    plt.figure()
    plt.subplot(211)
    plt.plot(xRuns,UT,'b.',label='$Accepted$')
    plt.plot(xRuns,aveRunUarr,'r.-')
    plt.ylabel('Average (kcal/mol)')
    plt.title(title1)
    plt.legend(loc=2)
    plt.subplot(212)
    plt.plot(xRuns,varRunUarr,'g.-')
    plt.ylabel('Variance (kcal/mol)$^{2}$')
    plt.xlabel('MMC Step')
    plt.savefig(fname1)

    # plot running average mean squared displacement
    title2 = 'MSD, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
    fname2 = np.str(N)+'_MSD_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
    plt.figure()
    plt.subplot(211)
    plt.plot(xRuns,aveRunDarr,'r.-')
    plt.ylabel('Average (angstrom$^{2}$)')
    plt.title(title2)
    plt.subplot(212)
    plt.plot(xRuns,varRunDarr,'g.-')
    plt.ylabel('Variance (angstrom$^{4}$)')
    plt.xlabel('MMC Step')
    plt.savefig(fname2)

    # plot OO radial distribution function g(r) vs radius
    title3 = '$g(r)\ O-O$, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
    fname3 = np.str(N)+'_GOO_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
    plt.figure()
    plt.plot(GRbins[0:Nbins],GROO,'b-',label='$g_{OO}$')
    plt.plot([GRbins[0],GRbins[Nbins]],[1,1],'k--')
    plt.xlabel('$r$ (angstrom)')
    plt.yticks([0,1,2,3])
    plt.ylabel('$g\ (r)$')
    plt.title(title3)
    plt.legend(loc=1)
    plt.savefig(fname3)

    # plot HH radial distribution function g(r) vs radius
    title4 = '$g(r)\ H-H$, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
    fname4 = np.str(N)+'_GHH_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
    plt.figure()
    plt.plot(GRbins[0:Nbins],GRHH,'b-',label='$g_{HH}$')
    plt.plot([GRbins[0],GRbins[Nbins]],[1,1],'k--')
    plt.xlabel('$r$ (angstrom)')
    plt.ylabel('$g\ (r)$')
    plt.yticks([0,0.5,1,1.5])
    plt.title(title4)
    plt.legend(loc=1)
    plt.savefig(fname4)

    # plot OH radial distribution function g(r) vs radius
    title5 = '$g(r)\ O-H$, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
    fname5 = np.str(N)+'_GOH_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
    plt.figure()
    plt.plot(GRbins[0:Nbins],GROH,'b-',label='$g_{OH}$')
    plt.plot([GRbins[0],GRbins[Nbins]],[1,1],'k--')
    plt.xlabel('$r$ (angstrom)')
    plt.ylabel('$g\ (r)$')
    plt.yticks([0,0.5,1,1.5,2])
    plt.title(title5)
    plt.legend(loc=1)
    plt.savefig(fname5)

    # plot FCC order parameter
    title6 = '$FCC$, '+r'$\rho$ = '+np.str(Rho)+', $T$ = '+np.str(T)+', NVT'
    fname6 = np.str(N)+'_FCC_Rho'+np.str(Rho)+'_T'+np.str(T)+'_NVT.png'
    plt.figure()
    plt.plot(range(runs+1),Ord)
    plt.ylabel('FCC Order')
    plt.xlabel('MMC Step')
    plt.title(title6)
    plt.savefig(fname6)

    # stack and store final atom positions for import to Avogadro
    #----------------------------------------------------
    molx = np.concatenate([Ox,H1x,H2x])
    moly = np.concatenate([Oy,H1y,H2y])
    molz = np.concatenate([Oz,H1z,H2z])
    stack = np.column_stack((nums,molx,moly,molz))
    np.savetxt('H2O_'+str(N)+'_FinalPos.txt',stack,fmt=('%i','%f4','%f4','%f4'))
    #----------------------------------------------------

    print 'runtime =',datetime.now()-t0